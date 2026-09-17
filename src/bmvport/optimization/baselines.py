"""Deterministic allocators: equal weight, minimum volatility, maximum Sharpe.

These are what make the genetic arms falsifiable inside the factorial. Equal weighting in
particular is a notoriously hard baseline -- it was the strongest Sharpe performer in the
original study -- and beating it is a real result rather than a formality.

All three are deterministic: the same inputs give the same weights, no seed is recorded, and
re-running changes nothing. When a convex solve fails to converge the cell is recorded as a
failure rather than quietly receiving a fallback portfolio, because a substituted allocation
that nobody notices is indistinguishable in the results table from one that succeeded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .objectives import Moments, apply_cap, cap_is_feasible

__all__ = ["AllocationResult", "equal_weight", "minimum_volatility", "maximum_sharpe",
           "ANALYTIC_ALLOCATORS"]

ANALYTIC_ALLOCATORS = ("equal_weight", "min_volatility", "max_sharpe")


@dataclass(frozen=True, slots=True)
class AllocationResult:
    """Weights from a deterministic allocator, or the reason there are none."""

    allocator: str
    weights: np.ndarray | None
    converged: bool
    message: str = ""

    @property
    def failed(self) -> bool:
        return self.weights is None


def _bounds(n: int, cap: float) -> list[tuple[float, float]]:
    return [(0.0, min(cap, 1.0))] * n


def _budget_constraint() -> dict:
    return {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}


def _start(n: int, cap: float) -> np.ndarray:
    return np.full(n, min(1.0 / n, cap))


def equal_weight(moments: Moments, *, cap: float = 1.0) -> AllocationResult:
    """Equal weight over the selection, projected onto the cap if one is in force."""
    n = len(moments.symbols)
    if n == 0:
        return AllocationResult("equal_weight", None, False, "empty selection")
    if not cap_is_feasible(n, cap):
        return AllocationResult("equal_weight", None, False, f"cap {cap} infeasible for {n}")
    weights = np.full(n, 1.0 / n)
    if cap < 1.0:
        weights = apply_cap(weights[None, :], cap)[0]
    return AllocationResult("equal_weight", weights, True)


def minimum_volatility(moments: Moments, *, cap: float = 1.0) -> AllocationResult:
    """Minimise portfolio variance subject to the budget constraint and the cap."""
    n = len(moments.symbols)
    if n == 0:
        return AllocationResult("min_volatility", None, False, "empty selection")
    if not cap_is_feasible(n, cap):
        return AllocationResult("min_volatility", None, False, f"cap {cap} infeasible for {n}")

    covariance = moments.covariance

    def variance(w: np.ndarray) -> float:
        return float(w @ covariance @ w)

    def gradient(w: np.ndarray) -> np.ndarray:
        return 2.0 * covariance @ w

    result = minimize(
        variance, _start(n, cap), jac=gradient, method="SLSQP",
        bounds=_bounds(n, cap), constraints=[_budget_constraint()],
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not result.success:
        return AllocationResult("min_volatility", None, False, str(result.message)[:200])
    return AllocationResult("min_volatility", _clean(result.x, cap), True)


def maximum_sharpe(
    moments: Moments, *, risk_free: float = 0.0, cap: float = 1.0
) -> AllocationResult:
    """Maximise the ex-ante Sharpe ratio subject to the budget constraint and the cap.

    The objective is maximised by minimising its negative. A portfolio whose variance
    collapses to zero would send the ratio to infinity, so the denominator is floored and
    the floor is part of the recorded behaviour rather than an accident of arithmetic.
    """
    n = len(moments.symbols)
    if n == 0:
        return AllocationResult("max_sharpe", None, False, "empty selection")
    if not cap_is_feasible(n, cap):
        return AllocationResult("max_sharpe", None, False, f"cap {cap} infeasible for {n}")

    mu = moments.expected_returns
    covariance = moments.covariance

    def negative_sharpe(w: np.ndarray) -> float:
        excess = float(w @ mu) - risk_free
        sigma = float(np.sqrt(max(w @ covariance @ w, 1e-18)))
        return -excess / sigma

    result = minimize(
        negative_sharpe, _start(n, cap), method="SLSQP",
        bounds=_bounds(n, cap), constraints=[_budget_constraint()],
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not result.success:
        return AllocationResult("max_sharpe", None, False, str(result.message)[:200])
    return AllocationResult("max_sharpe", _clean(result.x, cap), True)


def _clean(weights: np.ndarray, cap: float) -> np.ndarray:
    """Repair the small bound and budget violations SLSQP leaves behind."""
    w = np.clip(np.asarray(weights, dtype=float), 0.0, None)
    total = w.sum()
    w = w / total if total > 1e-12 else np.full(len(w), 1.0 / len(w))
    if cap < 1.0:
        w = apply_cap(w[None, :], cap)[0]
    return w
