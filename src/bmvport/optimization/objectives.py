"""Bi-objective mean-variance formulation, evaluated over a whole population at once.

Two objectives, both minimised: negative expected portfolio return and portfolio standard
deviation. Both come from an **ex-ante** estimate -- trailing daily returns ending on the
last session before the evaluation month -- and the study keeps that quantity strictly
separate from the risk a portfolio actually realises during the month it is held. The
original work uses one word for both; here they never share a name.

**Why the evaluation is vectorised.** The grid runs 7,560 NSGA-III searches of 100
individuals over 100 generations, which is 7.56 million objective evaluations. Each needs a
quadratic form over a covariance matrix of up to ~100 assets. Evaluated one individual at a
time in Python that is hours of interpreter overhead; evaluated as one matrix product per
generation it is a rounding error. The population form is not an optimisation, it is what
makes the design runnable.

**The budget constraint is enforced by renormalisation**, matching the original's
substitution of its own sum-to-one constraint: weights are divided by their sum after
variation rather than being constrained during it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "Moments",
    "estimate_moments",
    "renormalise",
    "apply_cap",
    "portfolio_objectives",
    "cap_is_feasible",
]

# Weight vectors summing below this are treated as degenerate rather than renormalised: the
# division would amplify numerical noise into a portfolio.
_DEGENERATE_SUM = 1e-12


@dataclass(frozen=True, slots=True)
class Moments:
    """Ex-ante expected returns and covariance, with the provenance to report them."""

    symbols: tuple[str, ...]
    expected_returns: np.ndarray
    covariance: np.ndarray
    trailing_days: int
    estimator: str
    window_start: str
    window_end: str
    regularisation: float = 0.0

    def __post_init__(self) -> None:
        n = len(self.symbols)
        if self.expected_returns.shape != (n,):
            raise ValueError("expected_returns does not match the symbol count")
        if self.covariance.shape != (n, n):
            raise ValueError("covariance does not match the symbol count")

    def validate(self, *, tolerance: float = 1e-8) -> None:
        """Check the covariance is symmetric and positive semi-definite.

        A covariance that is neither is not a small numerical annoyance: the risk objective
        would admit negative variances, and the optimiser would happily exploit them.

        Raises:
            ValueError: naming which property failed and by how much.
        """
        asymmetry = float(np.max(np.abs(self.covariance - self.covariance.T)))
        if asymmetry > tolerance:
            raise ValueError(f"covariance is not symmetric (max asymmetry {asymmetry:.2e})")
        eigenvalues = np.linalg.eigvalsh(self.covariance)
        smallest = float(eigenvalues.min())
        if smallest < -tolerance:
            raise ValueError(
                f"covariance is not positive semi-definite (smallest eigenvalue "
                f"{smallest:.2e}); the risk objective would admit negative variance"
            )


def estimate_moments(
    returns: pd.DataFrame,
    symbols: list[str],
    *,
    trailing_days: int,
    estimator: str = "sample",
    regularisation: float = 0.0,
) -> Moments:
    """Estimate expected returns and covariance from trailing daily returns.

    Args:
        returns: Daily returns indexed by date, one column per symbol. The caller has
            already truncated it to information available at the decision date.
        symbols: Assets to include, in the order the weights will use.
        trailing_days: Sessions of history to use, taken from the end.
        estimator: ``"sample"`` or ``"ledoit_wolf"``. The shrinkage estimator exists for the
            sensitivity arm; the factorial uses the sample form, faithful to the original
            mean-variance formulation.
        regularisation: Ridge added to the diagonal. Recorded rather than silent, because it
            changes the risk objective.

    Raises:
        ValueError: if fewer than two sessions remain, or a requested symbol is absent.
    """
    missing = [s for s in symbols if s not in returns.columns]
    if missing:
        raise ValueError(f"returns are missing symbols: {missing[:5]}")

    window = returns[symbols].tail(trailing_days).dropna(how="all")
    if len(window) < 2:
        raise ValueError(
            f"only {len(window)} usable sessions in the trailing window; a covariance "
            "cannot be estimated"
        )
    # Assets missing a day are filled with zero return rather than dropping the whole
    # session for every asset. The universe screen has already removed instruments where
    # this would be frequent.
    filled = window.fillna(0.0)
    values = filled.to_numpy(dtype=float)

    expected = values.mean(axis=0)
    if estimator == "sample":
        covariance = np.cov(values, rowvar=False, ddof=1)
    elif estimator == "ledoit_wolf":
        from sklearn.covariance import LedoitWolf

        covariance = LedoitWolf().fit(values).covariance_
    else:
        raise ValueError(f"unknown covariance estimator {estimator!r}")

    covariance = np.atleast_2d(np.asarray(covariance, dtype=float))
    if regularisation:
        covariance = covariance + regularisation * np.eye(len(symbols))
    # Symmetrise against floating-point drift before the validity check sees it.
    covariance = (covariance + covariance.T) / 2.0

    return Moments(
        symbols=tuple(symbols),
        expected_returns=np.asarray(expected, dtype=float),
        covariance=covariance,
        trailing_days=int(trailing_days),
        estimator=estimator,
        window_start=str(filled.index.min().date()),
        window_end=str(filled.index.max().date()),
        regularisation=float(regularisation),
    )


def renormalise(weights: np.ndarray) -> np.ndarray:
    """Divide each weight vector by its sum, repairing degenerate rows.

    A row summing to (numerically) zero carries no information about allocation, and
    dividing by that sum would turn floating-point noise into a portfolio. Such rows become
    equal weight instead -- a documented repair, not a silent one.
    """
    matrix = np.atleast_2d(np.asarray(weights, dtype=float))
    matrix = np.clip(matrix, 0.0, None)
    totals = matrix.sum(axis=1, keepdims=True)
    degenerate = (totals <= _DEGENERATE_SUM).ravel()
    safe = np.where(totals <= _DEGENERATE_SUM, 1.0, totals)
    out = matrix / safe
    if degenerate.any():
        out[degenerate] = 1.0 / matrix.shape[1]
    return out


def cap_is_feasible(n_assets: int, cap: float) -> bool:
    """Whether a per-asset cap admits any budget-feasible portfolio at all."""
    return n_assets * cap >= 1.0 - 1e-12


def apply_cap(weights: np.ndarray, cap: float, *, max_iterations: int = 100) -> np.ndarray:
    """Project weights onto ``{w : 0 <= w <= cap, sum(w) = 1}``.

    Excess above the cap is redistributed proportionally among the assets still below it,
    repeating until nothing exceeds the cap. Clipping alone would break the budget
    constraint, and rescaling alone would reintroduce breaches.

    Raises:
        ValueError: if the cap makes the budget constraint unsatisfiable.
    """
    matrix = renormalise(weights)
    n = matrix.shape[1]
    if cap >= 1.0:
        return matrix
    if not cap_is_feasible(n, cap):
        raise ValueError(
            f"a cap of {cap} over {n} assets cannot reach a total weight of one; the cell "
            "is infeasible and must be recorded as such rather than silently rescaled"
        )

    out = matrix.copy()
    for _ in range(max_iterations):
        excess = np.clip(out - cap, 0.0, None)
        total_excess = excess.sum(axis=1)
        if not np.any(total_excess > 1e-12):
            break
        out = np.minimum(out, cap)
        headroom = np.clip(cap - out, 0.0, None)
        capacity = headroom.sum(axis=1, keepdims=True)
        share = np.divide(
            headroom, capacity, out=np.zeros_like(headroom), where=capacity > 1e-12
        )
        out = out + share * total_excess[:, None]
    return np.minimum(out, cap)


def portfolio_objectives(weights: np.ndarray, moments: Moments) -> np.ndarray:
    """Evaluate both objectives for a whole population.

    Args:
        weights: ``(population, n_assets)`` or a single ``(n_assets,)`` vector. Assumed
            already renormalised and capped by the caller.
        moments: Ex-ante estimates.

    Returns:
        ``(population, 2)`` array of ``[-expected_return, standard_deviation]``, both to be
        minimised.
    """
    matrix = np.atleast_2d(np.asarray(weights, dtype=float))
    expected = matrix @ moments.expected_returns
    # einsum keeps the quadratic form to one pass without forming the full outer product.
    variance = np.einsum("ij,jk,ik->i", matrix, moments.covariance, matrix)
    # Tiny negatives arise from floating point on a semi-definite matrix, not from a real
    # negative variance; the validity check has already excluded the latter.
    variance = np.clip(variance, 0.0, None)
    return np.column_stack((-expected, np.sqrt(variance)))
