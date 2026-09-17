"""The three-tier comparator suite.

Beating our own convex allocators would only show that the genetic search allocates well on
inputs we produced. The claim the paper wants to make is about approaches that exist
independently of it, so every comparator runs on **the same universe, the same months, the
same executable pricing and the same cost scenarios** as the proposed method.

``Tier 0`` reference series
    CETES 28-day, the S&P/BMV IPC, and the S&P 500 expressed in pesos. Not investable
    strategies and not charged transaction costs; they are the floor and the market.

``Tier 1`` deterministic baselines, no machine learning
    Analytic minimum variance over the whole universe, equal weighting over the whole
    universe, inverse volatility, and cross-sectional momentum. **These decide whether the
    paper has a finding at all.** If the entire screening apparatus cannot beat "hold
    everything equally" or "buy last year's winners", that is the result, and it is cheaper
    to learn before tuning screeners than after.

``Tier 2`` literature-equivalent pipelines
    Approaches from the reviewed work, each recorded with the citation it represents. The
    single-objective Sharpe-maximising search is the one the thesis claims to beat and never
    runs; without it that claim has no support.

Cells of the proposed factorial that coincide with a published approach are **labelled**
rather than reimplemented, and the mapping is stored as data so the equivalence is a
checkable claim instead of a sentence in the related-work section.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ..optimization.baselines import minimum_volatility
from ..optimization.objectives import Moments, apply_cap, cap_is_feasible, estimate_moments

__all__ = [
    "ComparatorArm",
    "TIER0_SERIES",
    "TIER1_METHODS",
    "TIER2_METHODS",
    "LITERATURE_MAPPING",
    "equal_weight_universe",
    "inverse_volatility",
    "momentum_selection",
    "mv_analytic_universe",
    "reference_series_returns",
    "single_objective_ga_sharpe",
    "regression_expected_returns",
    "l1_regularised_mv",
    "build_tier1_arm",
]

TIER0_SERIES = ("cetes_28d", "bmv_ipc", "sp500_mxn")
TIER1_METHODS = (
    "mv_analytic_universe",
    "equal_weight_universe",
    "inverse_volatility",
    "momentum_equal_weight",
)
TIER2_METHODS = (
    "l4_single_objective_ga_sharpe",
    "l5_regression_mv",
    "l6_l1_shrinkage_mv",
)

# Cells of the proposed factorial that coincide with a published approach. Stored as data so
# the related-work claim can be checked against the results rather than asserted in prose.
# These are NOT reimplemented as separate comparators; a test enforces that.
LITERATURE_MAPPING: dict[tuple[str, str], dict[str, str]] = {
    ("svm", "min_volatility"): {
        "citation": "Paiva et al. (2019)",
        "equivalence": "SVM screening followed by analytic mean-variance allocation",
        "difference": "monthly rather than daily rebalancing; costs applied",
    },
    ("svm", "ga_max_return"): {
        "citation": "Gupta et al. (2011)",
        "equivalence": "SVM classification followed by a genetic allocator",
        "difference": "NSGA-III with two objectives rather than a scalarised GA",
    },
    ("all_stocks", "ga_knee"): {
        "citation": "Silva et al. (2014)",
        "equivalence": "multi-objective evolutionary allocation without a screening stage",
        "difference": "knee selection rather than an arbitrary front index",
    },
}


@dataclass(frozen=True, slots=True)
class ComparatorArm:
    """One comparator's weights for one month, or the reason it produced none."""

    name: str
    tier: str
    month: str
    symbols: tuple[str, ...]
    weights: np.ndarray | None
    citation: str = ""
    adaptation: str = ""
    skip_reason: str = ""

    @property
    def skipped(self) -> bool:
        return self.weights is None


def reference_series_returns(
    month: str,
    *,
    risk_free: float,
    ipc_prices: pd.Series,
    sp500_prices: pd.Series,
    fx_prices: pd.Series,
    calendar: pd.DatetimeIndex,
) -> dict[str, float]:
    """Monthly returns of the Tier 0 reference series.

    The foreign index is converted to pesos, because a peso investor's outcome from holding
    it includes the currency move. Reporting it in dollars alongside peso-denominated
    portfolios would compare two different questions.
    """
    from ..backtest.pricing import month_bounds

    first, last = month_bounds(calendar, month)

    def total_return(series: pd.Series) -> float:
        window = series[(series.index >= first) & (series.index <= last)].dropna()
        if len(window) < 2:
            return float("nan")
        return float(window.iloc[-1] / window.iloc[0] - 1.0)

    ipc = total_return(ipc_prices)
    sp500 = total_return(sp500_prices)
    fx = total_return(fx_prices)
    # Compounded, not added: a peso investor earns the product of the two.
    sp500_mxn = (1.0 + sp500) * (1.0 + fx) - 1.0 if np.isfinite(sp500 + fx) else float("nan")
    return {"cetes_28d": risk_free, "bmv_ipc": ipc, "sp500_mxn": sp500_mxn}


def equal_weight_universe(symbols: Sequence[str], *, cap: float = 1.0) -> np.ndarray:
    """1/N over the whole universe: the baseline that is famously hard to beat."""
    n = len(symbols)
    weights = np.full(n, 1.0 / n)
    return apply_cap(weights[None, :], cap)[0] if cap < 1.0 else weights


def inverse_volatility(moments: Moments, *, cap: float = 1.0) -> np.ndarray:
    """Weights inversely proportional to each asset's own volatility.

    A risk-parity-style rule that needs no covariance inversion, and therefore none of the
    estimation error that makes minimum variance fragile on short samples.
    """
    sigma = np.sqrt(np.clip(np.diag(moments.covariance), 1e-18, None))
    raw = 1.0 / sigma
    weights = raw / raw.sum()
    return apply_cap(weights[None, :], cap)[0] if cap < 1.0 else weights


def momentum_selection(
    monthly_returns: pd.DataFrame,
    month: str,
    eligible: Sequence[str],
    *,
    formation_months: int,
    skip_months: int,
    selection_fraction: float,
) -> list[str]:
    """Top-fraction cross-sectional momentum selection.

    A **quintile**, not the conventional decile, and the manuscript declares the deviation.
    The decile convention assumes universes of hundreds of names; on ninety instruments it
    would hold nine positions, and a baseline that concentrated loses on idiosyncratic
    variance rather than on the absence of momentum. Losing to an unfairly noisy baseline
    misleads exactly as much as losing to none.

    **Ranking is restricted to the month's admitted universe.** Scoring across every
    instrument with price history would let the baseline hold names the universe screen
    excluded, which would break the condition that every comparator runs under identical
    conditions -- and would hand it an advantage the proposed method never had.

    Returns an empty list when the formation window is not fully available, rather than
    scoring on a partial one.
    """
    months = sorted(m for m in monthly_returns.index if m < month)
    needed = formation_months + skip_months
    if len(months) < needed:
        return []
    window = months[-needed : len(months) - skip_months] if skip_months else months[-needed:]
    columns = [s for s in eligible if s in monthly_returns.columns]
    if not columns:
        return []
    formation = monthly_returns.loc[window, columns]
    cumulative = (1.0 + formation).prod() - 1.0
    ranked = cumulative.dropna().sort_values(ascending=False)
    if ranked.empty:
        return []
    count = max(int(round(selection_fraction * len(ranked))), 1)
    return ranked.index[:count].tolist()


def mv_analytic_universe(moments: Moments, *, cap: float = 1.0) -> np.ndarray | None:
    """Classical Markowitz minimum variance over the whole universe, no screening."""
    result = minimum_volatility(moments, cap=cap)
    return result.weights if result.converged else None


def build_tier1_arm(
    name: str,
    month: str,
    symbols: Sequence[str],
    *,
    moments: Moments | None = None,
    monthly_returns: pd.DataFrame | None = None,
    cap: float = 1.0,
    momentum_formation_months: int = 12,
    momentum_skip_months: int = 1,
    momentum_selection_fraction: float = 0.20,
) -> ComparatorArm:
    """Construct one Tier 1 baseline for one month."""

    def arm(weights: np.ndarray | None, used: Sequence[str], reason: str = "") -> ComparatorArm:
        return ComparatorArm(
            name=name, tier="tier1", month=month, symbols=tuple(used),
            weights=weights, skip_reason=reason,
        )

    if not symbols:
        return arm(None, (), "empty universe")
    if not cap_is_feasible(len(symbols), cap):
        return arm(None, (), f"cap {cap} infeasible for {len(symbols)} assets")

    if name == "equal_weight_universe":
        return arm(equal_weight_universe(symbols, cap=cap), symbols)
    if name == "inverse_volatility":
        if moments is None:
            return arm(None, (), "moments unavailable")
        return arm(inverse_volatility(moments, cap=cap), symbols)
    if name == "mv_analytic_universe":
        if moments is None:
            return arm(None, (), "moments unavailable")
        weights = mv_analytic_universe(moments, cap=cap)
        return arm(weights, symbols, "" if weights is not None else "convex solve failed")
    if name == "momentum_equal_weight":
        if monthly_returns is None:
            return arm(None, (), "monthly returns unavailable")
        picked = momentum_selection(
            monthly_returns, month, symbols,
            formation_months=momentum_formation_months,
            skip_months=momentum_skip_months,
            selection_fraction=momentum_selection_fraction,
        )
        if not picked:
            return arm(None, (), "formation window unavailable")
        return arm(equal_weight_universe(picked, cap=cap), picked)
    raise ValueError(f"unknown Tier 1 method {name!r}")


# --- Tier 2: literature-equivalent pipelines -------------------------------------------


def single_objective_ga_sharpe(
    moments: Moments,
    *,
    seed: int,
    population_size: int = 100,
    generations: int = 100,
    cap: float = 1.0,
    risk_free: float = 0.0,
) -> np.ndarray | None:
    """Genetic search maximising the Sharpe ratio as a single scalar objective.

    Represents Frausto et al. (2022) and, more importantly, supplies the contrast the
    original study asserts but never runs: it concludes that a multi-objective formulation
    beat collapsing return and risk into one number, without ever optimising that one
    number. Without this arm that conclusion has no support.

    Deliberately shares the population size, generation count, seeding discipline and weight
    handling of the multi-objective arm, so the comparison isolates the *formulation* rather
    than confounding it with search effort.
    """
    from pymoo.algorithms.soo.nonconvex.ga import GA
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize as pymoo_minimize

    n = len(moments.symbols)
    if n == 0 or not cap_is_feasible(n, cap):
        return None

    class _Scalarised(Problem):
        def __init__(self) -> None:
            super().__init__(n_var=n, n_obj=1, xl=0.0, xu=1.0)

        def _evaluate(self, X, out, *args, **kwargs):  # noqa: ANN001
            weights = apply_cap(X, cap) if cap < 1.0 else _renormalise(X)
            excess = weights @ moments.expected_returns - risk_free
            variance = np.einsum("ij,jk,ik->i", weights, moments.covariance, weights)
            sigma = np.sqrt(np.clip(variance, 1e-18, None))
            out["F"] = (-excess / sigma).reshape(-1, 1)

    result = pymoo_minimize(
        _Scalarised(), GA(pop_size=population_size), ("n_gen", generations),
        seed=seed, verbose=False,
    )
    raw = np.atleast_2d(result.X)
    weights = apply_cap(raw, cap) if cap < 1.0 else _renormalise(raw)
    return weights[0]


def regression_expected_returns(
    features: pd.DataFrame,
    target: pd.Series,
    predict_features: pd.DataFrame,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Predict next-month returns with a random forest, for use as expected returns.

    Represents Ma et al. (2021), which uses regression rather than classification to improve
    the optimiser's inputs. It is a genuinely different family from the study's screening
    approach: it changes *what the optimiser is told* instead of *which assets it is given*.
    """
    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=1)
    model.fit(features.to_numpy(float), target.to_numpy(float))
    return np.asarray(model.predict(predict_features.to_numpy(float)), dtype=float)


def l1_regularised_mv(
    moments: Moments, *, l1_penalty: float = 1e-3, cap: float = 1.0
) -> np.ndarray | None:
    """Minimum variance with an L1 penalty, on a shrinkage covariance estimate.

    Represents Dai and Kang (2021). The penalty produces sparse portfolios and the shrinkage
    estimator addresses the standard objection that sample covariance is fragile on short
    samples -- an objection this study's own gate result supports, since analytic minimum
    variance on sample covariance produced the worst risk-adjusted outcome of the Tier 1 set.

    The caller supplies moments already estimated with the shrinkage estimator.
    """
    from scipy.optimize import minimize as scipy_minimize

    n = len(moments.symbols)
    if n == 0 or not cap_is_feasible(n, cap):
        return None

    covariance = moments.covariance

    def objective(w: np.ndarray) -> float:
        return float(w @ covariance @ w + l1_penalty * np.abs(w).sum())

    result = scipy_minimize(
        objective, np.full(n, min(1.0 / n, cap)), method="SLSQP",
        bounds=[(0.0, min(cap, 1.0))] * n,
        constraints=[{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}],
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not result.success:
        return None
    weights = np.clip(result.x, 0.0, None)
    total = weights.sum()
    weights = weights / total if total > 1e-12 else np.full(n, 1.0 / n)
    return apply_cap(weights[None, :], cap)[0] if cap < 1.0 else weights


def _renormalise(X: np.ndarray) -> np.ndarray:
    from ..optimization.objectives import renormalise

    return renormalise(X)
