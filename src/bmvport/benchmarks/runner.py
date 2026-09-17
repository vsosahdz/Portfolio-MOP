"""Running the comparator suite into the same results table as the proposed method.

Every comparator is priced, costed, attributed and written by the code path the factorial
uses. That is what makes the confrontation a comparison rather than a rhetorical device: a
baseline evaluated under gentler assumptions is not a baseline.

Tier 0 reference series are the exception, and deliberately so. They are not investable
strategies, they carry no turnover, and charging them transaction costs would misrepresent
what they are. Their rows record that treatment so no table can imply cost parity with the
traded arms.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from ..backtest.metrics import apply_costs, realised_risk, sharpe_ratio, turnover
from ..backtest.pricing import price_portfolio
from ..backtest.results import ResultRow
from ..config import RunConfig
from ..grid import GridInputs
from ..optimization.nsga import derive_seed
from ..optimization.objectives import estimate_moments
from .tiers import (
    TIER0_SERIES,
    TIER1_METHODS,
    TIER2_METHODS,
    build_tier1_arm,
    l1_regularised_mv,
    reference_series_returns,
    single_objective_ga_sharpe,
)

__all__ = ["run_comparators"]


def _row(
    config: RunConfig, name: str, tier: str, month: str, cost_bps: float, **fields
) -> ResultRow:
    return ResultRow(
        arm_kind="comparator", month=month, labeling="", screener="", allocator=name,
        weight_cap=1.0, covariance_estimator=config.optimization.covariance_estimator,
        seed=None, cost_bps=float(cost_bps), comparator_tier=tier, **fields,
    )


def _reference_rows(config: RunConfig, inputs: GridInputs) -> list[ResultRow]:
    """Tier 0: the floor and the market, uncosted by design."""
    from ..marketdata.prices import load_prices

    ipc = load_prices(config.paths.cache, "^MXX")["close"].dropna()
    sp500 = load_prices(config.paths.cache, "^GSPC")["close"].dropna()
    from ..marketdata.rates import load_series

    fx = load_series("fx_fix", config.paths.cache)["value"]

    rows: list[ResultRow] = []
    for month in config.windows.eval_months():
        rf = float(inputs.risk_free.get(month, float("nan")))
        try:
            series = reference_series_returns(
                month, risk_free=rf, ipc_prices=ipc, sp500_prices=sp500,
                fx_prices=fx, calendar=inputs.calendar,
            )
        except ValueError as exc:
            for name in TIER0_SERIES:
                for bps in config.evaluation.cost_scenarios_bps:
                    rows.append(_row(config, name, "tier0", month, bps,
                                     skip_reason=str(exc)[:100]))
            continue
        for name, value in series.items():
            for bps in config.evaluation.cost_scenarios_bps:
                rows.append(
                    _row(
                        config, name, "tier0", month, bps,
                        # Reference series are uncosted: gross equals net, and turnover is
                        # zero rather than missing, so a table cannot imply cost parity.
                        return_gross=value, return_net=value, turnover=0.0,
                        risk_free=rf, holdings=0,
                        literature_reference="reference series, not investable; no costs",
                    )
                )
    return rows


def run_comparators(
    config: RunConfig, inputs: GridInputs, *, progress: bool = True
) -> tuple[list[ResultRow], list[dict]]:
    """Evaluate Tier 0, Tier 1 and Tier 2 under the study's own protocol."""
    rows = _reference_rows(config, inputs)
    failures: list[dict] = []
    previous: dict[str, dict[str, float]] = {}

    def evaluate(name: str, tier: str, month: str, symbols: Sequence[str],
                 weights: np.ndarray, citation: str = "") -> None:
        try:
            priced = price_portfolio(month, list(symbols), weights, inputs.prices, inputs.calendar)
        except ValueError as exc:
            for bps in config.evaluation.cost_scenarios_bps:
                rows.append(_row(config, name, tier, month, bps,
                                 skip_reason=f"pricing: {exc}"[:100]))
            return
        holdings = {s: float(w) for s, w in zip(priced.symbols, priced.weights)}
        turn = turnover(holdings, previous.get(name))
        previous[name] = holdings
        net = apply_costs(priced.gross_return, turn, config.evaluation.cost_scenarios_bps)
        sigma = realised_risk(priced.daily_returns)
        rf = float(inputs.risk_free.get(month, float("nan")))
        for bps in config.evaluation.cost_scenarios_bps:
            rows.append(
                _row(config, name, tier, month, bps,
                     return_gross=priced.gross_return, return_net=net[float(bps)],
                     sigma_realized=sigma, sharpe=sharpe_ratio(net[float(bps)], sigma, rf),
                     risk_free=rf, turnover=turn, holdings=len(holdings),
                     selection_size=len(symbols), literature_reference=citation)
            )

    for month in config.windows.eval_months():
        if progress:
            print(f"  comparadores {month}", flush=True)
        universe = [s for s in inputs.membership.get(month, ()) if s in inputs.daily_returns.columns]
        if len(universe) < 2:
            continue
        history = inputs.daily_returns[inputs.daily_returns.index < pd.Timestamp(f"{month}-01")]
        try:
            moments = estimate_moments(
                history, universe,
                trailing_days=config.optimization.covariance_trailing_days,
                estimator=config.optimization.covariance_estimator,
            )
        except ValueError as exc:
            failures.append({"month": month, "stage": "moments", "error": str(exc)[:200]})
            continue

        for name in TIER1_METHODS:
            arm = build_tier1_arm(
                name, month, universe, moments=moments,
                monthly_returns=inputs.monthly_returns,
                momentum_formation_months=config.benchmarks.momentum_formation_months,
                momentum_skip_months=config.benchmarks.momentum_skip_months,
                momentum_selection_fraction=config.benchmarks.momentum_selection_fraction,
            )
            if arm.skipped:
                for bps in config.evaluation.cost_scenarios_bps:
                    rows.append(_row(config, name, "tier1", month, bps,
                                     skip_reason=arm.skip_reason[:100]))
                continue
            evaluate(name, "tier1", month, arm.symbols, arm.weights)

        # Tier 2: literature-equivalent pipelines.
        try:
            weights = single_objective_ga_sharpe(
                moments, seed=derive_seed(config.fingerprint(), month, "l4"),
                population_size=config.optimization.population_size,
                generations=config.optimization.generations,
            )
            if weights is not None:
                evaluate("l4_single_objective_ga_sharpe", "tier2", month, universe,
                         weights, "Frausto et al. (2022); the scalarisation contrast")
        except Exception as exc:  # noqa: BLE001
            failures.append({"month": month, "arm": "l4", "error": str(exc)[:200]})

        try:
            shrunk = estimate_moments(
                history, universe,
                trailing_days=config.optimization.covariance_trailing_days,
                estimator=config.benchmarks.sensitivity_estimator,
            )
            weights = l1_regularised_mv(shrunk)
            if weights is not None:
                evaluate("l6_l1_shrinkage_mv", "tier2", month, universe, weights,
                         "Dai and Kang (2021); L1 with shrinkage covariance")
        except Exception as exc:  # noqa: BLE001
            failures.append({"month": month, "arm": "l6", "error": str(exc)[:200]})

    return rows, failures
