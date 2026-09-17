"""Grid orchestration: the one place every stage is composed into results.

Each cell of the factorial is (labeling, screener, allocator, cap, seed, month), and each
produces one row per cost scenario. The orchestrator is deliberately the only component that
knows the whole pipeline; every stage below it stays independently testable, which is what
made it possible to find the label-availability lag, the momentum universe leak and the
degenerate-cap arithmetic before anything ran end to end.

**Failures become rows, not gaps.** A cell that cannot produce a portfolio -- an empty
selection, an infeasible cap, a screener that raised, a month with no priceable holding --
is written with its reason. Dropping it would make the grid look complete and would silently
change the denominator of every aggregate computed over it.

**Turnover is tracked per configuration.** Each (labeling, screener, allocator, cap, seed)
carries its own previous holdings, because turnover is a property of a strategy's own
rebalancing, not of the grid's iteration order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .backtest.attribution import attribute_return
from .backtest.metrics import apply_costs, realised_risk, sharpe_ratio, turnover
from .backtest.pricing import price_portfolio
from .backtest.results import ResultRow
from .config import RunConfig
from .optimization.baselines import equal_weight, maximum_sharpe, minimum_volatility
from .optimization.front import cap_binding, extract_portfolios, hypervolume
from .optimization.nsga import derive_seed, run_nsga3
from .optimization.objectives import Moments, cap_is_feasible, estimate_moments
from .screening.selection import run_screening

__all__ = ["GridInputs", "run_grid", "load_grid_inputs"]


@dataclass(frozen=True, slots=True)
class GridInputs:
    """Everything the grid reads, loaded once and shared across cells."""

    features: pd.DataFrame
    prices: Mapping[str, pd.Series]
    daily_returns: pd.DataFrame
    calendar: pd.DatetimeIndex
    membership: Mapping[str, Sequence[str]]
    monthly_returns: pd.DataFrame
    risk_free: Mapping[str, float]
    fx_returns: Mapping[str, float]
    sic_symbols: frozenset[str]


def _skip_row(
    config: RunConfig, month: str, labeling: str, screener: str, allocator: str,
    cap: float, seed: int | None, reason: str,
) -> list[ResultRow]:
    """One skipped row per cost scenario, so the gap is visible at every scenario."""
    return [
        ResultRow(
            arm_kind="proposed", month=month, labeling=labeling, screener=screener,
            allocator=allocator, weight_cap=cap,
            covariance_estimator=config.optimization.covariance_estimator,
            seed=seed, cost_bps=float(bps), skip_reason=reason,
        )
        for bps in config.evaluation.cost_scenarios_bps
    ]


def _evaluate_portfolio(
    config: RunConfig, inputs: GridInputs, month: str, labeling: str, screener: str,
    allocator: str, cap: float, seed: int | None, symbols: Sequence[str],
    weights: np.ndarray, *, sigma_ex_ante: float, selection_size: int,
    previous: Mapping[str, float] | None, front_size: int = 0, hv: float = float("nan"),
    binding: bool | None = None,
) -> tuple[list[ResultRow], dict[str, float] | None]:
    """Price, cost, attribute and emit rows for one portfolio-month."""
    try:
        priced = price_portfolio(month, list(symbols), weights, inputs.prices, inputs.calendar)
    except ValueError as exc:
        return _skip_row(config, month, labeling, screener, allocator, cap, seed,
                         f"pricing failed: {exc}"[:120]), None

    holdings = {s: float(w) for s, w in zip(priced.symbols, priced.weights)}
    turn = turnover(holdings, previous)
    net = apply_costs(priced.gross_return, turn, config.evaluation.cost_scenarios_bps)
    sigma = realised_risk(priced.daily_returns)
    rf = float(inputs.risk_free.get(month, float("nan")))

    universe = [s for s in inputs.membership.get(month, ()) if s in inputs.monthly_returns.columns]
    universe_returns = {
        s: float(inputs.monthly_returns.at[month, s])
        for s in universe
        if month in inputs.monthly_returns.index and np.isfinite(inputs.monthly_returns.at[month, s])
    }
    selected_returns = {
        s: float(inputs.monthly_returns.at[month, s])
        for s in symbols
        if s in inputs.monthly_returns.columns and month in inputs.monthly_returns.index
        and np.isfinite(inputs.monthly_returns.at[month, s])
    }
    attribution = None
    if universe_returns:
        attribution = attribute_return(
            month, gross_return=priced.gross_return, portfolio_weights=holdings,
            selected_returns=selected_returns, universe_returns=universe_returns,
            sic_symbols=sorted(inputs.sic_symbols), fx_return=inputs.fx_returns.get(month, 0.0),
        )

    rows = []
    for bps in config.evaluation.cost_scenarios_bps:
        rows.append(
            ResultRow(
                arm_kind="proposed", month=month, labeling=labeling, screener=screener,
                allocator=allocator, weight_cap=cap,
                covariance_estimator=config.optimization.covariance_estimator,
                seed=seed, cost_bps=float(bps),
                return_gross=priced.gross_return, return_net=net[float(bps)],
                sigma_ex_ante=sigma_ex_ante, sigma_realized=sigma,
                sharpe=sharpe_ratio(net[float(bps)], sigma, rf), risk_free=rf,
                turnover=turn, holdings=len(holdings), selection_size=selection_size,
                attribution_market=attribution.market if attribution else float("nan"),
                attribution_selection=attribution.selection if attribution else float("nan"),
                attribution_allocation=attribution.allocation if attribution else float("nan"),
                attribution_fx_tilt=attribution.fx_tilt if attribution else float("nan"),
                attribution_residual=attribution.residual if attribution else float("nan"),
                portfolio_sic_weight=(
                    attribution.portfolio_sic_weight if attribution else float("nan")
                ),
                universe_sic_share=(
                    attribution.universe_sic_share if attribution else float("nan")
                ),
                cap_binding=binding, front_size=front_size, hypervolume=hv,
                dropped_at_entry=len(priced.dropped_at_entry),
                early_exits=len(priced.early_exits),
            )
        )
    return rows, holdings


def run_grid(
    config: RunConfig,
    inputs: GridInputs,
    *,
    labelings: Sequence[str] | None = None,
    screeners: Sequence[str] | None = None,
    progress: bool = True,
) -> tuple[list[ResultRow], list[dict]]:
    """Execute the factorial and return result rows plus a failure log."""
    labelings = list(labelings or config.screening.labeling_strategies)
    screeners = list(screeners or config.screening.screeners)
    rows: list[ResultRow] = []
    failures: list[dict] = []
    # Turnover is per strategy, so each configuration carries its own previous holdings.
    previous: dict[tuple, dict[str, float]] = {}

    for labeling in labelings:
        for screener in screeners:
            if progress:
                print(f"  {labeling} / {screener}", flush=True)
            screening = run_screening(
                inputs.features, config, labeling=labeling, screener=screener,
                membership=inputs.membership,
            )
            failures.extend(screening.failures)

            for selection in screening.selections:
                month = selection.month
                if selection.skipped:
                    for allocator in ("ga_min_risk", "ga_knee", "ga_max_return",
                                      "equal_weight", "min_volatility", "max_sharpe"):
                        for cap in config.optimization.weight_caps:
                            rows.extend(_skip_row(config, month, labeling, screener,
                                                  allocator, cap, None, selection.skipped))
                    continue

                symbols = list(selection.selected)
                history = inputs.daily_returns[inputs.daily_returns.index < pd.Timestamp(f"{month}-01")]
                usable = [s for s in symbols if s in history.columns]
                if len(usable) < 2:
                    for allocator in ("ga_min_risk", "ga_knee", "ga_max_return",
                                      "equal_weight", "min_volatility", "max_sharpe"):
                        for cap in config.optimization.weight_caps:
                            rows.extend(_skip_row(config, month, labeling, screener, allocator,
                                                  cap, None, "fewer than two priceable names"))
                    continue

                try:
                    moments = estimate_moments(
                        history, usable,
                        trailing_days=config.optimization.covariance_trailing_days,
                        estimator=config.optimization.covariance_estimator,
                    )
                    moments.validate()
                except ValueError as exc:
                    failures.append({"month": month, "labeling": labeling,
                                     "screener": screener, "error": str(exc)[:200]})
                    for allocator in ("ga_min_risk", "ga_knee", "ga_max_return",
                                      "equal_weight", "min_volatility", "max_sharpe"):
                        for cap in config.optimization.weight_caps:
                            rows.extend(_skip_row(config, month, labeling, screener, allocator,
                                                  cap, None, f"moments: {exc}"[:100]))
                    continue

                unconstrained_weights: np.ndarray | None = None
                for cap in config.optimization.weight_caps:
                    if not cap_is_feasible(len(usable), cap):
                        for allocator in ("ga_min_risk", "ga_knee", "ga_max_return",
                                          "equal_weight", "min_volatility", "max_sharpe"):
                            rows.extend(_skip_row(config, month, labeling, screener, allocator,
                                                  cap, None, f"cap {cap} infeasible"))
                        continue

                    for replicate in range(config.optimization.seeds):
                        seed = derive_seed(
                            config.fingerprint(), month, labeling, screener, cap, replicate
                        )
                        try:
                            front, _ = run_nsga3(
                                moments, seed=seed,
                                population_size=config.optimization.population_size,
                                generations=config.optimization.generations, cap=cap,
                            )
                        except Exception as exc:  # noqa: BLE001
                            failures.append({"month": month, "labeling": labeling,
                                             "screener": screener, "cap": cap,
                                             "error": f"{type(exc).__name__}: {exc}"[:200]})
                            continue
                        if cap >= 1.0 and replicate == 0:
                            unconstrained_weights = front.weights
                        binding = (
                            cap_binding(unconstrained_weights, cap)
                            if unconstrained_weights is not None and cap < 1.0 else None
                        )
                        hv = hypervolume(front.objectives, config.optimization.hypervolume_reference)

                        portfolios = extract_portfolios(front.weights, front.objectives)
                        for extracted in portfolios:
                            key = (labeling, screener, extracted.allocator, cap, replicate)
                            emitted, holdings = _evaluate_portfolio(
                                config, inputs, month, labeling, screener,
                                extracted.allocator, cap, seed, usable, extracted.weights,
                                sigma_ex_ante=extracted.sigma_ex_ante,
                                selection_size=len(usable), previous=previous.get(key),
                                front_size=front.size, hv=hv, binding=binding,
                            )
                            rows.extend(emitted)
                            if holdings is not None:
                                previous[key] = holdings

                        # A degenerate front yields fewer than three portfolios: with five
                        # names and a 0.20 cap the feasible set is the single equal-weight
                        # point, so there is no knee to take. That is a real property of the
                        # instance, but leaving the cell simply absent makes it
                        # indistinguishable from a run that never happened. Every cell is
                        # either a result or a recorded skip.
                        produced = {portfolio.allocator for portfolio in portfolios}
                        for name in ("ga_min_risk", "ga_knee", "ga_max_return"):
                            if name not in produced:
                                rows.extend(_skip_row(
                                    config, month, labeling, screener, name, cap, seed,
                                    f"front of {front.size} point(s) supports no {name}",
                                ))

                    # Deterministic allocators: no seed, evaluated once per cap.
                    for name, allocator_fn in (
                        ("equal_weight", equal_weight),
                        ("min_volatility", minimum_volatility),
                        ("max_sharpe", maximum_sharpe),
                    ):
                        result = allocator_fn(moments, cap=cap)
                        if result.failed:
                            rows.extend(_skip_row(config, month, labeling, screener, name,
                                                  cap, None, result.message[:100]))
                            continue
                        key = (labeling, screener, name, cap, None)
                        sigma_ex = float(
                            np.sqrt(result.weights @ moments.covariance @ result.weights)
                        )
                        emitted, holdings = _evaluate_portfolio(
                            config, inputs, month, labeling, screener, name, cap, None,
                            usable, result.weights, sigma_ex_ante=sigma_ex,
                            selection_size=len(usable), previous=previous.get(key),
                        )
                        rows.extend(emitted)
                        if holdings is not None:
                            previous[key] = holdings
    return rows, failures


def load_grid_inputs(config: RunConfig) -> GridInputs:
    """Load every stored artefact the grid needs, from the cache and stage outputs."""
    from .marketdata.prices import cache_path, load_prices
    from .marketdata.rates import load_series, monthly_risk_free
    from .marketdata.universe import read_candidates

    features = pd.read_parquet(config.paths.features / "monthly_features.parquet")
    membership_frame = pd.read_csv(config.paths.universe / "screen_membership.csv")
    membership = {
        month: group["provider_symbol"].tolist()
        for month, group in membership_frame.groupby("month")
    }

    candidates = read_candidates(config.paths.universe)
    sic = frozenset(c.provider_symbol for c in candidates if c.listing != "national")

    wanted = sorted({s for members in membership.values() for s in members})
    prices = {}
    for symbol in wanted:
        if cache_path(config.paths.cache, symbol).exists():
            series = load_prices(config.paths.cache, symbol)["close"].dropna()
            if not series.empty:
                prices[symbol] = series

    panel = pd.DataFrame(prices).sort_index()
    daily_returns = panel.pct_change()
    monthly = panel.resample("MS").last().pct_change()
    monthly.index = [str(p) for p in monthly.index.to_period("M")]

    months = list(config.windows.eval_months())
    risk_free = monthly_risk_free(months, config.paths.cache).to_dict()
    fx = load_series("fx_fix", config.paths.cache)["value"]
    fx_monthly = fx.resample("MS").last().pct_change()
    fx_returns = {
        str(p): float(v)
        for p, v in zip(fx_monthly.index.to_period("M"), fx_monthly.to_numpy())
        if np.isfinite(v)
    }

    return GridInputs(
        features=features, prices=prices, daily_returns=daily_returns,
        calendar=panel.index, membership=membership, monthly_returns=monthly,
        risk_free=risk_free, fx_returns=fx_returns, sic_symbols=sic,
    )
