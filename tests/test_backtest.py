"""Executable pricing, costs, metrics and return attribution."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bmvport.backtest.attribution import attribute_return
from bmvport.backtest.metrics import (
    apply_costs,
    evaluate_month,
    realised_risk,
    sharpe_ratio,
    turnover,
)
from bmvport.backtest.pricing import month_bounds, price_portfolio


def _calendar() -> pd.DatetimeIndex:
    return pd.bdate_range("2025-01-01", "2025-03-31", name="date")


def _prices(calendar, start: float, drift: float = 0.0) -> pd.Series:
    return pd.Series(start * (1 + drift) ** np.arange(len(calendar)), index=calendar)


# --- pricing --------------------------------------------------------------------------


def test_entry_and_exit_use_sessions_inside_the_month() -> None:
    calendar = _calendar()
    first, last = month_bounds(calendar, "2025-02")
    assert first.month == 2 and last.month == 2
    assert first == calendar[calendar.to_period("M").astype(str) == "2025-02"].min()


def test_entry_price_is_the_first_session_not_the_prior_close() -> None:
    """The original enters at the previous month's close, which is not transactable."""
    calendar = _calendar()
    prices = {"A.MX": _prices(calendar, 100.0, 0.01)}
    first, _ = month_bounds(calendar, "2025-02")
    priced = price_portfolio("2025-02", ["A.MX"], np.array([1.0]), prices, calendar)
    assert priced.entry_prices[0] == pytest.approx(float(prices["A.MX"].loc[first]))


def test_gross_return_matches_the_weighted_price_change() -> None:
    calendar = _calendar()
    prices = {"A.MX": _prices(calendar, 100.0, 0.01), "B.MX": _prices(calendar, 50.0, -0.005)}
    priced = price_portfolio("2025-02", ["A.MX", "B.MX"], np.array([0.6, 0.4]), prices, calendar)
    expected = float(
        np.sum(priced.weights * (priced.exit_prices / priced.entry_prices - 1.0))
    )
    assert priced.gross_return == pytest.approx(expected)


def test_instrument_without_an_entry_price_is_dropped_and_weights_renormalised() -> None:
    calendar = _calendar()
    late = _prices(calendar, 20.0)
    late[late.index < pd.Timestamp("2025-02-15")] = np.nan
    prices = {"A.MX": _prices(calendar, 100.0, 0.01), "LATE.MX": late}
    priced = price_portfolio("2025-02", ["A.MX", "LATE.MX"], np.array([0.5, 0.5]), prices, calendar)
    assert priced.dropped_at_entry == ("LATE.MX",)
    assert priced.symbols == ("A.MX",)
    assert priced.weights.sum() == pytest.approx(1.0)


def test_instrument_that_stops_trading_uses_its_last_close_and_is_recorded() -> None:
    calendar = _calendar()
    stopping = _prices(calendar, 40.0)
    stopping[stopping.index > pd.Timestamp("2025-02-14")] = np.nan
    prices = {"A.MX": _prices(calendar, 100.0), "GONE.MX": stopping}
    priced = price_portfolio("2025-02", ["A.MX", "GONE.MX"], np.array([0.5, 0.5]), prices, calendar)
    assert priced.early_exits == ("GONE.MX",)
    assert priced.size == 2


def test_unpriceable_portfolio_raises_rather_than_returning_zero() -> None:
    calendar = _calendar()
    with pytest.raises(ValueError, match="could be priced at entry"):
        price_portfolio("2025-02", ["MISSING.MX"], np.array([1.0]), {}, calendar)


def test_daily_returns_are_produced_for_realised_risk() -> None:
    calendar = _calendar()
    prices = {"A.MX": _prices(calendar, 100.0, 0.004)}
    priced = price_portfolio("2025-02", ["A.MX"], np.array([1.0]), prices, calendar)
    assert len(priced.daily_returns) > 5
    assert np.isfinite(priced.daily_returns).all()


# --- turnover and costs ---------------------------------------------------------------


def test_first_month_is_full_turnover_from_cash() -> None:
    assert turnover({"A": 0.5, "B": 0.5}, None) == 1.0


def test_identical_holdings_have_no_turnover() -> None:
    holdings = {"A": 0.5, "B": 0.5}
    assert turnover(holdings, holdings) == pytest.approx(0.0)


def test_complete_rotation_is_full_turnover() -> None:
    assert turnover({"C": 0.5, "D": 0.5}, {"A": 0.5, "B": 0.5}) == pytest.approx(1.0)


def test_costs_scale_with_turnover_and_are_charged_round_trip() -> None:
    net = apply_costs(0.05, 1.0, [0.0, 50.0])
    assert net[0.0] == pytest.approx(0.05)
    # 50 bps round trip on a fully rotated portfolio.
    assert net[50.0] == pytest.approx(0.05 - 2 * 1.0 * 0.005)


def test_zero_cost_scenario_is_the_gross_upper_bound() -> None:
    net = apply_costs(0.03, 0.8, [0.0, 25.0, 100.0])
    assert net[0.0] > net[25.0] > net[100.0]


# --- metrics --------------------------------------------------------------------------


def test_realised_risk_is_the_sample_standard_deviation() -> None:
    series = pd.Series([0.01, -0.02, 0.005, 0.0])
    assert realised_risk(series) == pytest.approx(series.std(ddof=1))


def test_realised_risk_is_undefined_for_a_single_session() -> None:
    assert np.isnan(realised_risk(pd.Series([0.01])))


def test_sharpe_is_undefined_rather_than_infinite_for_a_motionless_portfolio() -> None:
    """An infinity would dominate every aggregate it entered."""
    assert np.isnan(sharpe_ratio(0.02, 0.0, 0.005))
    assert np.isnan(sharpe_ratio(0.02, float("nan"), 0.005))


def test_sharpe_uses_the_month_specific_risk_free_rate() -> None:
    assert sharpe_ratio(0.02, 0.01, 0.008) == pytest.approx((0.02 - 0.008) / 0.01)


def test_ex_ante_and_realised_risk_are_separate_fields() -> None:
    """The original uses one word for both, which makes its risk tables uninterpretable."""
    metrics = evaluate_month(
        "2025-02", 0.03, pd.Series([0.01, -0.01, 0.02]),
        {"A": 0.5, "B": 0.5}, None,
        sigma_ex_ante=0.0123, risk_free=0.006, cost_scenarios_bps=[0.0, 50.0],
    )
    assert metrics.sigma_ex_ante == 0.0123
    assert metrics.sigma_realized != metrics.sigma_ex_ante
    assert set(metrics.net_returns) == {0.0, 50.0}
    assert set(metrics.sharpe) == {0.0, 50.0}


# --- attribution ----------------------------------------------------------------------


def _attribution(**overrides):
    kwargs = dict(
        gross_return=0.04,
        portfolio_weights={"A.MX": 0.6, "AAPL.MX": 0.4},
        selected_returns={"A.MX": 0.03, "AAPL.MX": 0.05},
        universe_returns={"A.MX": 0.03, "AAPL.MX": 0.05, "C.MX": -0.01, "D.MX": 0.00},
        sic_symbols=["AAPL.MX"],
        fx_return=0.02,
    )
    kwargs.update(overrides)
    return attribute_return("2025-02", **kwargs)


def test_decomposition_reconciles_exactly() -> None:
    """An unexplained remainder invites the suspicion that it holds the result."""
    result = _attribution()
    assert result.reconciles()
    assert result.residual == pytest.approx(0.0, abs=1e-12)
    assert result.market + result.selection + result.allocation == pytest.approx(
        result.gross_return
    )


def test_market_is_the_passive_equal_weight_universe() -> None:
    result = _attribution()
    assert result.market == pytest.approx(np.mean([0.03, 0.05, -0.01, 0.00]))


def test_selection_measures_choosing_the_names() -> None:
    result = _attribution()
    assert result.selection == pytest.approx(np.mean([0.03, 0.05]) - result.market)


def test_allocation_measures_the_weighting_given_the_selection() -> None:
    result = _attribution()
    assert result.allocation == pytest.approx(0.04 - np.mean([0.03, 0.05]))


def test_active_return_is_selection_plus_allocation() -> None:
    result = _attribution()
    assert result.active_return == pytest.approx(result.gross_return - result.market)


def test_fx_tilt_is_exposure_beyond_the_passive_alternative() -> None:
    result = _attribution()
    # Portfolio holds 40% SIC; the universe is 25% SIC.
    assert result.portfolio_sic_weight == pytest.approx(0.4)
    assert result.universe_sic_share == pytest.approx(0.25)
    assert result.fx_tilt == pytest.approx((0.4 - 0.25) * 0.02)


def test_matching_the_universe_exposure_leaves_no_fx_tilt() -> None:
    result = _attribution(portfolio_weights={"A.MX": 0.75, "AAPL.MX": 0.25})
    assert result.fx_tilt == pytest.approx(0.0)


def test_fx_tilt_is_not_added_to_the_identity() -> None:
    """Adding it would double-count: the exposure is already inside the peso returns."""
    result = _attribution()
    assert result.fx_tilt != 0.0
    assert result.market + result.selection + result.allocation == pytest.approx(
        result.gross_return
    )


def test_empty_universe_is_rejected() -> None:
    with pytest.raises(ValueError, match="no baseline"):
        _attribution(universe_returns={})


# --- the tidy results table -----------------------------------------------------------


def _row(**overrides):
    from bmvport.backtest.results import ResultRow

    base = dict(
        arm_kind="proposed", month="2025-02", labeling="historical_and_t1",
        screener="svm", allocator="ga_knee", weight_cap=1.0,
        covariance_estimator="sample", seed=7, cost_bps=50.0,
    )
    base.update(overrides)
    return ResultRow(**base)


def test_key_uniquely_identifies_a_row() -> None:
    from bmvport.backtest.results import build_results_table, check_key_uniqueness

    frame = build_results_table([_row(), _row(cost_bps=0.0), _row(seed=8)])
    check_key_uniqueness(frame)  # must not raise
    assert len(frame) == 3


def test_duplicate_keys_are_rejected() -> None:
    """Two portfolios claiming one identity would double count in every aggregate."""
    from bmvport.backtest.results import build_results_table, check_key_uniqueness

    frame = build_results_table([_row(), _row()])
    with pytest.raises(ValueError, match="duplicated result keys"):
        check_key_uniqueness(frame)


def test_schema_permits_both_readings_of_the_result() -> None:
    """Gross and net, return and turnover, ex-ante and realised risk, attribution.

    A table carrying only the flattering columns permits only the flattering account.
    """
    from bmvport.backtest.results import build_results_table

    frame = build_results_table([_row()])
    for column in (
        "return_gross", "return_net", "turnover",
        "sigma_ex_ante", "sigma_realized",
        "attribution_market", "attribution_selection",
        "attribution_allocation", "attribution_fx_tilt",
        "cap_binding", "skip_reason",
    ):
        assert column in frame.columns


def test_skipped_cells_are_rows_not_absences() -> None:
    from bmvport.backtest.results import build_results_table

    frame = build_results_table(
        [_row(), _row(month="2025-03", skip_reason="empty selection")]
    )
    assert len(frame) == 2
    assert frame["skip_reason"].astype(bool).sum() == 1


def test_summary_records_completeness_and_the_cost_side(tmp_path) -> None:
    import json

    from bmvport.backtest.results import build_results_table, write_results_table

    frame = build_results_table(
        [
            _row(turnover=0.8, holdings=20, cap_binding=True),
            _row(month="2025-03", turnover=0.6, holdings=18, cap_binding=False),
            _row(month="2025-04", skip_reason="infeasible cap"),
        ]
    )
    paths = write_results_table(frame, tmp_path)
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))

    assert summary["rows"] == 3
    assert summary["skipped_rows"] == 1
    assert summary["skip_reasons"] == {"infeasible cap": 1}
    assert summary["median_turnover"] == pytest.approx(0.7)
    assert summary["cap_binding_share"] == pytest.approx(0.5)


def test_writing_refuses_a_table_with_duplicate_keys(tmp_path) -> None:
    from bmvport.backtest.results import build_results_table, write_results_table

    frame = build_results_table([_row(), _row()])
    with pytest.raises(ValueError, match="duplicated result keys"):
        write_results_table(frame, tmp_path)


def test_empty_table_is_handled() -> None:
    from bmvport.backtest.results import build_results_table, check_key_uniqueness

    frame = build_results_table([])
    check_key_uniqueness(frame)
    assert frame.empty
