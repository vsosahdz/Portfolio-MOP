"""Data-quality gates.

The gates exist because the price source is fixed -- no institutional feed is available, so
a defect has to be handled rather than escaped. These tests pin the rule that matters: an
instrument can be quoted every session and still not be traded, and only a rule looking at
return variation and volume can tell the difference.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bmvport.config import UniverseConfig, load_config
from bmvport.marketdata.quality import (
    assess_ticker_months,
    find_stale_runs,
    quality_report,
)

ROOT = Path(__file__).resolve().parents[1]


def _universe(**overrides) -> UniverseConfig:
    defaults = dict(
        trailing_months=3,
        min_traded_day_fraction=0.90,
        min_median_traded_value_mxn=500_000.0,
        max_stale_run_days=5,
        min_universe_size=2,
    )
    defaults.update(overrides)
    return UniverseConfig(**defaults)


def _frame(prices: list[float], volumes: list[float] | None = None) -> pd.DataFrame:
    index = pd.bdate_range("2024-01-01", periods=len(prices), name="date")
    volumes = [1000.0] * len(prices) if volumes is None else volumes
    return pd.DataFrame({"close": prices, "volume": volumes}, index=index)


def test_stale_run_is_located_with_its_extent() -> None:
    prices = [10.0] + [12.0] * 9 + [13.0, 14.0]
    runs = find_stale_runs(_frame(prices)["close"], "TEST.MX", min_sessions=5)
    assert len(runs) == 1
    assert runs[0].sessions == 8  # eight sessions repeated the price that was set once
    assert runs[0].price == 12.0


def test_short_runs_are_not_reported() -> None:
    """Two-session repeats happen in any market; reporting them would bury real defects."""
    prices = [10.0, 10.0, 11.0, 11.0, 12.0, 13.0, 14.0, 15.0]
    assert find_stale_runs(_frame(prices)["close"], "TEST.MX", min_sessions=5) == []


def test_dormant_month_is_rejected() -> None:
    """Quoted every session, traded on none: the defect a session count cannot see."""
    prices = [19.5] * 20
    frame = _frame(prices, volumes=[0.0] * 20)
    result = assess_ticker_months(frame, "AMXB.MX", _universe(), min_trading_days=15)[0]

    assert result.sessions == 20  # a session-count rule would have admitted this month
    assert result.usable is False
    assert result.zero_return_fraction == pytest.approx(1.0)
    assert any("zero_return_fraction" in r for r in result.reasons)
    assert any("median_volume" in r for r in result.reasons)


def test_normal_month_is_accepted() -> None:
    rng = np.random.default_rng(0)
    prices = list(100 * np.cumprod(1 + rng.normal(0, 0.01, 20)))
    result = assess_ticker_months(_frame(prices), "OK.MX", _universe(), min_trading_days=15)[0]
    assert result.usable is True
    assert result.reasons == ()


def test_too_few_sessions_is_rejected() -> None:
    result = assess_ticker_months(
        _frame([10.0, 11.0, 12.0]), "THIN.MX", _universe(), min_trading_days=15
    )[0]
    assert result.usable is False
    assert any("sessions=" in r for r in result.reasons)


def test_non_positive_price_is_rejected() -> None:
    prices = [10.0, 11.0, 0.0] + [12.0 + i for i in range(17)]
    result = assess_ticker_months(_frame(prices), "BAD.MX", _universe(), min_trading_days=15)[0]
    assert result.usable is False
    assert any("non_positive_prices" in r for r in result.reasons)


def test_implausible_daily_return_is_rejected() -> None:
    prices = [10.0, 11.0, 100.0] + [101.0 + i for i in range(17)]
    result = assess_ticker_months(_frame(prices), "JUMP.MX", _universe(), min_trading_days=15)[0]
    assert result.usable is False
    assert any("max_abs_daily_return" in r for r in result.reasons)


def test_unusable_always_carries_a_reason() -> None:
    frame = _frame([5.0] * 20, volumes=[0.0] * 20)
    for assessment in assess_ticker_months(frame, "X.MX", _universe(), min_trading_days=15):
        assert assessment.usable is (assessment.reasons == ())


def test_thresholds_are_configurable_not_hard_coded() -> None:
    frame = _frame([19.5] * 20, volumes=[0.0] * 20)
    permissive = _universe(max_zero_return_fraction=1.01, min_month_median_volume=0.0)
    result = assess_ticker_months(frame, "X.MX", permissive, min_trading_days=15)[0]
    assert result.usable is True


def test_report_is_written_even_when_nothing_is_rejected(tmp_path: Path) -> None:
    """'We looked and found nothing' must be distinguishable from 'we did not look'."""
    rng = np.random.default_rng(1)
    prices = list(100 * np.cumprod(1 + rng.normal(0, 0.01, 20)))
    assessments = assess_ticker_months(_frame(prices), "OK.MX", _universe(), min_trading_days=15)
    months_path, runs_path = quality_report(assessments, [], tmp_path)
    assert months_path.exists() and runs_path.exists()
    assert len(pd.read_csv(months_path)) == len(assessments)


def test_empty_frame_yields_no_assessments() -> None:
    assert assess_ticker_months(pd.DataFrame(), "X.MX", _universe(), min_trading_days=15) == []


@pytest.mark.skipif(
    not (ROOT / "data" / "cache" / "AMXB.MX.parquet").exists(),
    reason="price cache not populated in this environment",
)
def test_amxb_dormant_months_are_caught_in_the_real_cache() -> None:
    """The concrete defect this gate was written for.

    América Móvil's B series spent 2023 quoted but dormant while it was being created. Those
    months must not reach the training set, and the months after must.
    """
    from bmvport.marketdata.prices import load_prices

    cfg = load_config(ROOT / "configs" / "default.yaml")
    frame = load_prices(cfg.paths.cache, "AMXB.MX")
    assessments = assess_ticker_months(
        frame, "AMXB.MX", cfg.universe, min_trading_days=cfg.features.min_trading_days_per_month
    )
    rejected = {a.month for a in assessments if not a.usable}
    assert {"2023-03", "2023-04", "2023-05", "2023-06", "2023-07", "2023-08"} <= rejected
    # The evaluation window is unaffected: the instrument traded normally from 2025.
    assert not any(a.month.startswith(("2025", "2026")) for a in assessments if not a.usable)
