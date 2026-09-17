"""The 43 daily indicators and their aggregation into 129 monthly features.

These tests carry the contract with the manuscript's feature description. A referee will
check the formulas against Table A.1, so the two places the table is ambiguous are pinned
explicitly rather than left to whatever a library happens to do.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bmvport.config import load_config
from bmvport.features.indicators import daily_indicators, indicator_specs
from bmvport.features.monthly import (
    KEY_COLUMNS,
    build_feature_matrix,
    monthly_features,
    monthly_returns,
    write_feature_dictionary,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


@pytest.fixture(scope="module")
def features():
    return load_config(CONFIGS / "default.yaml").features


def _prices(periods: int = 400, seed: int = 0) -> pd.DataFrame:
    index = pd.bdate_range("2024-01-01", periods=periods, name="date")
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0.0004, 0.012, periods))
    spread = np.abs(rng.normal(0.006, 0.002, periods)) * close
    return pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 0.003, periods)),
            "high": close + spread,
            "low": close - spread,
            "close": close,
            "adj_close": close,
            "volume": rng.integers(20_000, 200_000, periods).astype(float),
        },
        index=index,
    )


def test_exactly_43_daily_indicators(features) -> None:
    result = daily_indicators(_prices(), features)
    assert result.shape[1] == 43
    assert len(indicator_specs(features)) == 43
    assert list(result.columns) == [s.name for s in indicator_specs(features)]


def test_indicator_names_are_unique(features) -> None:
    names = [s.name for s in indicator_specs(features)]
    assert len(set(names)) == len(names)


def test_declared_count_mismatch_is_an_error(features) -> None:
    from dataclasses import replace

    with pytest.raises(Exception):
        # Declaring 40 while the windows imply 43 must fail rather than silently drift.
        replace(features, expected_daily_indicators=40)


def test_missing_price_column_is_rejected(features) -> None:
    frame = _prices().drop(columns=["volume"])
    with pytest.raises(ValueError, match="missing required columns"):
        daily_indicators(frame, features)


def test_warmup_leaves_long_windows_undefined(features) -> None:
    """No indicator may be computed from a partial trailing window."""
    result = daily_indicators(_prices(), features)
    # The 60-session families need 60 prior sessions.
    assert result["ma_60"].iloc[:59].isna().all()
    assert result["ma_60"].iloc[59:].notna().all()
    assert result["ccr_60"].iloc[:59].isna().all()
    # The first row has no predecessor at all.
    assert np.isnan(result["return"].iloc[0])


def test_rsi_uses_a_plain_mean_not_wilder_smoothing(features) -> None:
    """Table A.1 specifies an arithmetic mean; most libraries would smooth instead."""
    frame = _prices()
    result = daily_indicators(frame, features)
    change = frame["close"].diff()
    gain = change.where(change >= 0, 0.0)
    expected = gain.rolling(features.rsi_window, min_periods=features.rsi_window).mean()
    pd.testing.assert_series_equal(
        result["avg_gain"], expected, check_names=False, check_freq=False
    )


def test_loss_is_a_magnitude_so_rsi_stays_bounded(features) -> None:
    """The table's signed definition would make RS negative and RSI divergent."""
    result = daily_indicators(_prices(), features)
    losses = result["loss"].dropna()
    assert (losses >= 0).all()
    rsi = result["rsi"].dropna()
    assert not rsi.empty
    assert rsi.between(0, 100).all()


def test_rsi_is_100_when_the_window_has_no_down_days(features) -> None:
    index = pd.bdate_range("2024-01-01", periods=60, name="date")
    close = pd.Series(np.linspace(100, 160, 60), index=index)  # strictly rising
    frame = pd.DataFrame(
        {
            "open": close, "high": close * 1.01, "low": close * 0.99,
            "close": close, "adj_close": close, "volume": 100_000.0,
        },
        index=index,
    )
    result = daily_indicators(frame, features)
    assert result["rsi"].dropna().eq(100.0).all()


def test_stochastic_uses_the_window_maximum(features) -> None:
    """The conventional form, not the table's ``high_t`` rendering."""
    frame = _prices()
    result = daily_indicators(frame, features)
    n = features.stochastic_window
    lowest = frame["low"].rolling(n, min_periods=n).min()
    highest = frame["high"].rolling(n, min_periods=n).max()
    expected = (frame["close"] - lowest) / (highest - lowest) * 100.0
    pd.testing.assert_series_equal(
        result["so"], expected.where(highest > lowest), check_names=False, check_freq=False
    )
    assert result["so"].dropna().between(0, 100).all()


def test_flat_window_leaves_the_oscillator_undefined_not_infinite(features) -> None:
    index = pd.bdate_range("2024-01-01", periods=60, name="date")
    flat = pd.Series(50.0, index=index)
    frame = pd.DataFrame(
        {"open": flat, "high": flat, "low": flat, "close": flat,
         "adj_close": flat, "volume": 1000.0},
        index=index,
    )
    result = daily_indicators(frame, features)
    assert result["so"].isna().all()
    assert np.isfinite(result["so"].dropna()).all()


def test_no_look_ahead(features) -> None:
    """Perturbing any price after date d must not change any feature dated at or before d.

    This is the guarantee that cannot be checked after the fact. A feature that peeks
    forward produces results that look prescient in exactly the months that mattered, and
    nothing downstream can detect it.
    """
    frame = _prices(periods=300, seed=3)
    cutoff = frame.index[200]

    baseline = daily_indicators(frame, features)

    perturbed = frame.copy()
    after = perturbed.index > cutoff
    perturbed.loc[after, ["open", "high", "low", "close", "adj_close"]] *= 1.35
    perturbed.loc[after, "volume"] *= 7.0
    recomputed = daily_indicators(perturbed, features)

    pd.testing.assert_frame_equal(
        baseline.loc[:cutoff], recomputed.loc[:cutoff], check_freq=False
    )
    # And the perturbation really did change the future, so the test is not vacuous.
    assert not baseline.loc[cutoff:].equals(recomputed.loc[cutoff:])


def test_exactly_129_monthly_features(features) -> None:
    matrix, _ = monthly_features(_prices(), "TEST.MX", features)
    feature_columns = [c for c in matrix.columns if c not in KEY_COLUMNS]
    assert len(feature_columns) == 129
    assert matrix["provider_symbol"].eq("TEST.MX").all()


def test_monthly_aggregations_are_mean_std_last(features) -> None:
    frame = _prices()
    matrix, _ = monthly_features(frame, "TEST.MX", features)
    daily = daily_indicators(frame, features).dropna(how="any")
    month = matrix["month"].iloc[0]
    group = daily[daily.index.to_period("M").astype(str) == month]

    assert matrix["ma_5_mean"].iloc[0] == pytest.approx(group["ma_5"].mean())
    assert matrix["ma_5_std"].iloc[0] == pytest.approx(group["ma_5"].std(ddof=1))
    assert matrix["ma_5_last"].iloc[0] == pytest.approx(group["ma_5"].iloc[-1])


def test_short_month_is_excluded_with_its_session_count(features) -> None:
    frame = _prices(periods=200)
    # Remove most sessions of one month so it falls under the minimum.
    target = frame.index.to_period("M").astype(str) == "2024-09"
    keep = ~target | (np.cumsum(target) <= 4)
    matrix, exclusions = monthly_features(frame[keep], "TEST.MX", features)

    assert "2024-09" not in set(matrix["month"])
    dropped = [e for e in exclusions if e.month == "2024-09"]
    assert dropped and dropped[0].sessions < features.min_trading_days_per_month


def test_monthly_returns_use_adjusted_close() -> None:
    frame = _prices(periods=200)
    frame["adj_close"] = frame["close"] * 0.5  # a split-like adjustment
    result = monthly_returns(frame)
    expected = frame["adj_close"].resample("MS").last().pct_change().dropna()
    assert result["monthly_return"].to_numpy() == pytest.approx(expected.to_numpy())


def test_trailing_mean_requires_three_prior_months() -> None:
    result = monthly_returns(_prices(periods=200))
    assert pd.isna(result["trailing_3m_mean_return"].iloc[0])
    assert result["trailing_3m_mean_return"].notna().any()


def test_trailing_mean_excludes_the_current_month() -> None:
    """MA3 at month t uses t-1, t-2, t-3; including t would leak the label's own input."""
    result = monthly_returns(_prices(periods=300)).dropna()
    row = result.iloc[3]
    prior = result["monthly_return"].iloc[0:3]
    assert row["trailing_3m_mean_return"] == pytest.approx(prior.mean())


def test_quality_gates_drop_ticker_months(features) -> None:
    frames = {"A.MX": _prices(periods=300), "B.MX": _prices(periods=300, seed=1)}
    full, _ = build_feature_matrix(frames, features)
    months = sorted(set(full["month"]))
    blocked = {("A.MX", months[1]): False}
    gated, exclusions = build_feature_matrix(frames, features, usable=blocked)

    assert len(gated) == len(full) - 1
    assert any(e.reason == "rejected by data-quality gates" for e in exclusions)


def test_feature_dictionary_matches_the_matrix_in_both_directions(
    features, tmp_path: Path
) -> None:
    matrix, _ = monthly_features(_prices(), "TEST.MX", features)
    path = write_feature_dictionary(features, tmp_path / "dictionary.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    documented = {entry["column"] for entry in payload["columns"]}
    produced = {c for c in matrix.columns if c not in KEY_COLUMNS}
    produced -= set(payload["label_input_columns"])

    assert documented == produced
    assert payload["version"] == features.dictionary_version
    assert len(payload["columns"]) == 129


def test_dictionary_records_the_resolved_ambiguities(features, tmp_path: Path) -> None:
    """The two departures from Table A.1 must be discoverable, not buried in code."""
    path = write_feature_dictionary(features, tmp_path / "dictionary.json")
    notes = " ".join(e["note"] for e in json.loads(path.read_text(encoding="utf-8"))["columns"])
    assert "magnitude" in notes  # signed-loss resolution
    assert "max(high" in notes  # stochastic denominator resolution
    assert "Wilder" in notes  # plain-mean RSI
