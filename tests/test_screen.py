"""Monthly point-in-time universe screen.

The property these tests exist to protect is the one that cannot be checked after the fact:
a screen that consults the month it is selecting for produces a universe that looks
unusually tradeable in exactly the months that mattered, and nothing downstream can detect
it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bmvport.config import load_config
from bmvport.marketdata.screen import (
    admit_over_evaluation_window,
    run_screen,
    screen_month,
    write_screen_outputs,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


@pytest.fixture(scope="module")
def config():
    return load_config(CONFIGS / "default.yaml")


def _series(start: str, end: str, price: float = 100.0, volume: float = 50_000.0,
            drift: float = 0.0005) -> pd.DataFrame:
    index = pd.bdate_range(start, end, name="date")
    rng = np.random.default_rng(0)
    closes = price * np.cumprod(1 + rng.normal(drift, 0.012, len(index)))
    return pd.DataFrame({"close": closes, "volume": [volume] * len(index)}, index=index)


def test_screen_uses_only_data_before_the_month(config) -> None:
    """The decisive property: a spike inside the month must not change membership."""
    clean = _series("2023-01-02", "2026-06-30")
    frames = {f"OK{i}.MX": clean.copy() for i in range(70)}

    baseline = screen_month("2025-06", frames, config)

    # Make one instrument spectacularly liquid, but only from the month itself onward.
    poisoned = clean.copy()
    inside = poisoned.index >= pd.Timestamp("2025-06-01")
    poisoned.loc[inside, "volume"] = 10_000_000.0
    frames["OK0.MX"] = poisoned

    after = screen_month("2025-06", frames, config)
    assert after.members == baseline.members
    assert after.median_traded_value["OK0.MX"] == pytest.approx(
        baseline.median_traded_value["OK0.MX"]
    )


def test_illiquid_instrument_is_excluded_with_a_reason(config) -> None:
    frames = {f"OK{i}.MX": _series("2023-01-02", "2026-06-30") for i in range(70)}
    frames["THIN.MX"] = _series("2023-01-02", "2026-06-30", price=1.0, volume=10.0)

    result = screen_month("2025-06", frames, config)
    assert "THIN.MX" not in result.members
    assert result.exclusions.get("median_traded_value", 0) >= 1


def test_untraded_days_exclude_an_instrument(config) -> None:
    frames = {f"OK{i}.MX": _series("2023-01-02", "2026-06-30") for i in range(70)}
    dormant = _series("2023-01-02", "2026-06-30")
    dormant["volume"] = 0.0
    frames["DORMANT.MX"] = dormant

    result = screen_month("2025-06", frames, config)
    assert "DORMANT.MX" not in result.members
    assert result.exclusions.get("traded_day_fraction", 0) >= 1


def test_insufficient_warmup_is_excluded(config) -> None:
    frames = {f"OK{i}.MX": _series("2023-01-02", "2026-06-30") for i in range(70)}
    # Twenty sessions of history is far short of the 60-day indicator warm-up.
    frames["NEW.MX"] = _series("2025-05-01", "2025-05-30")

    result = screen_month("2025-06", frames, config)
    assert "NEW.MX" not in result.members
    assert result.exclusions.get("insufficient_warmup", 0) >= 1


def test_membership_can_change_between_months(config) -> None:
    """Liquidity is not a fixed property, and the screen must be able to say so."""
    frames = {f"OK{i}.MX": _series("2023-01-02", "2026-06-30") for i in range(70)}
    fading = _series("2023-01-02", "2026-06-30")
    fading.loc[fading.index >= pd.Timestamp("2025-03-01"), "volume"] = 5.0
    frames["FADING.MX"] = fading

    early = screen_month("2025-02", frames, config)
    late = screen_month("2025-09", frames, config)
    assert "FADING.MX" in early.members
    assert "FADING.MX" not in late.members


def test_below_minimum_universe_is_flagged(config) -> None:
    frames = {f"OK{i}.MX": _series("2023-01-02", "2026-06-30") for i in range(5)}
    result = screen_month("2025-06", frames, config)
    assert result.size == 5
    assert result.below_minimum is True


def test_late_listing_is_not_admitted_when_coverage_is_required(config) -> None:
    frames = {
        "FULL.MX": _series("2023-01-02", "2026-06-30"),
        "LATE.MX": _series("2025-09-01", "2026-06-30"),
    }
    listings = {"FULL.MX": "national", "LATE.MX": "national"}
    verdicts = {a.provider_symbol: a for a in admit_over_evaluation_window(frames, listings, config)}
    assert verdicts["FULL.MX"].admitted is True
    assert verdicts["LATE.MX"].admitted is False
    assert any("listed_late_by" in r for r in verdicts["LATE.MX"].reasons)


def test_instrument_that_stops_trading_is_not_admitted(config) -> None:
    frames = {
        "FULL.MX": _series("2023-01-02", "2026-06-30"),
        "GONE.MX": _series("2023-01-02", "2025-08-31"),
    }
    listings = {k: "national" for k in frames}
    verdicts = {a.provider_symbol: a for a in admit_over_evaluation_window(frames, listings, config)}
    assert verdicts["GONE.MX"].admitted is False
    assert any("stopped_trading_early_by" in r for r in verdicts["GONE.MX"].reasons)


def test_missing_2023_history_does_not_block_admission(config) -> None:
    """Coverage is judged over the evaluation window, not the download window.

    An instrument that listed in 2024 is holdable throughout 2025-2026; it merely has fewer
    early training observations.
    """
    frames = {
        "FULL.MX": _series("2023-01-02", "2026-06-30"),
        "FROM2024.MX": _series("2024-06-03", "2026-06-30"),
    }
    listings = {k: "national" for k in frames}
    verdicts = {a.provider_symbol: a for a in admit_over_evaluation_window(frames, listings, config)}
    assert verdicts["FROM2024.MX"].admitted is True


def test_rejection_always_carries_a_reason(config) -> None:
    frames = {
        "FULL.MX": _series("2023-01-02", "2026-06-30"),
        "LATE.MX": _series("2025-09-01", "2026-06-30"),
    }
    listings = {k: "national" for k in frames}
    for verdict in admit_over_evaluation_window(frames, listings, config):
        assert verdict.admitted is (verdict.reasons == ())


def test_outputs_are_written(config, tmp_path: Path) -> None:
    frames = {f"OK{i}.MX": _series("2023-01-02", "2026-06-30") for i in range(70)}
    listings = {k: "national" for k in frames}
    admissions, memberships = run_screen(frames, listings, config)
    paths = write_screen_outputs(admissions, memberships, tmp_path)

    assert set(paths) == {"admission", "membership", "diagnostics"}
    assert all(p.exists() for p in paths.values())
    assert len(memberships) == len(config.windows.eval_months())

    import json

    diagnostics = json.loads(paths["diagnostics"].read_text(encoding="utf-8"))
    assert len(diagnostics) == len(memberships)
    # Threshold distributions must be recorded so the cut-offs can be justified from data.
    assert "traded_value_percentiles" in diagnostics[0]
