"""The comparator suite: Tier 0 references, Tier 1 baselines, and the literature mapping."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bmvport.benchmarks.tiers import (
    LITERATURE_MAPPING,
    TIER1_METHODS,
    TIER2_METHODS,
    build_tier1_arm,
    equal_weight_universe,
    inverse_volatility,
    momentum_selection,
    reference_series_returns,
)
from bmvport.optimization.objectives import estimate_moments


def _moments(symbols, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2024-01-01", periods=300, name="date")
    data = rng.normal(0.0005, 0.012, (300, len(symbols)))
    frame = pd.DataFrame(data, index=index, columns=list(symbols))
    return estimate_moments(frame, list(symbols), trailing_days=252)


def _monthly(symbols, months=20, seed=1):
    rng = np.random.default_rng(seed)
    idx = [str(p) for p in pd.period_range("2024-01", periods=months, freq="M")]
    return pd.DataFrame(rng.normal(0.01, 0.05, (months, len(symbols))), index=idx,
                        columns=list(symbols))


# --- Tier 1 ---------------------------------------------------------------------------


def test_equal_weight_is_uniform_and_sums_to_one():
    weights = equal_weight_universe(["A", "B", "C", "D"])
    assert np.allclose(weights, 0.25)
    assert weights.sum() == pytest.approx(1.0)


def test_inverse_volatility_favours_the_calmer_asset():
    symbols = ["A", "B"]
    moments = _moments(symbols)
    # Force B to be twice as volatile.
    cov = np.array([[0.0001, 0.0], [0.0, 0.0004]])
    from bmvport.optimization.objectives import Moments

    forced = Moments(tuple(symbols), np.array([0.001, 0.001]), cov, 252, "sample", "a", "b")
    weights = inverse_volatility(forced)
    assert weights[0] > weights[1]
    assert weights.sum() == pytest.approx(1.0)


def test_momentum_ranks_only_inside_the_admitted_universe():
    """The bug the gate caught: ranking across all history inflated the baseline by 18pp."""
    symbols = [f"S{i}" for i in range(10)]
    monthly = _monthly(symbols)
    # Make an excluded name the strongest performer by far.
    monthly["S9"] = 0.40
    eligible = symbols[:5]  # S9 is not admitted this month

    picked = momentum_selection(
        monthly, "2025-06", eligible,
        formation_months=12, skip_months=1, selection_fraction=0.4,
    )
    assert "S9" not in picked
    assert set(picked) <= set(eligible)


def test_momentum_returns_nothing_when_the_formation_window_is_short():
    symbols = ["A", "B", "C"]
    monthly = _monthly(symbols, months=5)
    picked = momentum_selection(
        monthly, "2024-06", symbols,
        formation_months=12, skip_months=1, selection_fraction=0.5,
    )
    assert picked == []


def test_momentum_selects_a_quintile_not_a_decile():
    symbols = [f"S{i}" for i in range(20)]
    monthly = _monthly(symbols)
    picked = momentum_selection(
        monthly, "2025-06", symbols,
        formation_months=12, skip_months=1, selection_fraction=0.20,
    )
    assert len(picked) == 4  # 20% of 20


def test_every_tier1_method_is_buildable():
    symbols = [f"S{i}" for i in range(8)]
    moments = _moments(symbols)
    monthly = _monthly(symbols)
    for name in TIER1_METHODS:
        arm = build_tier1_arm(name, "2025-06", symbols, moments=moments, monthly_returns=monthly)
        assert arm.tier == "tier1"
        if not arm.skipped:
            assert arm.weights.sum() == pytest.approx(1.0)
            assert len(arm.symbols) == len(arm.weights)


def test_empty_universe_is_skipped_with_a_reason():
    arm = build_tier1_arm("equal_weight_universe", "2025-06", [])
    assert arm.skipped and arm.skip_reason


def test_infeasible_cap_is_skipped_with_a_reason():
    symbols = ["A", "B", "C"]
    arm = build_tier1_arm("equal_weight_universe", "2025-06", symbols, cap=0.2)
    assert arm.skipped and "infeasible" in arm.skip_reason


# --- Tier 0 ---------------------------------------------------------------------------


def test_foreign_index_is_converted_to_pesos_by_compounding():
    calendar = pd.bdate_range("2025-01-01", "2025-02-28", name="date")
    flat = pd.Series(100.0, index=calendar)
    rising = pd.Series(np.linspace(100, 110, len(calendar)), index=calendar)
    result = reference_series_returns(
        "2025-02", risk_free=0.006, ipc_prices=flat,
        sp500_prices=rising, fx_prices=rising, calendar=calendar,
    )
    # Both legs rise by the same fraction, so the peso return compounds them.
    leg = result["sp500_mxn"]
    assert leg > 0
    assert set(result) == {"cetes_28d", "bmv_ipc", "sp500_mxn"}


def test_risk_free_passes_through_unchanged():
    calendar = pd.bdate_range("2025-01-01", "2025-02-28", name="date")
    flat = pd.Series(100.0, index=calendar)
    result = reference_series_returns(
        "2025-02", risk_free=0.0061, ipc_prices=flat,
        sp500_prices=flat, fx_prices=flat, calendar=calendar,
    )
    assert result["cetes_28d"] == 0.0061


# --- literature mapping ---------------------------------------------------------------


def test_mapped_cells_are_not_reimplemented_as_comparators():
    """A mapped factorial cell and a Tier 2 arm would be the same pipeline run twice."""
    for (screener, allocator) in LITERATURE_MAPPING:
        assert f"{screener}_{allocator}" not in TIER2_METHODS


def test_every_mapping_records_its_citation_and_its_differences():
    for key, entry in LITERATURE_MAPPING.items():
        assert entry["citation"]
        assert entry["equivalence"]
        # An equivalence claim without its caveats is the kind a referee checks first.
        assert entry["difference"]


def test_the_scalarisation_contrast_is_present():
    """The thesis claims to beat a single-objective formulation and never runs one."""
    assert "l4_single_objective_ga_sharpe" in TIER2_METHODS
