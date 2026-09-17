"""Component ablation: the ladder, leave-one-out, and the resolution discipline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bmvport.ablation.components import (
    COMPONENTS,
    LADDER,
    ComponentEffect,
    compare_ablations,
    effect_between,
    ladder_effects,
    leave_one_out_effects,
)


def _results(months=18, effect_by_rung=None, seed=0):
    """Synthetic results table covering every ladder rung and leave-one-out configuration."""
    rng = np.random.default_rng(seed)
    effect_by_rung = effect_by_rung or {}
    month_labels = [str(p) for p in pd.period_range("2025-01", periods=months, freq="M")]
    rows = []
    for rung in LADDER:
        base = effect_by_rung.get(rung.name, 0.0)
        for m in month_labels:
            row = {
                "arm_kind": "proposed", "month": m, "labeling": "historical_and_t1",
                "screener": "xgboost", "allocator": "ga_knee", "weight_cap": 1.0,
                "covariance_estimator": "sample", "seed": 1, "cost_bps": 50.0,
                "return_net": base + rng.normal(0, 0.01), "skip_reason": "",
            }
            row.update(rung.selector)
            rows.append(row)
    # Leave-one-out configurations.
    full = dict(LADDER[-1].selector)
    for replacement in COMPONENTS.values():
        for m in month_labels:
            row = {
                "arm_kind": "proposed", "month": m, "labeling": "historical_and_t1",
                "screener": "xgboost", "allocator": "ga_knee", "weight_cap": 1.0,
                "covariance_estimator": "sample", "seed": 1, "cost_bps": 50.0,
                "return_net": rng.normal(0, 0.01), "skip_reason": "",
            }
            row.update({**full, **replacement})
            rows.append(row)
    return pd.DataFrame(rows)


def test_ladder_adds_one_component_per_rung():
    """Each rung must differ from its predecessor by exactly one named component."""
    assert LADDER[0].adds == "baseline"
    names = [r.name for r in LADDER]
    assert names == ["A0", "A1", "A2", "A3", "A4", "A5", "A6"]
    assert len(set(r.adds for r in LADDER)) == len(LADDER)


def test_every_factorial_rung_pins_the_same_factors():
    """An unpinned factor would be averaged over while another rung fixes it."""
    from bmvport.ablation.components import ladder_is_well_formed

    ladder_is_well_formed()  # must not raise


def test_an_unpinned_rung_is_rejected():
    from bmvport.ablation.components import Rung, ladder_is_well_formed

    malformed = (
        Rung("A2", "x", {"arm_kind": "proposed", "screener": "all_stocks",
                         "allocator": "ga_max_return", "labeling": "historical",
                         "weight_cap": 1.0}, ""),
        Rung("A3", "y", {"arm_kind": "proposed", "screener": "xgboost",
                         "allocator": "ga_max_return", "weight_cap": 1.0}, ""),
    )
    with pytest.raises(ValueError, match="pins a different factor set"):
        ladder_is_well_formed(malformed)


def test_a_degenerate_comparison_is_not_called_resolved():
    """Identical configurations give a zero bound; that is unresolvable, not a finding."""
    effect = ComponentEffect("ladder", "x", "A3", "A4",
                             effect=0.0, ci_low=0.0, ci_high=0.0,
                             minimum_detectable=0.0, n_blocks=18)
    assert effect.below_resolution is True


def test_ladder_produces_one_effect_per_step():
    effects = ladder_effects(_results(), metric="return_net", cost_bps=50.0)
    assert len(effects) == len(LADDER) - 1
    assert all(e.kind == "ladder" for e in effects)


def test_effect_carries_its_interval_and_its_bound():
    for effect in ladder_effects(_results(), metric="return_net", cost_bps=50.0):
        if effect.available:
            assert effect.ci_low <= effect.effect <= effect.ci_high
            assert effect.minimum_detectable > 0
            assert effect.n_blocks > 0


def test_small_effect_is_labelled_below_resolution():
    """A column of small positive numbers is a chain of noise, not of contributions."""
    effect = ComponentEffect("ladder", "screening", "A2", "A3",
                             effect=0.001, ci_low=-0.01, ci_high=0.012,
                             minimum_detectable=0.02, n_blocks=18)
    assert effect.below_resolution is True
    assert "below the resolution" in effect.verdict


def test_large_effect_is_resolved():
    effect = ComponentEffect("ladder", "screening", "A2", "A3",
                             effect=0.05, ci_low=0.03, ci_high=0.07,
                             minimum_detectable=0.02, n_blocks=18)
    assert effect.below_resolution is False
    assert effect.verdict == "resolved: increase"


def test_negative_resolved_effect_is_reported_as_a_decrease():
    effect = ComponentEffect("ladder", "cap", "A5", "A6",
                             effect=-0.05, ci_low=-0.07, ci_high=-0.03,
                             minimum_detectable=0.02, n_blocks=18)
    assert effect.verdict == "resolved: decrease"


def test_missing_rung_is_reported_not_closed_over():
    """Skipping a gap would attribute two components' effect to one of them."""
    results = _results()
    results = results[results["allocator"] != "mv_analytic_universe"]  # remove A1
    effects = ladder_effects(results, metric="return_net", cost_bps=50.0)
    unavailable = [e for e in effects if not e.available]
    assert unavailable
    assert all("absent from results" in e.unavailable_reason for e in unavailable)
    assert all(e.below_resolution for e in unavailable)


def test_leave_one_out_covers_every_component():
    effects = leave_one_out_effects(_results(), metric="return_net", cost_bps=50.0)
    assert {e.component for e in effects} == set(COMPONENTS)
    assert all(e.kind == "leave_one_out" for e in effects)


def test_leave_one_out_sign_matches_the_ladder_convention():
    """Positive means the component helped, in both ablations, or the table misleads."""
    results = _results(effect_by_rung={"A6": 0.05})
    effects = leave_one_out_effects(results, metric="return_net", cost_bps=50.0)
    resolved = [e for e in effects if e.available and not e.below_resolution]
    # A6 is inflated, so removing any component should show a positive loss.
    assert all(e.effect > 0 for e in resolved) or not resolved


def test_comparison_puts_both_ablations_side_by_side():
    results = _results()
    table = compare_ablations(
        ladder_effects(results, metric="return_net", cost_bps=50.0),
        leave_one_out_effects(results, metric="return_net", cost_bps=50.0),
    )
    for column in ("component", "ladder_effect", "loo_effect", "minimum_detectable", "disagrees"):
        assert column in table.columns


def test_disagreement_requires_both_effects_to_be_resolved():
    """Two noise values of opposite sign are not a disagreement worth reporting."""
    ladder = [ComponentEffect("ladder", "screening", "A2", "A3",
                              0.001, -0.01, 0.01, 0.02, 18)]
    loo = [ComponentEffect("leave_one_out", "screening", "A6 minus screening", "A6",
                           -0.001, -0.01, 0.01, 0.02, 18)]
    table = compare_ablations(ladder, loo)
    assert bool(table["disagrees"].iloc[0]) is False


def test_genuine_disagreement_is_flagged():
    ladder = [ComponentEffect("ladder", "screening", "A2", "A3",
                              0.06, 0.04, 0.08, 0.02, 18)]
    loo = [ComponentEffect("leave_one_out", "screening", "A6 minus screening", "A6",
                           -0.06, -0.08, -0.04, 0.02, 18)]
    table = compare_ablations(ladder, loo)
    assert bool(table["disagrees"].iloc[0]) is True


def test_effects_come_only_from_the_results_table():
    """No separate evaluation path, so ablation and headline figures cannot disagree."""
    empty = pd.DataFrame(columns=[
        "arm_kind", "month", "labeling", "screener", "allocator", "weight_cap",
        "covariance_estimator", "seed", "cost_bps", "return_net", "skip_reason",
    ])
    assert effect_between(empty, LADDER[0].selector, LADDER[1].selector,
                          metric="return_net", cost_bps=50.0) is None


def test_skipped_rows_are_excluded_from_effects():
    results = _results()
    results.loc[results["month"] == "2025-01", "skip_reason"] = "empty selection"
    effects = ladder_effects(results, metric="return_net", cost_bps=50.0)
    available = [e for e in effects if e.available]
    assert available and all(e.n_blocks <= 17 for e in available)
