"""Selection-size ablation: cost, tractability, and the comparison that is valid across sizes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bmvport.ablation.search_space import (
    AblationInstance,
    attainment_ratio,
    run_ablation_instance,
    search_space_size,
    truncate_selection,
)
from bmvport.optimization.objectives import estimate_moments


def _moments(n, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=300, name="date")
    syms = [f"A{i}" for i in range(n)]
    frame = pd.DataFrame(rng.normal(0.0005, 0.012, (300, n)), index=idx, columns=syms)
    return estimate_moments(frame, syms, trailing_days=252)


def test_truncation_keeps_the_highest_scoring():
    symbols = ["A", "B", "C", "D"]
    scores = {"A": 0.1, "B": 0.9, "C": 0.5, "D": 0.7}
    assert truncate_selection(symbols, scores, 2) == ["B", "D"]


def test_no_preselection_keeps_everything():
    symbols = ["A", "B", "C"]
    scores = {s: 0.5 for s in symbols}
    assert truncate_selection(symbols, scores, None) == symbols


def test_ties_break_deterministically():
    """An arbitrary tie-break would make the ablation irreproducible where scores are flat."""
    symbols = ["D", "B", "A", "C"]
    scores = {s: 0.5 for s in symbols}
    first = truncate_selection(symbols, scores, 2)
    second = truncate_selection(list(reversed(symbols)), scores, 2)
    assert first == second == ["A", "B"]


def test_missing_score_is_rejected():
    with pytest.raises(ValueError, match="no decision score"):
        truncate_selection(["A", "B"], {"A": 0.5}, 1)


def test_truncating_beyond_the_selection_is_harmless():
    symbols = ["A", "B"]
    assert len(truncate_selection(symbols, {s: 0.5 for s in symbols}, 10)) == 2


def test_cost_quantities_scale_quadratically():
    """The claim the thesis makes is about the search space; this is what it costs."""
    small = search_space_size(20)
    large = search_space_size(200)
    assert small["simplex_dimension"] == 19
    assert large["covariance_entries"] / small["covariance_entries"] == pytest.approx(100.0)
    assert large["flops_per_evaluation"] / small["flops_per_evaluation"] == pytest.approx(100.0)


def test_attainment_is_a_within_instance_ratio():
    assert attainment_ratio(0.5, 1.0) == pytest.approx(0.5)
    assert attainment_ratio(1.0, 1.0) == pytest.approx(1.0)


def test_attainment_against_a_degenerate_reference_is_undefined():
    """A ratio against a zero-volume front says nothing about how well the search did."""
    assert np.isnan(attainment_ratio(0.5, 0.0))
    assert np.isnan(attainment_ratio(None, 1.0))


def test_instance_labels_distinguish_no_preselection():
    assert AblationInstance("2025-06", "svm", None, ("A", "B")).label == "no-preselection"
    assert AblationInstance("2025-06", "svm", 10, ("A",) * 10).label == "10"


def test_run_records_cost_and_anytime_hypervolume():
    instance = AblationInstance("2025-06", "svm", 8, tuple(f"A{i}" for i in range(8)))
    run = run_ablation_instance(
        instance, _moments(8), seed=3,
        population_size=20, generations=5, budget_multiplier=3,
        checkpoint_evaluations=(100, 200, 300),
        hypervolume_reference=(1.1, 1.1),
    )
    assert run.n_variables == 8
    assert run.covariance_entries == 64
    assert run.wall_clock_seconds > 0
    assert run.evaluations > 0
    assert run.reference_hypervolume >= 0
    assert set(run.checkpoint_hypervolume) <= {100, 200, 300}


def test_fixed_budget_result_is_read_from_a_checkpoint():
    """One run yields the anytime profile, the fixed-budget result and the reference."""
    instance = AblationInstance("2025-06", "svm", 6, tuple(f"A{i}" for i in range(6)))
    run = run_ablation_instance(
        instance, _moments(6), seed=4,
        population_size=20, generations=5, budget_multiplier=4,
        checkpoint_evaluations=(100,),
        hypervolume_reference=(1.1, 1.1),
    )
    assert 100 in run.checkpoint_hypervolume
    ratio = run.attainment(100)
    assert np.isnan(ratio) or 0.0 <= ratio <= 1.5


def test_attainment_of_an_absent_checkpoint_is_undefined():
    instance = AblationInstance("2025-06", "svm", 6, tuple(f"A{i}" for i in range(6)))
    run = run_ablation_instance(
        instance, _moments(6), seed=5,
        population_size=20, generations=4, budget_multiplier=2,
        checkpoint_evaluations=(80,),
        hypervolume_reference=(1.1, 1.1),
    )
    assert np.isnan(run.attainment(999_999))


def test_raw_hypervolume_comparison_across_instances_is_refused():
    """Ranking selection sizes by raw hypervolume compares unrelated quantities."""
    from bmvport.ablation.search_space import raw_hypervolume_comparison_guard

    a = AblationInstance("2025-06", "svm", 10, ("A",) * 10)
    b = AblationInstance("2025-06", "svm", 40, ("A",) * 40)
    raw_hypervolume_comparison_guard([a, a])  # same instance: fine
    with pytest.raises(ValueError, match="not comparable across"):
        raw_hypervolume_comparison_guard([a, b])


def test_flat_attainment_is_not_confounded():
    """The measured case: attainment near 0.90-0.95 across sizes, largest arm highest."""
    from bmvport.ablation.search_space import confounded_by_tractability

    measured = {10: 0.915, 20: 0.917, 40: 0.942, 80: 0.897, 90: 0.952}
    assert confounded_by_tractability(measured) is False


def test_degrading_attainment_is_flagged_as_confounded():
    from bmvport.ablation.search_space import confounded_by_tractability

    starved = {10: 0.98, 40: 0.80, 90: 0.55}
    assert confounded_by_tractability(starved) is True


def test_a_single_instance_cannot_be_confounded():
    from bmvport.ablation.search_space import confounded_by_tractability

    assert confounded_by_tractability({20: 0.9}) is False


def test_cost_context_is_captured_and_comparable():
    """Wall-clock across levels means nothing if the levels ran on different machines."""
    from bmvport.ablation.search_space import cost_context

    a, b = cost_context(), cost_context()
    assert a.cpu_count > 0
    assert a.comparable_to(b)


def test_cost_context_detects_an_incomparable_run():
    from bmvport.ablation.search_space import CostContext

    a = CostContext("arm64", "p", 12, {}, "3.12.13")
    b = CostContext("arm64", "p", 4, {}, "3.12.13")
    assert a.comparable_to(b) is False


def test_flat_attainment_gives_a_slope_near_zero():
    """The measured case: solving does not get harder with size, so the contrast is readable."""
    from bmvport.ablation.search_space import attainment_versus_dimension

    fit = attainment_versus_dimension({10: 0.915, 20: 0.917, 40: 0.942, 80: 0.897, 90: 0.952})
    assert abs(fit["slope"]) < 0.05
    assert fit["n"] == 5


def test_degrading_attainment_gives_a_negative_slope():
    from bmvport.ablation.search_space import attainment_versus_dimension

    fit = attainment_versus_dimension({10: 0.98, 40: 0.80, 90: 0.55})
    assert fit["slope"] < -0.1


def test_two_points_do_not_make_a_relationship():
    from bmvport.ablation.search_space import attainment_versus_dimension

    fit = attainment_versus_dimension({10: 0.9, 90: 0.5})
    assert np.isnan(fit["slope"])


def test_front_coverage_measures_against_the_closed_form_not_another_front():
    """Coverage must be blind to optimiser bias, which is the whole reason it exists."""
    import numpy as np

    from bmvport.ablation.search_space import front_coverage

    extremes = {"available": True, "return_high": 0.02, "return_low": 0.002,
                "risk_high": 0.05, "risk_low": 0.006}
    # A front confined to a narrow band well inside the attainable range.
    objectives = np.column_stack([
        -np.linspace(0.004, 0.006, 40), np.linspace(0.007, 0.009, 40),
    ])
    result = front_coverage(objectives, extremes)

    assert result["return_coverage"] == pytest.approx(0.002 / 0.018, rel=1e-6)
    assert result["risk_coverage"] == pytest.approx(0.002 / 0.044, rel=1e-6)


def test_high_attainment_with_low_coverage_is_flagged_as_self_referential():
    """The executed sweep's exact signature: attainment 0.964, coverage near a tenth.

    Attainment compares a front to a reference produced by the same optimiser, so a search
    that never reaches the extremes scores well while spanning almost nothing. Reporting
    attainment without coverage reverses the tractability conclusion of D28, which is why
    this pairing is a guard and not a note.
    """
    from bmvport.ablation.search_space import attainment_is_self_referential

    assert attainment_is_self_referential(0.964, 0.105)
    assert not attainment_is_self_referential(0.964, 0.61)
    assert not attainment_is_self_referential(0.42, 0.105)
