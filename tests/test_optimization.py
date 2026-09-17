"""Objectives, NSGA-III search, front extraction and the deterministic allocators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bmvport.optimization.baselines import (
    equal_weight,
    maximum_sharpe,
    minimum_volatility,
)
from bmvport.optimization.front import (
    cap_binding,
    extract_portfolios,
    hypervolume,
    knee_index,
    normalise_objectives,
)
from bmvport.optimization.nsga import derive_seed, run_nsga3
from bmvport.optimization.objectives import (
    Moments,
    apply_cap,
    cap_is_feasible,
    estimate_moments,
    portfolio_objectives,
    renormalise,
)


def _returns(n_assets: int = 8, days: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2024-01-01", periods=days, name="date")
    data = rng.normal(0.0005, 0.012, (days, n_assets))
    return pd.DataFrame(data, index=index, columns=[f"A{i}.MX" for i in range(n_assets)])


def _moments(n_assets: int = 8, seed: int = 0) -> Moments:
    frame = _returns(n_assets, seed=seed)
    return estimate_moments(frame, list(frame.columns), trailing_days=252)


# --- moments --------------------------------------------------------------------------


def test_moments_record_their_provenance() -> None:
    moments = _moments()
    assert moments.trailing_days == 252
    assert moments.estimator == "sample"
    assert moments.window_start < moments.window_end
    moments.validate()


def test_covariance_validity_is_checked() -> None:
    """A non-PSD covariance would let the optimiser exploit negative variance."""
    broken = Moments(
        symbols=("A", "B"),
        expected_returns=np.array([0.01, 0.01]),
        covariance=np.array([[1.0, 3.0], [3.0, 1.0]]),  # symmetric but indefinite
        trailing_days=10, estimator="sample", window_start="x", window_end="y",
    )
    with pytest.raises(ValueError, match="positive semi-definite"):
        broken.validate()


def test_asymmetric_covariance_is_rejected() -> None:
    broken = Moments(
        symbols=("A", "B"),
        expected_returns=np.array([0.01, 0.01]),
        covariance=np.array([[1.0, 0.5], [0.2, 1.0]]),
        trailing_days=10, estimator="sample", window_start="x", window_end="y",
    )
    with pytest.raises(ValueError, match="not symmetric"):
        broken.validate()


def test_shrinkage_estimator_is_available_for_the_sensitivity_arm() -> None:
    frame = _returns()
    shrunk = estimate_moments(
        frame, list(frame.columns), trailing_days=252, estimator="ledoit_wolf"
    )
    shrunk.validate()
    assert shrunk.estimator == "ledoit_wolf"


def test_too_short_a_window_is_rejected() -> None:
    frame = _returns(days=1)
    with pytest.raises(ValueError, match="cannot be estimated"):
        estimate_moments(frame, list(frame.columns), trailing_days=252)


def test_regularisation_is_recorded_not_silent() -> None:
    frame = _returns()
    moments = estimate_moments(
        frame, list(frame.columns), trailing_days=252, regularisation=1e-6
    )
    assert moments.regularisation == 1e-6


# --- renormalisation and the cap ------------------------------------------------------


def test_renormalisation_makes_weights_sum_to_one() -> None:
    weights = renormalise(np.array([[1.0, 2.0, 3.0], [0.5, 0.5, 0.0]]))
    assert np.allclose(weights.sum(axis=1), 1.0)


def test_degenerate_row_becomes_equal_weight_not_noise() -> None:
    """Dividing a zero-sum row by its sum would turn floating-point noise into a portfolio."""
    weights = renormalise(np.array([[0.0, 0.0, 0.0]]))
    assert np.allclose(weights, 1.0 / 3.0)


def test_cap_is_respected_and_budget_preserved() -> None:
    weights = apply_cap(np.array([[0.9, 0.05, 0.05], [0.5, 0.3, 0.2]]), 0.4)
    assert np.all(weights <= 0.4 + 1e-9)
    assert np.allclose(weights.sum(axis=1), 1.0)


def test_infeasible_cap_is_rejected() -> None:
    assert cap_is_feasible(5, 0.2) is True
    assert cap_is_feasible(3, 0.2) is False
    with pytest.raises(ValueError, match="infeasible"):
        apply_cap(np.array([[0.4, 0.3, 0.3]]), 0.2)


def test_cap_of_one_leaves_weights_untouched() -> None:
    raw = np.array([[0.6, 0.3, 0.1]])
    assert np.allclose(apply_cap(raw, 1.0), raw)


# --- objectives -----------------------------------------------------------------------


def test_objectives_are_both_minimised() -> None:
    moments = _moments()
    weights = renormalise(np.ones((4, len(moments.symbols))))
    objectives = portfolio_objectives(weights, moments)
    assert objectives.shape == (4, 2)
    # First objective is negated expected return, so a better portfolio scores lower.
    assert np.allclose(objectives[:, 0], -(weights @ moments.expected_returns))
    assert np.all(objectives[:, 1] >= 0)


def test_population_evaluation_matches_one_at_a_time() -> None:
    """The vectorised form is what makes the grid runnable; it must be exact, not close."""
    moments = _moments()
    rng = np.random.default_rng(1)
    population = renormalise(rng.random((25, len(moments.symbols))))
    batched = portfolio_objectives(population, moments)
    individually = np.vstack([portfolio_objectives(w, moments) for w in population])
    assert np.allclose(batched, individually)


def test_risk_is_never_negative_from_floating_point() -> None:
    moments = _moments()
    weights = np.zeros((1, len(moments.symbols)))
    weights[0, 0] = 1.0
    assert portfolio_objectives(weights, moments)[0, 1] >= 0.0


# --- fronts ---------------------------------------------------------------------------


def test_normalisation_maps_onto_the_unit_square() -> None:
    normalised = normalise_objectives(np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]))
    assert normalised.min() == 0.0 and normalised.max() == 1.0


def test_degenerate_objective_does_not_divide_by_zero() -> None:
    normalised = normalise_objectives(np.array([[1.0, 5.0], [2.0, 5.0]]))
    assert np.all(np.isfinite(normalised))
    assert np.allclose(normalised[:, 1], 0.0)


def test_knee_lies_between_the_extremes() -> None:
    # A convex front: the knee should be an interior point, not an endpoint.
    x = np.linspace(0.1, 1.0, 20)
    front = np.column_stack((x, 1.0 / x))
    index = knee_index(front)
    assert index is not None
    assert 0 < index < len(front) - 1


def test_knee_is_absent_on_a_front_too_small() -> None:
    assert knee_index(np.array([[1.0, 2.0], [2.0, 1.0]])) is None


def test_knee_is_absent_when_extremes_coincide() -> None:
    assert knee_index(np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])) is None


def test_extraction_returns_the_three_portfolios() -> None:
    x = np.linspace(0.1, 1.0, 20)
    objectives = np.column_stack((-x, 1.0 / x))
    weights = np.tile(np.array([0.5, 0.5]), (20, 1))
    extracted = extract_portfolios(weights, objectives)
    assert {p.allocator for p in extracted} == {"ga_min_risk", "ga_knee", "ga_max_return"}
    lowest_risk = next(p for p in extracted if p.allocator == "ga_min_risk")
    assert lowest_risk.sigma_ex_ante == pytest.approx(objectives[:, 1].min())


def test_extraction_omits_the_knee_rather_than_substituting_a_midpoint() -> None:
    objectives = np.array([[-1.0, 2.0], [-2.0, 1.0]])
    extracted = extract_portfolios(np.tile([0.5, 0.5], (2, 1)), objectives)
    assert {p.allocator for p in extracted} == {"ga_min_risk", "ga_max_return"}


def test_hypervolume_rewards_a_better_front() -> None:
    reference = (1.1, 1.1)
    poor = np.array([[0.0, 1.0], [1.0, 0.0]])
    good = np.vstack([poor, np.array([[0.2, 0.2]])])
    assert hypervolume(good, reference) > hypervolume(poor, reference)


def test_hypervolume_of_an_empty_front_is_zero() -> None:
    assert hypervolume(np.empty((0, 2)), (1.1, 1.1)) == 0.0


# --- NSGA-III -------------------------------------------------------------------------


def test_seed_derivation_is_deterministic_and_discriminating() -> None:
    assert derive_seed("abc", "2025-01", "svm", 1.0) == derive_seed("abc", "2025-01", "svm", 1.0)
    assert derive_seed("abc", "2025-01", "svm", 1.0) != derive_seed("abc", "2025-02", "svm", 1.0)
    assert derive_seed("abc", "2025-01", "svm", 1.0) != derive_seed("xyz", "2025-01", "svm", 1.0)
    assert 0 <= derive_seed("abc", "x") < 2**32


def test_search_produces_a_valid_front() -> None:
    moments = _moments(n_assets=6)
    front, _ = run_nsga3(moments, seed=7, population_size=20, generations=10)
    assert front.size > 0
    assert front.weights.shape[1] == 6
    assert np.allclose(front.weights.sum(axis=1), 1.0)
    assert front.evaluations > 0
    assert front.wall_clock_seconds > 0
    assert front.configuration["algorithm"] == "NSGA3"


def test_same_seed_reproduces_the_front_exactly() -> None:
    moments = _moments(n_assets=6)
    a, _ = run_nsga3(moments, seed=11, population_size=20, generations=8)
    b, _ = run_nsga3(moments, seed=11, population_size=20, generations=8)
    assert np.allclose(a.objectives, b.objectives)
    assert np.allclose(a.weights, b.weights)


def test_different_seeds_explore_differently() -> None:
    """Independent replicates must actually differ, or seed dispersion means nothing.

    Fronts can differ in size as well as in position, so the comparison cannot assume a
    common shape.
    """
    moments = _moments(n_assets=6)
    a, _ = run_nsga3(moments, seed=1, population_size=20, generations=8)
    b, _ = run_nsga3(moments, seed=2, population_size=20, generations=8)
    differs = a.objectives.shape != b.objectives.shape or not np.allclose(
        a.objectives, b.objectives
    )
    assert differs


def test_cap_is_enforced_in_the_returned_front() -> None:
    moments = _moments(n_assets=8)
    front, _ = run_nsga3(moments, seed=3, population_size=20, generations=8, cap=0.25)
    assert np.all(front.weights <= 0.25 + 1e-9)
    assert np.allclose(front.weights.sum(axis=1), 1.0)
    assert front.cap == 0.25


def test_checkpoints_snapshot_the_front_during_the_run() -> None:
    """The ablation reads a fixed-budget result off a longer run, not a second search."""
    moments = _moments(n_assets=6)
    _, snapshots = run_nsga3(
        moments, seed=5, population_size=20, generations=10,
        checkpoint_evaluations=(40, 100),
    )
    assert set(snapshots) == {40, 100}
    assert all(s.shape[1] == 2 for s in snapshots.values())


def test_infeasible_cap_is_refused_before_searching() -> None:
    moments = _moments(n_assets=3)
    with pytest.raises(ValueError, match="infeasible"):
        run_nsga3(moments, seed=1, population_size=10, generations=2, cap=0.2)


# --- analytic allocators --------------------------------------------------------------


def test_equal_weight_is_uniform() -> None:
    result = equal_weight(_moments(n_assets=5))
    assert result.converged
    assert np.allclose(result.weights, 0.2)


def test_minimum_volatility_beats_equal_weight_on_risk() -> None:
    moments = _moments(n_assets=8)
    minimum = minimum_volatility(moments)
    uniform = equal_weight(moments)
    assert minimum.converged
    risk_min = portfolio_objectives(minimum.weights, moments)[0, 1]
    risk_eq = portfolio_objectives(uniform.weights, moments)[0, 1]
    assert risk_min <= risk_eq + 1e-9


def test_maximum_sharpe_beats_equal_weight_on_sharpe() -> None:
    moments = _moments(n_assets=8)
    best = maximum_sharpe(moments)
    uniform = equal_weight(moments)
    assert best.converged

    def sharpe(w):
        o = portfolio_objectives(w, moments)[0]
        return -o[0] / o[1]

    assert sharpe(best.weights) >= sharpe(uniform.weights) - 1e-9


def test_allocators_are_deterministic() -> None:
    moments = _moments(n_assets=6)
    for allocator in (equal_weight, minimum_volatility, maximum_sharpe):
        first = allocator(moments)
        second = allocator(moments)
        assert np.allclose(first.weights, second.weights)


def test_allocators_respect_the_cap() -> None:
    moments = _moments(n_assets=8)
    for allocator in (equal_weight, minimum_volatility, maximum_sharpe):
        result = allocator(moments, cap=0.2)
        assert result.converged
        assert np.all(result.weights <= 0.2 + 1e-9)
        assert result.weights.sum() == pytest.approx(1.0)


def test_infeasible_cap_is_reported_not_substituted() -> None:
    moments = _moments(n_assets=3)
    for allocator in (equal_weight, minimum_volatility, maximum_sharpe):
        result = allocator(moments, cap=0.2)
        assert result.failed
        assert "infeasible" in result.message


def test_empty_selection_is_reported() -> None:
    empty = Moments(
        symbols=(), expected_returns=np.array([]), covariance=np.zeros((0, 0)),
        trailing_days=10, estimator="sample", window_start="x", window_end="y",
    )
    assert equal_weight(empty).failed


def test_cap_binding_is_detected() -> None:
    """A cap that never constrained anything must be distinguishable from one that did."""
    concentrated = np.array([[0.7, 0.2, 0.1]])
    diffuse = np.array([[0.34, 0.33, 0.33]])
    assert cap_binding(concentrated, 0.2) is True
    assert cap_binding(diffuse, 0.5) is False
    # An unconstrained setting never binds.
    assert cap_binding(concentrated, 1.0) is False


def test_cap_binds_on_small_selections_and_not_on_large_ones() -> None:
    """The measured mechanism behind D24, on synthetic but realistic moments."""
    small, _ = run_nsga3(_moments(n_assets=6), seed=9, population_size=40, generations=30)
    large, _ = run_nsga3(_moments(n_assets=60), seed=9, population_size=40, generations=30)
    assert small.weights.max() > large.weights.max()
