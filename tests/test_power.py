"""Detectable-effect reporting.

These tests pin the property that makes a null result publishable: the study must be able
to say what it could have seen, not merely that it saw nothing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bmvport.config import ConfigError, PrimaryHypothesis, load_config
from bmvport.stats.power import (
    achieved_power,
    assess_non_rejection,
    minimum_detectable_effect,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


def test_mde_shrinks_with_more_blocks() -> None:
    """More evaluation months detect smaller effects."""
    small = minimum_detectable_effect(sd=0.02, n_blocks=18)
    large = minimum_detectable_effect(sd=0.02, n_blocks=72)
    assert large < small


def test_mde_grows_with_dispersion() -> None:
    quiet = minimum_detectable_effect(sd=0.01, n_blocks=18)
    noisy = minimum_detectable_effect(sd=0.05, n_blocks=18)
    assert noisy > quiet


def test_mde_at_the_study_design() -> None:
    """A concrete bound at n=18: the effect must exceed roughly 0.7 sd/sqrt(n)."""
    mde = minimum_detectable_effect(sd=0.02, n_blocks=18, alpha=0.05, power=0.80)
    # (t_alpha + t_beta) / sqrt(18) is about 1.44 at 17 degrees of freedom
    assert 0.013 < mde < 0.016


def test_zero_dispersion_detects_everything() -> None:
    assert minimum_detectable_effect(sd=0.0, n_blocks=18) == 0.0


def test_too_few_blocks_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least two"):
        minimum_detectable_effect(sd=0.02, n_blocks=1)


def test_power_rises_with_effect_size() -> None:
    weak = achieved_power(effect=0.002, sd=0.02, n_blocks=18)
    strong = achieved_power(effect=0.02, sd=0.02, n_blocks=18)
    assert 0.0 < weak < strong <= 1.0


def test_power_at_the_mde_matches_the_target() -> None:
    """Internal consistency: an effect exactly at the MDE should reach ~80% power."""
    sd, n = 0.02, 18
    mde = minimum_detectable_effect(sd=sd, n_blocks=n, power=0.80)
    assert achieved_power(mde, sd, n) == pytest.approx(0.80, abs=0.03)


def test_non_rejection_is_characterised() -> None:
    rng = np.random.default_rng(0)
    differences = rng.normal(0.0, 0.02, 18)
    result = assess_non_rejection(differences)

    assert result.n_blocks == 18
    assert result.mde > 0
    assert 0.0 <= result.achieved_power <= 1.0
    assert "would have been detected" in result.interpretation()


def test_non_rejection_needs_two_observations() -> None:
    with pytest.raises(ValueError, match="at least two finite"):
        assess_non_rejection([0.01])


def test_non_finite_differences_are_dropped() -> None:
    result = assess_non_rejection([0.01, np.nan, 0.02, np.inf, 0.03])
    assert result.n_blocks == 3


def test_sample_standard_deviation_is_used() -> None:
    """The population form would understate the uncertainty the test actually faced."""
    values = [0.01, 0.03, 0.02, 0.04]
    result = assess_non_rejection(values)
    assert result.observed_sd == pytest.approx(np.std(values, ddof=1))


def test_confirmatory_family_is_factor_contrasts_only() -> None:
    """Head-to-head tests cannot reject on this design, so they cannot be confirmatory.

    Measured: a paired difference between two ~20-name arms carries about 2.9% monthly
    dispersion, giving a minimum detectable effect near 24% annualised, against a real
    measured gap of 6.1%. A factor contrast over 72 cells drops that to about 3.7%.
    """
    cfg = load_config(CONFIGS / "default.yaml")
    hypotheses = cfg.stats.primary_hypotheses
    assert len(hypotheses) == 3
    assert all(h.kind == "factor_effect" for h in hypotheses)
    assert {h.factor for h in hypotheses} == {"screener", "labeling", "weight_cap"}
    # The central question must be in the confirmatory set.
    assert any(h.factor == "screener" for h in hypotheses)


def test_head_to_head_cannot_be_declared_confirmatory() -> None:
    from bmvport.config import PrimaryHypothesis

    with pytest.raises(ConfigError, match="may be confirmatory"):
        PrimaryHypothesis(
            name="H_bad",
            kind="head_to_head",
            factor="none",
            contrast="best_arm_vs_tier1",
            metric="return_net",
        )


def test_measured_power_gap_between_test_designs() -> None:
    """The empirical basis for restricting the confirmatory family."""
    head_to_head = minimum_detectable_effect(sd=0.0291, n_blocks=18) * 12
    factor_contrast = minimum_detectable_effect(sd=0.0044, n_blocks=18) * 12
    assert head_to_head > 0.20  # cannot see a realistic effect
    assert factor_contrast < 0.05  # can
    assert head_to_head / factor_contrast > 4


def test_a_run_without_declared_hypotheses_is_rejected(tmp_path: Path) -> None:
    """Without a pre-specified family every reported test is exploratory."""
    import yaml

    raw = yaml.safe_load((CONFIGS / "default.yaml").read_text(encoding="utf-8"))
    raw["stats"]["primary_hypotheses"] = []
    path = tmp_path / "mutated.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="no primary hypotheses declared"):
        load_config(path, root=tmp_path.parent)


def test_an_oversized_confirmatory_family_is_rejected() -> None:
    with pytest.raises(ConfigError, match="not\n?\\s*confirmatory|confirmatory in any"):
        from bmvport.config import StatsConfig

        StatsConfig(
            alpha=0.05,
            bootstrap_resamples=1000,
            bootstrap_scheme="iid_month",
            bootstrap_seed=1,
            correction_method="holm",
            seed_aggregation="median",
            headline_set_size=10,
            primary_hypotheses=tuple(
                PrimaryHypothesis(f"H{i}", "factor_effect", "screener", "a_vs_b", "return_net")
                for i in range(6)
            ),
        )



