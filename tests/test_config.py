"""Configuration loading and the invariants that must fail loudly."""

from __future__ import annotations

import copy
import datetime as dt
from pathlib import Path

import pytest
import yaml

from bmvport.config import (
    ALLOCATORS,
    ConfigError,
    FeatureConfig,
    WindowConfig,
    load_config,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


@pytest.fixture(params=["default.yaml", "smoke.yaml"])
def config_path(request: pytest.FixtureRequest) -> Path:
    return CONFIGS / request.param


def test_shipped_configs_load(config_path: Path) -> None:
    cfg = load_config(config_path)
    assert cfg.name
    assert cfg.paths.root == ROOT
    assert cfg.features.expected_daily_indicators == 43
    assert cfg.features.expected_monthly_features == 129


def test_results_window_is_confined_to_2025_2026() -> None:
    """Every evaluation month lies inside 2025-2026; earlier data is model input only."""
    cfg = load_config(CONFIGS / "default.yaml")
    months = cfg.windows.eval_months()
    assert len(months) == 18
    assert months[0] == "2025-01"
    assert months[-1] == "2026-06"
    assert all(m[:4] in ("2025", "2026") for m in months)
    # The download reaches back only to supply warm-up, training, covariance and momentum.
    assert cfg.windows.download_start.isoformat() == "2023-01-01"


def test_default_grid_dimensions() -> None:
    """The design's stated grid size must be what the config actually implies."""
    cfg = load_config(CONFIGS / "default.yaml")
    # 3 labeling x 6 screeners x 6 allocators x 2 caps = 216 cells
    assert cfg.factorial_cell_count() == 216
    # 216 cells x 18 months = 3,888 portfolio-months
    assert cfg.factorial_cell_count() * len(cfg.windows.eval_months()) == 3888
    # One front serves all three GA allocators, so allocators do not multiply here:
    # 3 x 6 x 2 x 18 x 10 seeds = 6,480 NSGA-III runs
    assert cfg.nsga_run_count() == 6480
    assert cfg.optimization.evaluation_budget == 10_000


def test_allocator_set_is_six() -> None:
    assert len(ALLOCATORS) == 6


def test_screener_taxonomy() -> None:
    """Seven levels with a rationale each, not an accumulation of similar models."""
    from bmvport.config import SCREENERS

    assert SCREENERS[0] == "all_stocks"  # the no-screening baseline
    thesis_set = {"svm", "random_forest", "neural_network", "pbc4cip"}
    assert thesis_set <= set(SCREENERS)  # methodological fidelity
    assert "xgboost" in SCREENERS  # one modern booster, not three variants
    assert "tabpfn" in SCREENERS  # one tabular foundation model
    assert not {"lightgbm", "catboost"} & set(SCREENERS)


def test_pbc4cip_omission_is_disclosed_as_affecting_results() -> None:
    """An omitted screener must not read as one that was tried and did poorly.

    Measured cost at the reference ensemble size is 137 hours for the grid, and no
    conclusion depends on it: a seventh screener returning 100% annualised would leave the
    pooled H1 effect below the design's detectable bound.
    """
    from bmvport.screening.screeners import availability_substitutions

    cfg = load_config(CONFIGS / "default.yaml")
    assert "pbc4cip" not in cfg.screening.screeners
    disclosed = availability_substitutions(cfg.screening.screeners)
    omission = [s for s in disclosed if s.component == "pbc4cip"]
    assert omission and omission[0].affects_results is True
    assert "137 h" in omission[0].reason


def test_fingerprint_is_stable_and_sensitive() -> None:
    a = load_config(CONFIGS / "default.yaml")
    b = load_config(CONFIGS / "default.yaml")
    assert a.fingerprint() == b.fingerprint()
    assert a.fingerprint() != load_config(CONFIGS / "smoke.yaml").fingerprint()


def test_feature_windows_must_imply_declared_counts() -> None:
    """A window change that alters dimensionality must not pass silently.

    The 43/129 counts are the contract with the published feature description, so drifting
    away from them has to be an error rather than a new, undocumented feature set.
    """
    with pytest.raises(ConfigError, match="daily columns"):
        FeatureConfig(
            ma_windows=(5, 10, 20),  # one fewer family member than the table
            ratio_windows=(1, 10, 20, 40, 60),
            rsi_window=14,
            mfi_window=14,
            stochastic_window=14,
            min_trading_days_per_month=15,
        )


def test_monthly_count_must_match_aggregations() -> None:
    with pytest.raises(ConfigError, match="expected_monthly_features"):
        FeatureConfig(
            ma_windows=(5, 10, 20, 40, 60),
            ratio_windows=(1, 10, 20, 40, 60),
            rsi_window=14,
            mfi_window=14,
            stochastic_window=14,
            min_trading_days_per_month=15,
            aggregations=("mean", "last"),  # 43 x 2 = 86, not 129
        )


def test_warmup_is_longest_window() -> None:
    cfg = load_config(CONFIGS / "default.yaml")
    assert cfg.features.warmup_days() == 60


def test_evaluation_window_must_follow_download_start() -> None:
    with pytest.raises(ConfigError, match="must start after download_start"):
        WindowConfig(
            download_start=dt.date(2025, 1, 1),
            download_end=dt.date(2026, 6, 30),
            eval_start_month="2025-01",
            eval_end_month="2026-06",
        )


def test_full_window_coverage_flag_is_explicit() -> None:
    """The coverage restriction introduces survivorship bias, so it must be visible."""
    cfg = load_config(CONFIGS / "default.yaml")
    assert cfg.universe.require_full_window_coverage is True
    assert cfg.universe.min_window_coverage_fraction == 0.95


def test_impossible_calendar_is_rejected(tmp_path: Path) -> None:
    """Warm-up, label lag and training history must fit before the first evaluation month.

    Without this guard the earliest evaluation months would train on one month of data, or
    none, and the screening stage would emit noise indistinguishable from a result.
    """

    def mutate(raw: dict) -> None:
        raw["windows"]["download_start"] = "2024-09-01"
        raw["windows"]["eval_start_month"] = "2024-12"

    path = _mutated(CONFIGS / "default.yaml", tmp_path, mutate)
    with pytest.raises(ConfigError, match="months after download_start but needs at least"):
        load_config(path, root=tmp_path.parent)


def test_momentum_holds_enough_positions_to_be_a_fair_baseline() -> None:
    """A quintile, not the conventional decile.

    The decile convention assumes universes of hundreds of names. On this 93-instrument
    universe it would hold nine positions, and a baseline that concentrated loses on
    idiosyncratic variance rather than on the absence of momentum.
    """
    cfg = load_config(CONFIGS / "default.yaml")
    assert cfg.benchmarks.momentum_selection_fraction == 0.20


def test_overly_concentrated_momentum_is_rejected(tmp_path: Path) -> None:
    """The guard that would have caught the original decile setting."""

    def mutate(raw: dict) -> None:
        raw["benchmarks"]["momentum_selection_fraction"] = 0.10

    path = _mutated(CONFIGS / "default.yaml", tmp_path, mutate)
    with pytest.raises(ConfigError, match="fair baseline rather than a"):
        load_config(path, root=tmp_path.parent)


def test_conventional_momentum_is_feasible() -> None:
    """The 12-1 baseline must be genuinely available, not just named in the config."""
    cfg = load_config(CONFIGS / "default.yaml")
    assert cfg.benchmarks.momentum_formation_months == 12
    assert cfg.benchmarks.momentum_skip_months == 1
    assert cfg.optimization.covariance_trailing_days == 252


def test_infeasible_momentum_window_is_rejected(tmp_path: Path) -> None:
    """Naming 12-1 while only part of it exists would misreport a standard baseline.

    A 2024-01 download start with evaluation from 2025-01 clears the labeling calendar but
    leaves only 11 months of returns, so the conventional 12-1 formation does not exist for
    the first evaluation month. This is exactly the case that made 2024 a worse choice than
    2023 despite both yielding 18 evaluation blocks.
    """

    def mutate(raw: dict) -> None:
        raw["windows"]["download_start"] = "2024-01-01"

    path = _mutated(CONFIGS / "default.yaml", tmp_path, mutate)
    with pytest.raises(ConfigError, match="momentum comparator needs"):
        load_config(path, root=tmp_path.parent)


def _mutated(base: Path, tmp_path: Path, mutate) -> Path:
    raw = yaml.safe_load(base.read_text(encoding="utf-8"))
    mutate(raw)
    out = tmp_path / "mutated.yaml"
    out.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return out


def test_gross_only_cost_scenarios_are_rejected(tmp_path: Path) -> None:
    """A study that reports only gross returns must not be configurable."""

    def mutate(raw: dict) -> None:
        raw["evaluation"]["cost_scenarios_bps"] = [0.0]
        raw["evaluation"]["primary_cost_scenario_bps"] = 0.0

    path = _mutated(CONFIGS / "default.yaml", tmp_path, mutate)
    with pytest.raises(ConfigError, match="must not be the zero-cost case"):
        load_config(path, root=tmp_path.parent)


def test_ablation_requires_a_no_preselection_level(tmp_path: Path) -> None:
    """Without the null level the ablation cannot answer its own question."""

    def mutate(raw: dict) -> None:
        raw["ablation"]["selection_sizes"] = [10, 20, 40]

    path = _mutated(CONFIGS / "default.yaml", tmp_path, mutate)
    with pytest.raises(ConfigError, match="no preselection"):
        load_config(path, root=tmp_path.parent)


def test_ablation_checkpoints_must_include_main_grid_budget(tmp_path: Path) -> None:
    """The fixed-budget result is read from a checkpoint, never from a separate run."""

    def mutate(raw: dict) -> None:
        raw["ablation"]["checkpoint_evaluations"] = [1000, 25_000, 100_000]

    path = _mutated(CONFIGS / "default.yaml", tmp_path, mutate)
    with pytest.raises(ConfigError, match="main grid budget"):
        load_config(path, root=tmp_path.parent)


def test_ablation_screeners_must_exist_in_the_run(tmp_path: Path) -> None:
    def mutate(raw: dict) -> None:
        raw["ablation"]["screeners"] = ["neural_network"]
        raw["screening"]["screeners"] = ["all_stocks", "svm", "xgboost"]

    path = _mutated(CONFIGS / "default.yaml", tmp_path, mutate)
    with pytest.raises(ConfigError, match="screeners not in the run"):
        load_config(path, root=tmp_path.parent)


def test_unknown_screener_is_rejected(tmp_path: Path) -> None:
    def mutate(raw: dict) -> None:
        raw["screening"]["screeners"] = ["all_stocks", "lightgbm"]

    path = _mutated(CONFIGS / "default.yaml", tmp_path, mutate)
    with pytest.raises(ConfigError, match="unknown screeners"):
        load_config(path, root=tmp_path.parent)


def test_config_is_immutable() -> None:
    """Stages must not mutate the config they were handed; the manifest describes it."""
    cfg = load_config(CONFIGS / "default.yaml")
    with pytest.raises((AttributeError, TypeError)):
        cfg.optimization.seeds = 99  # type: ignore[misc]


def test_as_dict_is_json_serialisable() -> None:
    import json

    cfg = load_config(CONFIGS / "default.yaml")
    payload = json.dumps(cfg.as_dict(), sort_keys=True)
    assert "hypervolume_reference" in payload
    assert copy.deepcopy(cfg.as_dict()) == cfg.as_dict()


def test_tabpfn_without_a_pinned_checkpoint_is_rejected(tmp_path: Path) -> None:
    """An absent grid yields empty settings, which resolves to a gated checkpoint."""

    def mutate(raw: dict) -> None:
        raw["screening"]["hyperparameter_grids"].pop("tabpfn", None)

    path = _mutated(CONFIGS / "default.yaml", tmp_path, mutate)
    with pytest.raises(ConfigError, match="names no\\s+model_path"):
        load_config(path, root=tmp_path.parent)


def test_shipped_configs_pin_a_public_tabpfn_checkpoint(config_path: Path) -> None:
    cfg = load_config(config_path)
    if "tabpfn" not in cfg.screening.screeners:
        pytest.skip("tabpfn not in this configuration")
    assert cfg.screening.hyperparameter_grids["tabpfn"]["model_path"] == [
        "tabpfn-v2-classifier.ckpt"
    ]


def test_threads_are_pinned_before_numerical_libraries_load() -> None:
    """XGBoost and PyTorch segfault together unless every OpenMP runtime is pinned.

    The failure is order-dependent, so a grid could survive or die depending on which
    screener ran first. Importing the package must have already prevented it.
    """
    import os

    from bmvport._threading import THREAD_VARIABLES, active_thread_settings

    active = active_thread_settings()
    assert set(active) == set(THREAD_VARIABLES)
    assert all(v == "1" for v in active.values())
    assert os.environ["OMP_NUM_THREADS"] == "1"


def test_existing_thread_setting_is_respected() -> None:
    """A caller who configured threading deliberately should not have it changed."""
    import os

    from bmvport._threading import pin_threads

    os.environ["OMP_NUM_THREADS"] = "4"
    try:
        pin_threads(1)
        assert os.environ["OMP_NUM_THREADS"] == "4"
        pin_threads(1, override=True)
        assert os.environ["OMP_NUM_THREADS"] == "1"
    finally:
        os.environ["OMP_NUM_THREADS"] = "1"
