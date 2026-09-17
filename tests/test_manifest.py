"""Run manifest contents and the disclosure path for substitutions."""

from __future__ import annotations

from pathlib import Path

from bmvport.config import load_config
from bmvport.manifest import (
    TRACKED_PACKAGES,
    Substitution,
    build_manifest,
    read_manifest,
    write_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


def test_manifest_records_the_reproducibility_set() -> None:
    cfg = load_config(CONFIGS / "default.yaml")
    manifest = build_manifest(cfg, data_vintage="2026-07-27")
    payload = manifest.as_dict()

    assert payload["data_vintage"] == "2026-07-27"
    assert payload["config_fingerprint"] == cfg.fingerprint()
    assert payload["feature_dictionary_version"] == "1.0.0"
    assert payload["python_version"].startswith("3.12")
    assert payload["seeds"]["optimization_seeds"] == cfg.optimization.seeds
    assert payload["seeds"]["bootstrap_seed"] == cfg.stats.bootstrap_seed
    for package in TRACKED_PACKAGES:
        assert package in payload["package_versions"]


def test_manifest_records_hardware_for_ablation_cost_comparability() -> None:
    """Ablation wall-clock is only comparable across selection sizes on fixed hardware."""
    cfg = load_config(CONFIGS / "default.yaml")
    info = build_manifest(cfg).as_dict()["platform"]
    assert info["cpu_count"] is not None
    assert "machine" in info
    assert "thread_env" in info


def test_best_arm_rule_is_recorded_as_pre_specified() -> None:
    """The comparator confrontation is circular unless this predates the results."""
    cfg = load_config(CONFIGS / "default.yaml")
    payload = build_manifest(cfg).as_dict()
    assert payload["best_arm_rule"] == cfg.benchmarks.best_arm_rule
    assert payload["best_arm_rule_predates_results"] is True


def test_revision_absence_is_explicit_not_fabricated() -> None:
    """A missing commit hash must read as missing, not as a plausible value."""
    cfg = load_config(CONFIGS / "default.yaml")
    revision = build_manifest(cfg).revision
    assert revision
    assert revision.startswith("unavailable:") or len(revision.split("+")[0]) == 40


def test_substitutions_split_by_whether_they_change_results() -> None:
    cfg = load_config(CONFIGS / "default.yaml")
    harmless = Substitution(
        component="pbc4cip",
        used_instead="reference implementation with compatibility patch",
        reason="np.object removed in NumPy 1.24",
        affects_results=False,
    )
    material = Substitution(
        component="pbc4cip",
        used_instead="balanced random forest",
        reason="reference implementation unusable",
        affects_results=True,
    )
    manifest = build_manifest(cfg, substitutions=[harmless, material])
    assert len(manifest.substitutions) == 2
    disclosable = manifest.disclosable_substitutions()
    assert [s.used_instead for s in disclosable] == ["balanced random forest"]


def test_manifest_roundtrips_to_disk(tmp_path: Path) -> None:
    cfg = load_config(CONFIGS / "default.yaml")
    manifest = build_manifest(cfg, data_vintage="2026-07-27")
    path = write_manifest(manifest, tmp_path / "run")
    assert path.name == "manifest.json"
    assert read_manifest(path) == manifest.as_dict()
