"""Run manifest: what was run, against which data, with which code.

A result without a manifest is not a result -- it cannot be reproduced and it cannot be
defended to a referee. The manifest records the data vintage, the resolved configuration,
the exact dependency versions, the seeds, the hardware the ablation's cost metrics were
measured on, and any substitution that had to be made for an intended component.

Two entries deserve explanation:

``best_arm_rule_predates_results``
    The comparator confrontation tests "the best proposed arm" against every comparator.
    If that arm were chosen after seeing which cell won, the test would be circular. The
    rule lives in the configuration and this flag records that it was fixed before any
    result existed.

``substitutions``
    Any component that could not be used as intended. The manuscript must disclose these,
    so they are recorded as data rather than left to memory.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

from .config import RunConfig

__all__ = ["Manifest", "Substitution", "build_manifest", "write_manifest", "read_manifest"]

# Packages whose versions materially affect results. Recorded exactly, not as a range.
TRACKED_PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "pymoo",
    "statsmodels",
    "matplotlib",
    "pyarrow",
    "PyYAML",
    "yfinance",
    "PBC4cip",
)

MANIFEST_FILENAME = "manifest.json"


@dataclass(frozen=True, slots=True)
class Substitution:
    """A component that could not be used as intended.

    Attributes:
        component: What was meant to be used, e.g. ``"pbc4cip"``.
        used_instead: What was actually used.
        reason: Why the intended component was unusable.
        affects_results: Whether stored results change because of this. A compatibility
            shim that preserves the algorithm sets this False; swapping in a different
            model sets it True, and the manuscript must then report the swap wherever
            those results appear.
    """

    component: str
    used_instead: str
    reason: str
    affects_results: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "used_instead": self.used_instead,
            "reason": self.reason,
            "affects_results": self.affects_results,
        }


@dataclass(frozen=True, slots=True)
class Manifest:
    """Everything needed to reproduce, or to audit, a run."""

    config_name: str
    config_fingerprint: str
    config: dict[str, Any]
    data_vintage: str | None
    created_at: str
    revision: str
    python_version: str
    package_versions: dict[str, str]
    platform_info: dict[str, Any]
    seeds: dict[str, Any]
    feature_dictionary_version: str
    best_arm_rule: str
    best_arm_rule_predates_results: bool
    substitutions: list[Substitution] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "config_name": self.config_name,
            "config_fingerprint": self.config_fingerprint,
            "config": self.config,
            "data_vintage": self.data_vintage,
            "created_at": self.created_at,
            "revision": self.revision,
            "python_version": self.python_version,
            "package_versions": self.package_versions,
            "platform": self.platform_info,
            "seeds": self.seeds,
            "feature_dictionary_version": self.feature_dictionary_version,
            "best_arm_rule": self.best_arm_rule,
            "best_arm_rule_predates_results": self.best_arm_rule_predates_results,
            "substitutions": [s.as_dict() for s in self.substitutions],
            "notes": self.notes,
        }

    def disclosable_substitutions(self) -> list[Substitution]:
        """Substitutions that change results and must appear in the manuscript."""
        return [s for s in self.substitutions if s.affects_results]


def _revision() -> str:
    """Return a source revision identifier.

    The project is not necessarily a git repository. Rather than fabricate a revision or
    fail, record the absence explicitly -- a reader can then tell that provenance is
    weaker than a commit hash, instead of being misled by a plausible-looking value.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=Path(__file__).resolve().parents[2],
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable:git-not-runnable"
    if out.returncode != 0:
        return "unavailable:not-a-git-repository"
    rev = out.stdout.strip()
    try:
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=Path(__file__).resolve().parents[2],
        )
        if dirty.returncode == 0 and dirty.stdout.strip():
            return f"{rev}+dirty"
    except (OSError, subprocess.SubprocessError):
        pass
    return rev


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _platform_info() -> dict[str, Any]:
    """Hardware and concurrency context.

    The ablation compares wall-clock across selection sizes, which is only meaningful if
    all compared runs executed under the same hardware and thread settings. Recording
    them is what lets a reader check that condition instead of assuming it.
    """
    thread_env = {
        var: os.environ[var]
        for var in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        )
        if var in os.environ
    }
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "thread_env": thread_env,
    }


def build_manifest(
    config: RunConfig,
    *,
    data_vintage: str | None = None,
    substitutions: list[Substitution] | None = None,
    notes: dict[str, Any] | None = None,
) -> Manifest:
    """Assemble a manifest for a run.

    Args:
        config: The resolved configuration the run used.
        data_vintage: ISO date the price cache was retrieved. ``None`` means the cache has
            not been populated yet; a manifest accompanying results must carry a value.
        substitutions: Components that could not be used as intended.
        notes: Free-form supplementary context.
    """
    return Manifest(
        config_name=config.name,
        config_fingerprint=config.fingerprint(),
        config=config.as_dict(),
        data_vintage=data_vintage,
        created_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        revision=_revision(),
        python_version=sys.version,
        package_versions=_package_versions(),
        platform_info=_platform_info(),
        seeds={
            "optimization_seeds": config.optimization.seeds,
            "ablation_seeds": config.ablation.seeds,
            "bootstrap_seed": config.stats.bootstrap_seed,
            "derivation": (
                "per-run seeds are derived deterministically from the run fingerprint and "
                "the cell key; see optimization.seeding"
            ),
        },
        feature_dictionary_version=config.features.dictionary_version,
        best_arm_rule=config.benchmarks.best_arm_rule,
        best_arm_rule_predates_results=True,
        substitutions=list(substitutions or []),
        notes=dict(notes or {}),
    )


def write_manifest(manifest: Manifest, directory: str | Path) -> Path:
    """Write ``manifest.json`` into ``directory`` and return its path."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / MANIFEST_FILENAME
    path.write_text(
        json.dumps(manifest.as_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def read_manifest(path: str | Path) -> dict[str, Any]:
    """Read a manifest back as a plain mapping."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
