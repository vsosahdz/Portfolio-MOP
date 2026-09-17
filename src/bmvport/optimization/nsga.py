"""NSGA-III search over the bi-objective mean-variance problem.

**Seeds are derived, not chosen.** Each run's seed comes from a hash of the run
configuration's fingerprint and the cell key, so re-running any cell reproduces its front
exactly while different cells remain independent. A hand-maintained seed list would drift
from the cells it labels; a derived seed cannot.

**The evaluation budget is population x generations**, and it is recorded with every front.
The ablation of D19 reads fixed-budget results off a checkpoint of a longer run rather than
launching a separate search, so the budget has to be an explicit, comparable quantity rather
than an implicit consequence of the termination criterion.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from .objectives import Moments, apply_cap, cap_is_feasible, portfolio_objectives, renormalise

__all__ = ["FrontResult", "derive_seed", "run_nsga3"]


@dataclass(frozen=True, slots=True)
class FrontResult:
    """One NSGA-III run: its non-dominated set and the context to reproduce it."""

    weights: np.ndarray
    objectives: np.ndarray
    symbols: tuple[str, ...]
    seed: int
    population_size: int
    generations: int
    evaluations: int
    cap: float
    wall_clock_seconds: float
    configuration: Mapping[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.objectives)

    @property
    def expected_returns(self) -> np.ndarray:
        """Expected returns, sign-restored from the minimised first objective."""
        return -self.objectives[:, 0]

    @property
    def risks(self) -> np.ndarray:
        return self.objectives[:, 1]


def derive_seed(fingerprint: str, *key_parts: Any) -> int:
    """Deterministic seed for one cell.

    The same configuration and cell always yield the same seed; a different month, screener,
    cap or replicate index yields a different one.
    """
    payload = "|".join([fingerprint, *(str(p) for p in key_parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def run_nsga3(
    moments: Moments,
    *,
    seed: int,
    population_size: int = 100,
    generations: int = 100,
    cap: float = 1.0,
    checkpoint_evaluations: tuple[int, ...] = (),
) -> tuple[FrontResult, dict[int, np.ndarray]]:
    """Search the bi-objective problem and return the non-dominated set.

    Args:
        moments: Ex-ante estimates defining both objectives.
        seed: Random seed, normally from :func:`derive_seed`.
        population_size: Individuals per generation.
        generations: Generations to run.
        cap: Maximum per-asset weight. ``1.0`` reproduces the unconstrained formulation.
        checkpoint_evaluations: Evaluation counts at which to snapshot the current
            non-dominated objectives. Used by the ablation to read a fixed-budget result off
            a longer run instead of launching a second search.

    Returns:
        The front, and a mapping from checkpoint evaluation count to the objectives of the
        non-dominated set at that point.

    Raises:
        ValueError: if the cap makes the budget constraint unsatisfiable for this selection.
    """
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize
    from pymoo.util.ref_dirs import get_reference_directions

    n_assets = len(moments.symbols)
    if not cap_is_feasible(n_assets, cap):
        raise ValueError(
            f"cap {cap} over {n_assets} assets is infeasible; record the cell as skipped"
        )

    snapshots: dict[int, np.ndarray] = {}
    wanted = sorted(checkpoint_evaluations)
    counter = {"evaluations": 0}

    class _MeanVariance(Problem):
        def __init__(self) -> None:
            super().__init__(n_var=n_assets, n_obj=2, xl=0.0, xu=1.0)

        def _evaluate(self, X, out, *args, **kwargs):  # noqa: ANN001
            weights = apply_cap(X, cap) if cap < 1.0 else renormalise(X)
            objectives = portfolio_objectives(weights, moments)
            out["F"] = objectives
            counter["evaluations"] += len(X)
            for target in wanted:
                if target not in snapshots and counter["evaluations"] >= target:
                    snapshots[target] = _non_dominated(objectives).copy()

    # NSGA-III is defined around reference directions; for two objectives the Das-Dennis
    # simplex reduces to an evenly spaced set along the front, sized to the population.
    reference_directions = get_reference_directions(
        "das-dennis", 2, n_partitions=max(population_size - 1, 1)
    )
    algorithm = NSGA3(pop_size=population_size, ref_dirs=reference_directions)

    started = time.perf_counter()
    result = minimize(
        _MeanVariance(),
        algorithm,
        ("n_gen", generations),
        seed=seed,
        verbose=False,
        save_history=False,
    )
    elapsed = time.perf_counter() - started

    raw = np.atleast_2d(result.X)
    weights = apply_cap(raw, cap) if cap < 1.0 else renormalise(raw)
    objectives = np.atleast_2d(result.F)

    return (
        FrontResult(
            weights=weights,
            objectives=objectives,
            symbols=moments.symbols,
            seed=int(seed),
            population_size=int(population_size),
            generations=int(generations),
            evaluations=int(counter["evaluations"]),
            cap=float(cap),
            wall_clock_seconds=float(elapsed),
            configuration={
                "algorithm": "NSGA3",
                "reference_directions": "das-dennis",
                "n_partitions": max(population_size - 1, 1),
                "covariance_estimator": moments.estimator,
                "covariance_trailing_days": moments.trailing_days,
                "covariance_window": [moments.window_start, moments.window_end],
            },
        ),
        snapshots,
    )


def _non_dominated(objectives: np.ndarray) -> np.ndarray:
    """Non-dominated subset of a minimisation objective array."""
    points = np.atleast_2d(objectives)
    keep = np.ones(len(points), dtype=bool)
    for i, point in enumerate(points):
        if not keep[i]:
            continue
        dominated = np.all(points <= point, axis=1) & np.any(points < point, axis=1)
        if dominated.any():
            keep[i] = False
    return points[keep]
