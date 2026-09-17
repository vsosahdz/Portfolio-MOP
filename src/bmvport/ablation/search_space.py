"""Selection-size ablation: what preselecting stocks actually buys.

The thesis asserts that screening "narrows the search space for the portfolio optimization
stage" and that a good screener makes optimisation "a straightforward task by reducing
decision variables". Neither claim is ever measured. This module measures both, and
separates them from a third thing they are easily confused with.

**Three kinds of evidence, because they can disagree.**

*Objectives.* Does out-of-sample performance improve? The finance question.

*Search cost.* Wall-clock, objective evaluations, covariance dimension. The engineering
question, and the one the thesis's claim is literally about.

*Tractability.* Can the optimiser actually solve the larger instance within a realistic
budget? This is the confound, and it is not optional. The decision space is the
``(n-1)``-simplex, so screening does not merely shrink it -- it changes the instance's
difficulty. Holding the budget fixed at 10,000 evaluations gives a ninety-variable problem
the same effort as a twenty-variable one. A fixed-budget result showing "screening wins" may
therefore be showing "the optimiser was starved on the larger problem", which is an
attractive conclusion reached invalidly.

**Comparing across selection sizes.** Raw hypervolume is meaningless across levels: a
different ``k`` means a different asset subset, a different front and different objective
ranges. What is comparable is the **within-instance attainment ratio** -- hypervolume at a
given budget divided by that same instance's reference hypervolume from a long run. That
ratio is dimensionless and answers the tractability question directly: if attainment degrades
as ``k`` grows, the unscreened arm is under-solved and the fixed-budget contrast is
confounded rather than informative.

Each instance is run **once** with an extended budget while recording hypervolume at
checkpoints. That single run yields the anytime profile, the fixed-budget result read off the
10,000-evaluation checkpoint, and the reference front -- rather than launching separate
searches whose only difference would be where they stopped.
"""

from __future__ import annotations

import resource
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from ..optimization.front import hypervolume
from ..optimization.nsga import run_nsga3
from ..optimization.objectives import Moments

__all__ = [
    "AblationInstance",
    "AblationRun",
    "truncate_selection",
    "search_space_size",
    "run_ablation_instance",
    "attainment_ratio",
    "raw_hypervolume_comparison_guard",
    "confounded_by_tractability",
    "analytic_objective_extremes",
    "front_coverage",
    "attainment_is_self_referential",
    "CostContext",
    "cost_context",
    "attainment_versus_dimension",
]


@dataclass(frozen=True, slots=True)
class AblationInstance:
    """One (month, screener, selection size) problem."""

    month: str
    screener: str
    selection_size: int | None  # None means no preselection
    symbols: tuple[str, ...]

    @property
    def n_variables(self) -> int:
        return len(self.symbols)

    @property
    def label(self) -> str:
        return "no-preselection" if self.selection_size is None else str(self.selection_size)


@dataclass(frozen=True, slots=True)
class AblationRun:
    """Result of one extended-budget run with checkpointed hypervolume."""

    instance: AblationInstance
    seed: int
    n_variables: int
    covariance_entries: int
    evaluations: int
    wall_clock_seconds: float
    peak_memory_bytes: int
    reference_hypervolume: float
    checkpoint_hypervolume: Mapping[int, float] = field(default_factory=dict)

    def attainment(self, budget: int) -> float:
        """Fraction of this instance's own reference hypervolume reached at ``budget``."""
        return attainment_ratio(self.checkpoint_hypervolume.get(budget), self.reference_hypervolume)


def truncate_selection(
    symbols: Sequence[str], scores: Mapping[str, float], size: int | None
) -> list[str]:
    """Keep the top ``size`` symbols by decision score, or all of them when ``size`` is None.

    Ties are broken by symbol name so the truncation is deterministic; an arbitrary tie-break
    would make the ablation irreproducible in exactly the cells where scores are flat.

    Raises:
        ValueError: if a symbol has no score. Silently dropping it would shrink the
            selection in a way the recorded size would not reflect.
    """
    missing = [s for s in symbols if s not in scores]
    if missing:
        raise ValueError(f"no decision score for {missing[:5]}")
    if size is None:
        return list(symbols)
    ordered = sorted(symbols, key=lambda s: (-float(scores[s]), s))
    return ordered[: min(size, len(ordered))]


def search_space_size(n_variables: int) -> dict[str, int]:
    """Cost quantities that scale with the number of decision variables.

    The simplex dimension is what the thesis's claim is about; the covariance entry count and
    the per-evaluation flop estimate are what it costs in practice.
    """
    return {
        "n_variables": int(n_variables),
        "simplex_dimension": int(max(n_variables - 1, 0)),
        "covariance_entries": int(n_variables * n_variables),
        "flops_per_evaluation": int(2 * n_variables * n_variables),
    }


def attainment_ratio(value: float | None, reference: float) -> float:
    """Within-instance attainment: hypervolume reached over reference hypervolume.

    Returns ``nan`` rather than a misleading number when the reference is degenerate; a
    ratio against a zero-volume front says nothing about how well the search did.
    """
    if value is None or not np.isfinite(reference) or reference <= 0:
        return float("nan")
    return float(value / reference)


def run_ablation_instance(
    instance: AblationInstance,
    moments: Moments,
    *,
    seed: int,
    population_size: int,
    generations: int,
    budget_multiplier: int,
    checkpoint_evaluations: Sequence[int],
    hypervolume_reference: tuple[float, float],
    cap: float = 1.0,
) -> AblationRun:
    """Run one instance at the extended budget, recording cost and anytime hypervolume.

    The extended budget is the base budget times ``budget_multiplier``; the reference front
    is the run's final non-dominated set, and the fixed-budget result is read off the
    checkpoint at the base budget.
    """
    extended_generations = generations * budget_multiplier
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    front, snapshots = run_nsga3(
        moments,
        seed=seed,
        population_size=population_size,
        generations=extended_generations,
        cap=cap,
        checkpoint_evaluations=tuple(checkpoint_evaluations),
    )

    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    reference = hypervolume(front.objectives, hypervolume_reference)
    checkpoints = {
        int(k): hypervolume(v, hypervolume_reference) for k, v in snapshots.items()
    }

    return AblationRun(
        instance=instance,
        seed=int(seed),
        n_variables=instance.n_variables,
        covariance_entries=instance.n_variables ** 2,
        evaluations=front.evaluations,
        wall_clock_seconds=front.wall_clock_seconds,
        peak_memory_bytes=int(max(after, before)),
        reference_hypervolume=reference,
        checkpoint_hypervolume=checkpoints,
    )


def raw_hypervolume_comparison_guard(instances: Sequence[AblationInstance]) -> None:
    """Refuse a comparison of raw hypervolume across different instances.

    Different selection sizes mean different asset subsets, different Pareto fronts and
    different objective ranges. A table ranking selection sizes by raw hypervolume would look
    like a search-quality comparison and would be a comparison of unrelated quantities.
    Only the within-instance attainment ratio is defined across levels.

    Raises:
        ValueError: whenever more than one distinct instance is supplied.
    """
    distinct = {(i.month, i.screener, i.selection_size) for i in instances}
    if len(distinct) > 1:
        raise ValueError(
            "raw hypervolume is not comparable across ablation instances; use "
            "attainment_ratio, which is defined within an instance"
        )


def confounded_by_tractability(
    attainments: Mapping[int, float], *, degradation_tolerance: float = 0.05
) -> bool:
    """Whether the fixed-budget contrast is confounded by the larger instance being starved.

    The fixed-budget comparison across selection sizes is only interpretable if the optimiser
    solved each instance comparably well. If attainment falls materially as the variable count
    grows, an apparent win for screening may be a win for having handed the optimiser an
    easier problem, and the contrast must be reported as confounded rather than as evidence.

    Args:
        attainments: Attainment at the main-grid budget, keyed by decision-variable count.
        degradation_tolerance: Drop from the smallest instance that counts as material.

    Returns:
        True when the largest instance attains materially less than the smallest.
    """
    usable = {k: v for k, v in attainments.items() if np.isfinite(v)}
    if len(usable) < 2:
        return False
    smallest = usable[min(usable)]
    largest = usable[max(usable)]
    return bool(smallest - largest > degradation_tolerance)


@dataclass(frozen=True, slots=True)
class CostContext:
    """Hardware and concurrency under which cost metrics were measured.

    Wall-clock is only comparable across ablation levels if every level ran under the same
    conditions. Recording them is what lets a reader check that rather than assume it -- and
    what stops a timing table gathered across a machine change from being read as a scaling
    result.
    """

    machine: str
    processor: str
    cpu_count: int
    thread_env: Mapping[str, str]
    python_version: str

    def comparable_to(self, other: "CostContext") -> bool:
        return (
            self.machine == other.machine
            and self.cpu_count == other.cpu_count
            and dict(self.thread_env) == dict(other.thread_env)
        )


def cost_context() -> CostContext:
    """Capture the current hardware and threading context."""
    import os
    import platform
    import sys

    tracked = (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    )
    return CostContext(
        machine=platform.machine(),
        processor=platform.processor(),
        cpu_count=os.cpu_count() or 0,
        thread_env={v: os.environ[v] for v in tracked if v in os.environ},
        python_version=sys.version.split()[0],
    )


def attainment_versus_dimension(
    attainments: Mapping[int, float]
) -> dict[str, float]:
    """Relate attainment at the main budget to the decision-variable count.

    Fitted on ``log(n_variables)`` because the cost quantities scale as powers of it, so a
    slope in log space is the natural statement of "does solving get harder with size".

    A slope indistinguishable from zero is the informative outcome here: it says the larger
    instances were solved as completely as the small ones, and therefore that the
    fixed-budget contrast across selection sizes is interpretable.

    Returns the slope, intercept, the fraction of variance explained, and the observation
    count. Returns ``nan`` slopes when fewer than three usable points are supplied, rather
    than fitting a line through two points and reporting it as a relationship.
    """
    usable = {k: v for k, v in attainments.items() if k > 0 and np.isfinite(v)}
    if len(usable) < 3:
        return {"slope": float("nan"), "intercept": float("nan"),
                "r_squared": float("nan"), "n": float(len(usable))}
    x = np.log(np.array(sorted(usable), dtype=float))
    y = np.array([usable[int(k)] for k in sorted(usable)], dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    residual = float(np.sum((y - predicted) ** 2))
    total = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - residual / total if total > 0 else float("nan")
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_squared),
        "n": float(len(usable)),
    }



def analytic_objective_extremes(moments) -> dict[str, float]:
    """Closed-form extremes of the unconstrained mean-variance problem on the simplex.

    Maximum expected return is the whole budget in the single highest-mean asset; minimum
    variance is the analytic portfolio the deterministic allocators already compute. Both
    are exact, so they bound what any optimiser on this instance could reach.
    """
    from ..optimization.baselines import minimum_volatility

    mu = np.asarray(moments.expected_returns, dtype=float)
    minimum = minimum_volatility(moments, cap=1.0)
    if minimum.failed:
        return {"available": False}
    vertex = np.zeros(mu.size)
    vertex[int(np.argmax(mu))] = 1.0
    return {
        "available": True,
        "return_high": float(mu.max()),
        "return_low": float(minimum.weights @ mu),
        "risk_high": float(np.sqrt(vertex @ moments.covariance @ vertex)),
        "risk_low": float(np.sqrt(minimum.weights @ moments.covariance @ minimum.weights)),
    }


def front_coverage(objectives: np.ndarray, extremes: Mapping[str, float]) -> dict[str, float]:
    """Fraction of the analytically attainable range that a produced front actually spans.

    This is the quantity ``attainment_ratio`` cannot measure. Attainment compares a front
    against a reference front built by the same optimiser at a larger budget, so a bias
    shared by both is invisible to it: if the search cannot reach the extremes at all, both
    fronts stay in the same narrow band and the ratio reports success. Coverage compares
    against the closed-form bound instead, which no amount of shared bias can move.
    """
    if not extremes.get("available", False):
        return {"available": False}
    produced_return = abs(float((-objectives[:, 0]).max() - (-objectives[:, 0]).min()))
    produced_risk = abs(float(objectives[:, 1].max() - objectives[:, 1].min()))
    span_return = abs(extremes["return_high"] - extremes["return_low"])
    span_risk = abs(extremes["risk_high"] - extremes["risk_low"])
    return {
        "available": True,
        "return_coverage": produced_return / span_return if span_return > 0 else float("nan"),
        "risk_coverage": produced_risk / span_risk if span_risk > 0 else float("nan"),
        "attainable_return_span": span_return,
        "attainable_risk_span": span_risk,
    }


def attainment_is_self_referential(attainment: float, coverage: float) -> bool:
    """True where attainment looks healthy while coverage says the front is a narrow band.

    The executed sweep hit exactly this: attainment rose with problem size, to 0.964 at
    ninety-four variables, while return coverage fell to about a tenth of the attainable
    range. Reading the first number alone reverses the conclusion the second supports, so a
    reported attainment is required to carry its coverage beside it.
    """
    return bool(np.isfinite(attainment) and np.isfinite(coverage)
                and attainment >= 0.90 and coverage < 0.25)
