"""Component ablation: what each piece of the proposal is worth.

The search-space ablation asks whether preselecting stocks is justified at all. This asks a
different question -- given that the pipeline has several components, what does each
contribute? A reader deciding whether to adopt the approach needs that breakdown.

**Two ablations, because either alone misleads.**

The *nested ladder* starts from a minimal baseline and adds one component at a time. It is
**order-dependent**: in a system where components interact, what a piece appears to
contribute depends on what is already present.

*Leave-one-out* starts from the full proposal and removes one component at a time. It is
order-independent, but it answers "what is this worth given everything else", which is not
the same question.

Reporting one without the other invites exactly the question the other answers, so both are
computed and disagreements between them are reported rather than resolved.

**The constraint that keeps this honest.** The minimum detectable effect for a head-to-head
comparison on this design is around 24% annualised. Most rung deltas will fall well below it.
Every effect is therefore reported with its bootstrap interval *and* that bound, and one that
does not clear it is labelled below the design's resolution rather than described as a
contribution. Ablation tables commonly fail here: a column of small positive numbers reads as
a chain of contributions when it is a chain of noise.

Everything is computed from the tidy results table, so the ablation and the headline figures
cannot disagree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ..stats.power import minimum_detectable_effect

__all__ = [
    "Rung",
    "ComponentEffect",
    "LADDER",
    "COMPONENTS",
    "effect_between",
    "ladder_effects",
    "leave_one_out_effects",
    "compare_ablations",
    "ladder_is_well_formed",
]


@dataclass(frozen=True, slots=True)
class Rung:
    """One configuration on the ladder, and the component it adds."""

    name: str
    adds: str
    selector: Mapping[str, object]
    description: str


# Each rung differs from its predecessor by exactly one component, and every rung that names
# a factor pins ALL of them. A rung that left a factor unpinned would silently average over
# the levels the next rung fixes, so its "one component" difference would really be a
# difference between an aggregate and a specific configuration. ``ladder_is_well_formed``
# checks this, and a test enforces it.
#
# A0 and A1 are Tier 1 comparators; A2 onward are cells of the factorial, so the ladder is
# largely a reading of the results table rather than a separate set of runs.
LADDER: tuple[Rung, ...] = (
    Rung("A0", "baseline",
         {"arm_kind": "comparator", "allocator": "equal_weight_universe"},
         "1/N over the admitted universe: no screening, no optimisation"),
    Rung("A1", "optimisation",
         {"arm_kind": "comparator", "allocator": "mv_analytic_universe"},
         "analytic mean-variance over the same universe"),
    Rung("A2", "multi_objective",
         {"arm_kind": "proposed", "screener": "all_stocks", "allocator": "ga_max_return",
          "labeling": "historical", "weight_cap": 1.0},
         "NSGA-III search, still without screening"),
    Rung("A3", "screening",
         {"arm_kind": "proposed", "screener": "xgboost", "allocator": "ga_max_return",
          "labeling": "historical", "weight_cap": 1.0},
         "machine-learning screening added, on the conventional labeling"),
    Rung("A4", "proposed_labeling",
         {"arm_kind": "proposed", "screener": "xgboost", "allocator": "ga_max_return",
          "labeling": "historical_and_t1", "weight_cap": 1.0},
         "the proposed labeling strategy"),
    Rung("A5", "knee_selection",
         {"arm_kind": "proposed", "screener": "xgboost", "allocator": "ga_knee",
          "labeling": "historical_and_t1", "weight_cap": 1.0},
         "knee-point selection instead of an extreme"),
    Rung("A6", "concentration_cap",
         {"arm_kind": "proposed", "screener": "xgboost", "allocator": "ga_knee",
          "labeling": "historical_and_t1", "weight_cap": 0.20},
         "the full proposal, with the concentration cap"),
)

# Components that leave-one-out removes from the full proposal, with the configuration that
# results from removing each.
COMPONENTS: Mapping[str, Mapping[str, object]] = {
    "screening": {"screener": "all_stocks"},
    "proposed_labeling": {"labeling": "historical"},
    "knee_selection": {"allocator": "ga_max_return"},
    "concentration_cap": {"weight_cap": 1.0},
}


@dataclass(frozen=True, slots=True)
class ComponentEffect:
    """One ablation effect, with everything needed to read it responsibly."""

    kind: str  # "ladder" | "leave_one_out"
    component: str
    from_label: str
    to_label: str
    effect: float
    ci_low: float
    ci_high: float
    minimum_detectable: float
    n_blocks: int
    available: bool = True
    unavailable_reason: str = ""

    @property
    def below_resolution(self) -> bool:
        """Whether the design could have distinguished this effect from zero.

        A degenerate comparison -- identical configurations, or a zero-dispersion difference
        series -- yields a zero bound, and calling that "resolved" would report an artefact
        as a finding. It is treated as unresolvable.
        """
        if not self.available or not np.isfinite(self.effect):
            return True
        if not np.isfinite(self.minimum_detectable) or self.minimum_detectable <= 0:
            return True
        return abs(self.effect) < self.minimum_detectable

    @property
    def verdict(self) -> str:
        if not self.available:
            return f"unavailable: {self.unavailable_reason}"
        if self.below_resolution:
            return "below the resolution of this design"
        return "resolved: increase" if self.effect > 0 else "resolved: decrease"


def ladder_is_well_formed(ladder: Sequence[Rung] = LADDER) -> None:
    """Check that every factorial rung pins the same set of factors.

    Raises:
        ValueError: naming the rung that leaves a factor unpinned. Such a rung would compare
            an aggregate against a specific configuration and report the difference as one
            component's contribution.
    """
    factorial = [r for r in ladder if r.selector.get("arm_kind") == "proposed"]
    if not factorial:
        return
    expected = set(factorial[0].selector)
    for rung in factorial[1:]:
        if set(rung.selector) != expected:
            missing = expected.symmetric_difference(set(rung.selector))
            raise ValueError(
                f"rung {rung.name} pins a different factor set than {factorial[0].name}; "
                f"differing keys: {sorted(missing)}. An unpinned factor would be averaged "
                "over while another rung fixes it."
            )


def _series(
    results: pd.DataFrame, selector: Mapping[str, object], metric: str, cost_bps: float
) -> pd.Series | None:
    """Monthly series for one configuration, or ``None`` when it is absent."""
    frame = results[(results["cost_bps"] == cost_bps) & (~results["skip_reason"].astype(bool))]
    for column, value in selector.items():
        if column not in frame.columns:
            return None
        frame = frame[frame[column] == value]
    if frame.empty:
        return None
    # Median across seeds, as elsewhere: seed dispersion is optimiser stability, not method.
    return frame.groupby("month")[metric].median().sort_index()


def effect_between(
    results: pd.DataFrame,
    from_selector: Mapping[str, object],
    to_selector: Mapping[str, object],
    *,
    metric: str,
    cost_bps: float,
    alpha: float = 0.05,
    power: float = 0.80,
    resamples: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float, float, int] | None:
    """Paired effect between two configurations, with a bootstrap interval and the bound.

    Returns ``None`` when either configuration is missing from the results, so the caller can
    report the rung as unavailable rather than closing the ladder over a gap.
    """
    before = _series(results, from_selector, metric, cost_bps)
    after = _series(results, to_selector, metric, cost_bps)
    if before is None or after is None:
        return None
    months = before.index.intersection(after.index)
    if len(months) < 2:
        return None

    differences = (after.loc[months] - before.loc[months]).to_numpy(dtype=float)
    differences = differences[np.isfinite(differences)]
    if differences.size < 2:
        return None

    effect = float(differences.mean())
    rng = np.random.default_rng(seed)
    draws = rng.choice(differences, size=(resamples, differences.size), replace=True)
    means = draws.mean(axis=1)
    low, high = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    mde = minimum_detectable_effect(
        float(differences.std(ddof=1)), differences.size, alpha=alpha, power=power
    )
    return effect, float(low), float(high), mde, int(differences.size)


def ladder_effects(
    results: pd.DataFrame, *, metric: str, cost_bps: float, **kwargs
) -> list[ComponentEffect]:
    """Incremental effect of each rung over its predecessor.

    A missing rung is reported as unavailable and the ladder is **not** closed over it:
    skipping a gap would silently attribute two components' combined effect to one of them.
    """
    out: list[ComponentEffect] = []
    for previous, rung in zip(LADDER, LADDER[1:]):
        computed = effect_between(
            results, previous.selector, rung.selector, metric=metric, cost_bps=cost_bps, **kwargs
        )
        if computed is None:
            out.append(
                ComponentEffect(
                    "ladder", rung.adds, previous.name, rung.name,
                    float("nan"), float("nan"), float("nan"), float("nan"), 0,
                    available=False,
                    unavailable_reason=f"{previous.name} or {rung.name} absent from results",
                )
            )
            continue
        effect, low, high, mde, n = computed
        out.append(
            ComponentEffect("ladder", rung.adds, previous.name, rung.name,
                            effect, low, high, mde, n)
        )
    return out


def leave_one_out_effects(
    results: pd.DataFrame, *, metric: str, cost_bps: float, **kwargs
) -> list[ComponentEffect]:
    """Effect of removing each component from the full proposal.

    Reported as the *loss* from removal, so a positive value means the component helped --
    matching the ladder's sign convention, which otherwise inverts and invites misreading.
    """
    full = dict(LADDER[-1].selector)
    out: list[ComponentEffect] = []
    for component, replacement in COMPONENTS.items():
        reduced = {**full, **replacement}
        computed = effect_between(
            results, reduced, full, metric=metric, cost_bps=cost_bps, **kwargs
        )
        if computed is None:
            out.append(
                ComponentEffect(
                    "leave_one_out", component, f"A6 minus {component}", "A6",
                    float("nan"), float("nan"), float("nan"), float("nan"), 0,
                    available=False, unavailable_reason="configuration absent from results",
                )
            )
            continue
        effect, low, high, mde, n = computed
        out.append(
            ComponentEffect("leave_one_out", component, f"A6 minus {component}", "A6",
                            effect, low, high, mde, n)
        )
    return out


def compare_ablations(
    ladder: Sequence[ComponentEffect], leave_one_out: Sequence[ComponentEffect]
) -> pd.DataFrame:
    """Put the two ablations side by side and flag where they disagree.

    Disagreement is a finding about interaction, not an inconsistency to be reconciled: a
    component can be worth little when added early and a great deal once the rest is present,
    or the reverse.
    """
    by_component = {e.component: e for e in leave_one_out}
    rows = []
    for step in ladder:
        counterpart = by_component.get(step.component)
        both_resolved = (
            counterpart is not None
            and not step.below_resolution
            and not counterpart.below_resolution
        )
        disagrees = bool(
            both_resolved and np.sign(step.effect) != np.sign(counterpart.effect)
        )
        rows.append(
            {
                "component": step.component,
                "ladder_effect": step.effect,
                "ladder_verdict": step.verdict,
                "loo_effect": counterpart.effect if counterpart else float("nan"),
                "loo_verdict": counterpart.verdict if counterpart else "not evaluated",
                "minimum_detectable": step.minimum_detectable,
                "disagrees": disagrees,
            }
        )
    return pd.DataFrame(rows)
