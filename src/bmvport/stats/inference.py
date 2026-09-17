"""Inference over the results table.

Three layers, in the order the manuscript reports them.

**Confirmatory.** The pre-specified factor contrasts of the configuration, Holm-corrected
across that family alone. These are the only claims the paper makes as confirmatory, and the
family was fixed before the grid ran.

**Comparator confrontation.** The best proposed arm against every Tier 0/1/2 arm, blocked on
month. Reported as effect sizes with intervals and the detectable-effect bound, because a
head-to-head comparison on eighteen blocks cannot resolve differences smaller than roughly
24% annualised -- and saying "no significant difference" without that bound would present a
design limit as a finding about the world.

**Exploratory.** Everything else: factor effects at other cost scenarios, allocator
rankings, the ablations. Labelled as such.

Every non-rejection carries the minimum detectable effect and the block count, so a reader
can tell an absent effect from an invisible one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ..config import RunConfig
from .power import assess_non_rejection, minimum_detectable_effect

# Metrics expressed as a monthly return, for which multiplying by twelve is a meaningful
# annualisation. Sharpe ratios and dispersion measures are not in this set.
RETURN_METRICS = frozenset({"return_net", "return_gross"})

__all__ = [
    "TestResult",
    "RETURN_METRICS",
    "monthly_series",
    "paired_test",
    "holm_adjust",
    "confirmatory_tests",
    "comparator_confrontation",
    "factor_effects",
    "mixed_model",
    "hierarchical_friedman",
    "critical_difference",
    "month_block_standard_errors",
]


@dataclass(frozen=True, slots=True)
class TestResult:
    """One blocked comparison, with everything needed to read it responsibly."""

    family: str
    name: str
    effect: float
    ci_low: float
    ci_high: float
    p_value: float
    p_adjusted: float
    minimum_detectable: float
    n_blocks: int
    metric: str
    cost_bps: float
    note: str = ""

    @property
    def rejected(self) -> bool:
        return bool(np.isfinite(self.p_adjusted) and self.p_adjusted < 0.05)

    @property
    def below_resolution(self) -> bool:
        if not np.isfinite(self.effect) or self.minimum_detectable <= 0:
            return True
        return abs(self.effect) < self.minimum_detectable

    @property
    def verdict(self) -> str:
        if self.rejected:
            return "rejected" if self.effect < 0 else "rejected: higher"
        if self.below_resolution:
            return "not rejected; effect below the design's resolution"
        return "not rejected despite a resolvable effect size"

    def to_record(self) -> dict:
        """Serialise the result including the derived verdict fields.

        ``dataclasses.asdict`` drops properties, so a plain asdict writes a record without
        ``rejected``, ``below_resolution`` or ``verdict`` -- the three fields that say what
        the test concluded. The comparator table reads ``verdict``, so the omission surfaces
        as a missing column rather than as an error, which is how it survived one round of
        regeneration.
        """
        import dataclasses

        record = dataclasses.asdict(self)
        record["rejected"] = self.rejected
        record["below_resolution"] = self.below_resolution
        record["verdict"] = self.verdict
        return record


def monthly_series(
    results: pd.DataFrame, selector: Mapping[str, object], metric: str, cost_bps: float
) -> pd.Series | None:
    """Median-across-seeds monthly series for one configuration.

    Seeds are collapsed before comparison: across-seed dispersion is a statement about
    optimiser stability, not about which method is better, and mixing the two would let
    replicate noise enter a method ranking.
    """
    frame = results[(results["cost_bps"] == cost_bps) & (~results["skip_reason"].astype(bool))]
    for column, value in selector.items():
        if column not in frame.columns:
            return None
        frame = frame[frame[column] == value]
    if frame.empty:
        return None
    series = frame.groupby("month")[metric].median().sort_index()
    return series if len(series) else None


def paired_test(
    treatment: pd.Series, control: pd.Series, *, alpha: float = 0.05, resamples: int = 5000,
    seed: int = 0,
) -> tuple[float, float, float, float, float, int] | None:
    """Wilcoxon signed-rank on the paired monthly differences, with a bootstrap interval.

    Non-parametric because eighteen monthly returns are not reliably normal and a t-test
    would borrow precision it does not have.
    """
    months = treatment.index.intersection(control.index)
    if len(months) < 3:
        return None
    differences = (treatment.loc[months] - control.loc[months]).to_numpy(dtype=float)
    differences = differences[np.isfinite(differences)]
    if differences.size < 3 or np.allclose(differences, 0):
        return None

    from scipy.stats import wilcoxon

    try:
        p_value = float(wilcoxon(differences).pvalue)
    except ValueError:
        return None

    rng = np.random.default_rng(seed)
    means = rng.choice(differences, size=(resamples, differences.size), replace=True).mean(axis=1)
    low, high = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    mde = minimum_detectable_effect(float(differences.std(ddof=1)), differences.size)
    return float(differences.mean()), float(low), float(high), p_value, float(mde), int(differences.size)


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjustment, applied within a declared family.

    Correcting within a family only controls error inside that family; the manuscript
    therefore names the families rather than leaving the reader to infer them.
    """
    values = list(p_values)
    order = np.argsort(values)
    adjusted = [float("nan")] * len(values)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (len(values) - rank) * values[index]
        running = max(running, candidate)
        adjusted[index] = float(min(running, 1.0))
    return adjusted


def _finalise(family: str, raw: list[tuple[str, tuple, str, str]],
              cost_bps: float) -> list[TestResult]:
    """Apply the family correction and assemble results.

    Each test carries its own metric: the family mixes return and Sharpe contrasts, and
    labelling them all as returns would invite annualising a Sharpe ratio by twelve, which
    is meaningless -- Sharpe scales with the square root of the horizon, not linearly.
    """
    if not raw:
        return []
    adjusted = holm_adjust([r[1][3] for r in raw])
    out = []
    for (name, stats, note, metric), p_adj in zip(raw, adjusted):
        effect, low, high, p, mde, n = stats
        out.append(
            TestResult(family=family, name=name, effect=effect, ci_low=low, ci_high=high,
                       p_value=p, p_adjusted=p_adj, minimum_detectable=mde, n_blocks=n,
                       metric=metric, cost_bps=cost_bps, note=note)
        )
    return out


def confirmatory_tests(results: pd.DataFrame, config: RunConfig) -> list[TestResult]:
    """The pre-specified factor contrasts, Holm-corrected across that family alone."""
    cost = config.evaluation.primary_cost_scenario_bps
    proposed = results[results["arm_kind"] == "proposed"]
    raw: list[tuple[str, tuple, str, str]] = []

    for hypothesis in config.stats.primary_hypotheses:
        metric = hypothesis.metric
        if hypothesis.factor == "screener":
            ml = [s for s in config.screening.screeners if s != "all_stocks"]
            treat = monthly_series(proposed, {}, metric, cost) if False else None
            frame = proposed[
                (proposed["cost_bps"] == cost) & (~proposed["skip_reason"].astype(bool))
            ]
            treat = frame[frame["screener"].isin(ml)].groupby("month")[metric].median()
            ctrl = frame[frame["screener"] == "all_stocks"].groupby("month")[metric].median()
        elif hypothesis.factor == "labeling":
            frame = proposed[
                (proposed["cost_bps"] == cost) & (~proposed["skip_reason"].astype(bool))
            ]
            treat = frame[frame["labeling"] == "historical_and_t1"].groupby("month")[metric].median()
            ctrl = frame[frame["labeling"] != "historical_and_t1"].groupby("month")[metric].median()
        else:  # weight_cap
            frame = proposed[
                (proposed["cost_bps"] == cost) & (~proposed["skip_reason"].astype(bool))
            ]
            # The contrast must be PAIRED at the cell level: the same (labeling, screener,
            # allocator) with and without the cap. Comparing medians of two different cell
            # populations is not an effect of the cap, it is a difference between whatever
            # cells happened to fall on each side.
            #
            # It is also conditioned on cells where the cap actually bound. Averaging over
            # cells where nothing was constrained would dilute the effect towards zero and
            # describe a design artefact as a finding (see D24).
            cell = ["labeling", "screener", "allocator", "month"]
            capped = frame[(frame["weight_cap"] < 1.0) & (frame["cap_binding"] == True)]  # noqa: E712
            if capped.empty:
                continue
            uncapped = frame[frame["weight_cap"] == 1.0]
            merged = capped.groupby(cell)[metric].median().to_frame("capped").join(
                uncapped.groupby(cell)[metric].median().to_frame("uncapped"), how="inner"
            ).dropna()
            if merged.empty:
                continue
            treat = merged.groupby("month")["capped"].median()
            ctrl = merged.groupby("month")["uncapped"].median()

        stats = paired_test(treat, ctrl, seed=config.stats.bootstrap_seed)
        if stats is None:
            continue
        raw.append((hypothesis.name, stats, hypothesis.contrast, metric))
    return _finalise("confirmatory", raw, cost)


def comparator_confrontation(results: pd.DataFrame, config: RunConfig) -> list[TestResult]:
    """Best proposed arm against every comparator, blocked on month."""
    cost = config.evaluation.primary_cost_scenario_bps
    evaluated = results[(results["cost_bps"] == cost) & (~results["skip_reason"].astype(bool))]
    proposed = evaluated[evaluated["arm_kind"] == "proposed"]
    if proposed.empty:
        return []

    # The pre-specified rule: highest mean net return at the primary cost scenario.
    ranked = proposed.groupby(["labeling", "screener", "allocator", "weight_cap"])[
        "return_net"
    ].mean().sort_values(ascending=False)
    best_key = ranked.index[0]
    best = proposed[
        (proposed["labeling"] == best_key[0]) & (proposed["screener"] == best_key[1])
        & (proposed["allocator"] == best_key[2]) & (proposed["weight_cap"] == best_key[3])
    ].groupby("month")["return_net"].median()

    raw: list[tuple[str, tuple, str, str]] = []
    comparators = evaluated[evaluated["arm_kind"] == "comparator"]
    for name, group in comparators.groupby("allocator"):
        control = group.groupby("month")["return_net"].median()
        stats = paired_test(best, control, seed=config.stats.bootstrap_seed)
        if stats is None:
            continue
        tier = str(group["comparator_tier"].iloc[0])
        raw.append((name, stats, f"{tier}; best arm {'/'.join(map(str, best_key))}", "return_net"))
    return _finalise("comparator_confrontation", raw, cost)


def factor_effects(
    results: pd.DataFrame, config: RunConfig, *, metric: str = "return_net"
) -> pd.DataFrame:
    """Exploratory factor summaries at every cost scenario."""
    evaluated = results[
        (results["arm_kind"] == "proposed") & (~results["skip_reason"].astype(bool))
    ]
    rows = []
    for factor in ("screener", "labeling", "allocator", "weight_cap"):
        for cost, cost_group in evaluated.groupby("cost_bps"):
            for level, group in cost_group.groupby(factor):
                series = group.groupby("month")[metric].median()
                rows.append(
                    {
                        "family": "exploratory", "factor": factor, "level": level,
                        "cost_bps": cost, "metric": metric,
                        "mean_monthly": float(series.mean()),
                        "annualised": float(series.mean() * 12),
                        "median_sharpe": float(group["sharpe"].median()),
                        "n_blocks": int(series.size),
                    }
                )
    return pd.DataFrame(rows)


def mixed_model(
    results: pd.DataFrame, config: RunConfig, *, metric: str = "return_net",
    cost_bps: float | None = None,
) -> dict:
    """Linear mixed model on monthly net returns, month as a random effect.

    Designed in D14 as the primary inference because a flat comparison across cells cannot
    reject anything at this scale. What the executed grid showed is that the factor
    contrasts cannot either -- the detectable bound is 13 to 17% annualised and every effect
    falls below it. The model is still fitted and reported, for two reasons: a referee in
    this literature expects it, and coefficient estimates with intervals are informative
    even where the corresponding test does not reject.

    The labeling-by-allocator interaction is included because it is the direct test of the
    concentration-versus-signal hypothesis: an apparent labeling advantage that varies with
    how concentrated the allocator is would be a property of concentration, not of the label.
    """
    import statsmodels.formula.api as smf

    cost = config.evaluation.primary_cost_scenario_bps if cost_bps is None else cost_bps
    frame = results[
        (results["arm_kind"] == "proposed")
        & (results["cost_bps"] == cost)
        & (~results["skip_reason"].astype(bool))
    ][["month", "labeling", "screener", "allocator", "weight_cap", metric]].dropna()
    if frame.empty:
        return {"fitted": False, "reason": "no rows for this metric and cost scenario"}

    frame = frame.rename(columns={metric: "y"})
    frame["weight_cap"] = frame["weight_cap"].astype(str)
    formula = "y ~ C(screener) + C(labeling) * C(allocator) + C(weight_cap)"
    try:
        fit = smf.mixedlm(formula, frame, groups=frame["month"]).fit(reml=True)
    except Exception as exc:  # noqa: BLE001
        return {"fitted": False, "reason": f"{type(exc).__name__}: {exc}"[:200]}

    terms = []
    for name, estimate in fit.params.items():
        if name in ("Group Var", "Intercept"):
            continue
        terms.append(
            {
                "term": name,
                "estimate_monthly": float(estimate),
                "estimate_annualised": float(estimate) * 12 if metric in RETURN_METRICS else float("nan"),
                "std_error_monthly": float(fit.bse.get(name, float("nan"))),
                "p_value": float(fit.pvalues.get(name, float("nan"))),
                "is_interaction": ":" in name,
            }
        )
    dependence = month_block_standard_errors(frame, config, metric="y")
    return {
        "fitted": True, "formula": formula, "metric": metric, "cost_bps": cost,
        "n_observations": int(len(frame)), "n_months": int(frame["month"].nunique()),
        "group_variance": float(fit.cov_re.iloc[0, 0]) if fit.cov_re.size else float("nan"),
        "terms": terms,
        "interaction_terms": [t for t in terms if t["is_interaction"]],
        "dependence_check": dependence,
        "inference_basis": (
            "The month-paired confirmatory tests govern. The p-values above are reported for "
            "completeness and are anti-conservative by the factor in dependence_check: a "
            "single random intercept absorbs a month's common shift but not the "
            "cross-sectional dependence among cells that hold overlapping stocks."
        ),
    }


def month_block_standard_errors(
    frame: pd.DataFrame, config: RunConfig, *, metric: str = "return_net",
    reference: str = "all_stocks", n_boot: int | None = None,
) -> dict:
    """Re-estimate each screener contrast with the month as the unit of analysis.

    The mixed model of D14 treats every cell-month as an observation. Cells within a month
    hold overlapping stocks and move together, and a random intercept removes only the
    month's common level. This resamples whole months instead, which is the dependence
    structure the design actually has, and reports the ratio between the two standard errors.

    The ratio is not a diagnostic to glance at: it is the reason the pre-registration named
    the month-paired test confirmatory and this model descriptive. Where the ratio is large,
    the model's nominal p-values must not be read as evidence.
    """
    rng = np.random.default_rng(config.stats.bootstrap_seed)
    n_boot = config.stats.bootstrap_resamples if n_boot is None else n_boot
    pivot = frame.pivot_table(index="month", columns="screener", values=metric, aggfunc="mean")
    if reference not in pivot.columns:
        return {"available": False, "reason": f"reference level {reference!r} absent"}
    contrasts = []
    for level in sorted(c for c in pivot.columns if c != reference):
        paired = pivot[[reference, level]].dropna()
        if paired.shape[0] < 3:
            continue
        differences = (paired[level] - paired[reference]).to_numpy()
        draws = rng.choice(differences, size=(n_boot, differences.size), replace=True).mean(axis=1)
        p_value = 2 * min(float((draws <= 0).mean()), float((draws >= 0).mean()))
        contrasts.append(
            {
                "level": level, "reference": reference, "n_blocks": int(differences.size),
                "effect_monthly": float(differences.mean()),
                "effect_annualised": float(differences.mean()) * 12
                if metric in RETURN_METRICS or metric == "y" else float("nan"),
                "std_error_monthly": float(draws.std(ddof=1)),
                "p_value": min(1.0, p_value),
                "ci_low_monthly": float(np.quantile(draws, 0.025)),
                "ci_high_monthly": float(np.quantile(draws, 0.975)),
            }
        )
    return {"available": True, "n_boot": n_boot, "unit_of_analysis": "evaluation month",
            "contrasts": contrasts}


def hierarchical_friedman(
    results: pd.DataFrame, config: RunConfig, *, metric: str = "return_net"
) -> pd.DataFrame:
    """Friedman test per factor, holding the other factors at their best level.

    Blocked on evaluation month. The held-constant levels are recorded with each result: a
    Friedman statistic without them describes a comparison the reader cannot reconstruct.
    """
    from scipy.stats import friedmanchisquare

    cost = config.evaluation.primary_cost_scenario_bps
    frame = results[
        (results["arm_kind"] == "proposed")
        & (results["cost_bps"] == cost)
        & (~results["skip_reason"].astype(bool))
    ]
    factors = ("screener", "labeling", "allocator", "weight_cap")
    best = {
        f: frame.groupby(f)[metric].mean().idxmax() for f in factors
    }
    rows = []
    for factor in factors:
        held = {f: v for f, v in best.items() if f != factor}
        subset = frame
        for f, v in held.items():
            subset = subset[subset[f] == v]
        pivot = subset.pivot_table(index="month", columns=factor, values=metric, aggfunc="median")
        pivot = pivot.dropna()
        if pivot.shape[1] < 3 or pivot.shape[0] < 3:
            rows.append({"factor": factor, "levels": int(pivot.shape[1]),
                         "n_blocks": int(pivot.shape[0]), "statistic": float("nan"),
                         "p_value": float("nan"),
                         "held_constant": "; ".join(f"{k}={v}" for k, v in held.items()),
                         "note": "too few levels or blocks for a Friedman test"})
            continue
        statistic, p_value = friedmanchisquare(*[pivot[c].to_numpy() for c in pivot.columns])
        rows.append(
            {
                "factor": factor, "levels": int(pivot.shape[1]),
                "n_blocks": int(pivot.shape[0]), "statistic": float(statistic),
                "p_value": float(p_value),
                "held_constant": "; ".join(f"{k}={v}" for k, v in held.items()),
                "note": "",
            }
        )
    return pd.DataFrame(rows)


def critical_difference(n_levels: int, n_blocks: int, alpha: float = 0.05) -> float:
    """Nemenyi critical difference for a headline set of ``n_levels`` over ``n_blocks``.

    Reported with every critical-difference diagram, and it is the number that shows why the
    flat comparison of D14 was abandoned: the width grows as sqrt(k(k+1)/6N), so at
    216 cells over 18 blocks nothing could ever separate.
    """
    from scipy.stats import studentized_range

    if n_levels < 2 or n_blocks < 2:
        return float("nan")
    q = float(studentized_range.ppf(1 - alpha, n_levels, np.inf)) / np.sqrt(2.0)
    return float(q * np.sqrt(n_levels * (n_levels + 1) / (6.0 * n_blocks)))
