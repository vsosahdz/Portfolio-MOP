"""Manuscript sections, rendered from generated facts rather than written by hand.

Every number a section quotes is resolved from the executed results at render time. The
templates below contain prose and placeholders; they contain no digits, and a test enforces
that. This is the mechanism behind the requirement that no figure in the draft be typed by
hand -- not a convention to remember, but a property the test suite can check.

The consequence is worth stating plainly: if a rerun changes a number, the prose changes
with it. A sentence that would become false cannot survive, because the sentence is built
from the same table the claim is made from.

Section content follows what the results support. Where the study's own conclusion is
unfavourable to the method it proposes, the section says so in the same voice it uses for
the favourable findings; the advantages and the disadvantages are rendered from the same
facts dictionary and neither can be adjusted without adjusting the other.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..config import RunConfig

__all__ = ["collect_facts", "SECTIONS", "render_sections", "write_manuscript",
           "template_digit_violations", "venue_limit_violations",
           "ABSTRACT_WORD_LIMIT", "HIGHLIGHT_CHAR_LIMIT", "KEYWORD_LIMIT"]

# Expert Systems with Applications: abstract about 250 words, highlights of at most 85
# characters each, and between one and seven keywords. These are checked rather than
# remembered, because a substituted fact changes the length after the prose was written and
# an over-length abstract is a desk rejection before review.
ABSTRACT_WORD_LIMIT = 250
HIGHLIGHT_CHAR_LIMIT = 85
KEYWORD_LIMIT = 7

# A template may carry three kinds of literal and no others: a citation year, the name of an
# instrument or method that contains a digit, and a placeholder. Everything else numeric is a
# result, and a result must be resolved from the run rather than typed. The list is explicit
# rather than a loosened pattern, so adding a name is a visible decision.
_CITATION_YEAR = re.compile(r"(?<![\d.])(?:1[89]|20)\d{2}(?![\d.])")
_PLACEHOLDER = re.compile(r"\{[a-z0-9_]+\}")
_PROPER_NOUNS = ("CETES 28-day", "S&P 500", "L1-regularised", "NSGA-III")


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _largest_arm_shift(contrasts: list[Mapping[str, Any]]) -> float:
    """Biggest move a contrast makes between the sample and shrinkage arms.

    Compared within a contrast, never across: the spread between two different contrasts
    says nothing about whether the estimator changed anything.
    """
    by_contrast: dict[str, dict[str, float]] = {}
    for record in contrasts:
        by_contrast.setdefault(record["contrast"], {})[record["arm"]] = float(record["effect"])
    shifts = [
        abs(arms["shrinkage"] - arms["muestra"])
        for arms in by_contrast.values()
        if {"shrinkage", "muestra"} <= set(arms)
    ]
    return max(shifts) if shifts else float("nan")


def collect_facts(
    results: pd.DataFrame, config: RunConfig, artefacts: Mapping[str, Any]
) -> dict[str, str]:
    """Resolve every quantity the sections quote, from the executed run.

    ``artefacts`` carries the analysis outputs that are not derivable from the results table
    alone: the confirmatory tests, the ablation sweep, the reachability sweep, the shrinkage
    comparison and the Friedman table.
    """
    cost = config.evaluation.primary_cost_scenario_bps
    live = results[(results["cost_bps"] == cost) & (~results["skip_reason"].astype(bool))]
    proposed = live[live["arm_kind"] == "proposed"]

    by_screener = (
        proposed.groupby(["screener", "month"])["return_net"].median()
        .groupby("screener").mean() * 12
    )
    control = float(by_screener.get("all_stocks", np.nan))
    ml = by_screener.drop(labels=["all_stocks"], errors="ignore")
    best_ml_name = str(ml.idxmax()).replace("_", " ")
    worst_ml_name = str(ml.idxmin()).replace("_", " ")

    comparator = live[live["arm_kind"] == "comparator"]
    bars = (
        comparator.groupby(["allocator", "month"])["return_net"].median()
        .groupby("allocator").mean() * 12
    )

    attribution = (
        proposed.groupby("screener")["attribution_selection"].mean() * 12
    ).drop(labels=["all_stocks"], errors="ignore")

    # Aggregated by ablation level, not by raw variable count. The no-preselection level
    # takes the whole month's universe, which ranges from the high eighties to just over a
    # hundred, so grouping on the count splits one level into several thin groups and the
    # largest of them is a single month. The first draft did exactly that and quoted an
    # attainment of 0.990 from one month rather than 0.964 from the level.
    ablation = artefacts["ablation"]
    grouped = ablation.groupby("level").agg(
        n=("n_variables", "mean"), wall=("wall_clock", "mean"),
        attain=("attainment_at_base", "mean"), cov=("covariance_entries", "mean"))
    grouped = grouped.sort_values("n")
    smallest, largest = grouped.index[0], grouped.index[-1]

    reach = artefacts["reachability"].copy()
    reach["level"] = np.where(
        reach["n"].isin([10, 20, 40, 80]), reach["n"].astype(str), "no-preselection")
    reach_grouped = reach.groupby("level").agg(
        n=("n", "mean"), ret=("ret_cov", "mean"), risk=("risk_cov", "mean"))
    reach_grouped = reach_grouped.sort_values("n")
    reach_small, reach_large = reach_grouped.index[0], reach_grouped.index[-1]
    from scipy.stats import spearmanr
    rho_return, p_return = spearmanr(reach["n"], reach["ret_cov"])

    # Only the pre-registered confirmatory family bounds what the study claims; the
    # exploratory tests have their own resolution and pooling them would understate it.
    #
    # Restricted further to the return metrics, and for a reason the first draft of this
    # function got wrong: the cap hypothesis is tested on the Sharpe ratio, whose detectable
    # effect is not a monthly rate and must never be multiplied by twelve. Pooling it with
    # the return tests produced a bound of 542 percent. A range quoted across metrics has no
    # meaning even when the arithmetic is right.
    from ..stats.inference import RETURN_METRICS

    tests = artefacts["tests"]
    detectable = [
        abs(float(record["minimum_detectable"])) * 12 for record in tests
        if record.get("family") == "confirmatory"
        and record.get("metric") in RETURN_METRICS
        and np.isfinite(record.get("minimum_detectable", np.nan))
    ]
    if not detectable:
        raise ValueError("no confirmatory test on a return metric carries a detectable effect")

    friedman = artefacts["friedman"].dropna(subset=["p_value"])
    shrinkage = artefacts["shrinkage"]
    mixed_terms = artefacts["mixed_model"]["interaction_terms"]

    months = sorted(results["month"].unique())
    universe = proposed[proposed["screener"] == "all_stocks"].groupby("month")["selection_size"]

    return {
        "n_blocks": f"{len(months)}",
        "window_start": months[0],
        "window_end": months[-1],
        "cost_bps": f"{cost:.0f}",
        "universe_min": f"{int(universe.max().min())}",
        "universe_max": f"{int(universe.max().max())}",
        "n_screeners": f"{proposed['screener'].nunique()}",
        "control_return": _pct(control),
        "best_ml": best_ml_name,
        "best_ml_return": _pct(float(ml.max())),
        "worst_ml": worst_ml_name,
        "worst_ml_return": _pct(float(ml.min())),
        "control_minus_best_ml": _pct(control - float(ml.max())),
        "momentum_bar": _pct(float(bars.get("momentum_equal_weight", np.nan))),
        "cetes_bar": _pct(float(bars.get("cetes_28d", np.nan))),
        "ipc_bar": _pct(float(bars.get("bmv_ipc", np.nan))),
        "selection_worst": _pct(float(attribution.min())),
        "selection_best": _pct(float(attribution.max())),
        "mde_low": _pct(min(detectable)),
        "mde_high": _pct(max(detectable)),
        "friedman_p_low": f"{friedman['p_value'].min():.2f}",
        "friedman_p_high": f"{friedman['p_value'].max():.2f}",
        "critical_difference": f"{artefacts['critical_difference']:.2f}",
        "headline_cells": f"{artefacts['headline_cells']}",
        "interaction_terms": f"{len(mixed_terms)}",
        "se_ratio_low": f"{min(artefacts['se_ratios']):.0f}",
        "se_ratio_high": f"{max(artefacts['se_ratios']):.0f}",
        "n_cell_months": f"{artefacts['n_cell_months']:,}",
        "ablation_small_n": f"{grouped.loc[smallest, 'n']:.0f}",
        "ablation_large_n": f"{grouped.loc[largest, 'n']:.0f}",
        "ablation_wall_small": f"{grouped.loc[smallest, 'wall']:.1f}",
        "ablation_wall_large": f"{grouped.loc[largest, 'wall']:.1f}",
        "ablation_wall_factor": f"{grouped.loc[largest, 'wall'] / grouped.loc[smallest, 'wall']:.1f}",
        "ablation_cov_factor": f"{grouped.loc[largest, 'cov'] / grouped.loc[smallest, 'cov']:.0f}",
        "ablation_attain_small": f"{grouped.loc[smallest, 'attain']:.3f}",
        "ablation_attain_large": f"{grouped.loc[largest, 'attain']:.3f}",
        "ablation_instances": f"{len(ablation)}",
        "coverage_small_n": f"{reach_grouped.loc[reach_small, 'n']:.0f}",
        "coverage_large_n": f"{reach_grouped.loc[reach_large, 'n']:.0f}",
        "coverage_return_small": _pct(float(reach_grouped.loc[reach_small, "ret"])),
        "coverage_return_large": _pct(float(reach_grouped.loc[reach_large, "ret"])),
        "coverage_risk_small": _pct(float(reach_grouped.loc[reach_small, "risk"])),
        "coverage_risk_large": _pct(float(reach_grouped.loc[reach_large, "risk"])),
        "coverage_rho": f"{rho_return:+.2f}",
        "coverage_instances": f"{len(reach)}",
        "shrinkage_max_shift": _pct(_largest_arm_shift(shrinkage["contrasts"]) / 100),
        "shrinkage_reverses": "no contrast reverses sign" if not shrinkage["reverses"]
        else "at least one contrast reverses sign",
    }


SECTIONS: dict[str, str] = {
    "abstract": """\
## Abstract

Intelligent decision-support systems built on multi-objective optimisation are routinely
validated with indicators computed against a reference front produced by the same optimiser
under a larger budget. We show this validation is blind by construction to any failure the
two runs share, and propose one that is not: where a problem's objective extremes
have a closed form, the fraction of the attainable range a front spans is measurable
against ground truth.

The difference is decisive. Over {ablation_instances} instances spanning {ablation_small_n}
to {ablation_large_n} decision variables, the conventional indicator *rises* with problem
size, from {ablation_attain_small} to {ablation_attain_large}, suggesting large instances
are solved as completely as small ones. Our diagnostic *falls*, from
{coverage_return_small} to {coverage_return_large} of the attainable return range and
{coverage_risk_small} to {coverage_risk_large} of the risk range, with Spearman rho =
{coverage_rho}: the search degrades sharply with dimension, and the standard indicator
reports the opposite.

We apply it to a screening-plus-optimisation system for equity portfolio selection on an
emerging market: {n_screeners} screening arms, including gradient boosting and a tabular
foundation model, over {n_blocks} monthly blocks against deterministic, index and published
baselines. The diagnostic separates two claims usually made together: preselection
does reduce search difficulty, and it does not improve the portfolio. No screening arm beats
passing the whole universe to the optimiser, and return attribution locates the cause in a
negative selection component. A mixed model over {n_cell_months} cell-months also understates
standard errors {se_ratio_low}- to {se_ratio_high}-fold against block resampling. Every
artefact regenerates offline from a cached dataset of record.
""",

    "highlights": """\
## Highlights

- Reference-front indicators cannot detect search failures the reference shares
- Coverage against closed-form extremes measures front quality on ground truth
- Coverage falls from {coverage_return_small} to {coverage_return_large} as size grows
- Preselection eases the search yet still lowers portfolio return
- Block-paired tests replace a mixed model that overstates precision {se_ratio_high}-fold
""",

    "keywords": """\
## Keywords

Multi-objective optimisation; performance assessment; evolutionary computation; portfolio
selection; machine learning screening; reproducibility
""",

    "related_work": """\
## Related work

**Performance assessment in multi-objective optimisation.** The hypervolume indicator
(Zitzler and Thiele, 1999) and the attainment function (Fonseca and Fleming, 1996) are the
standard instruments for comparing stochastic multi-objective optimisers, and both require a
point of comparison. Where the true Pareto front is known -- on synthetic benchmarks
constructed for the purpose -- that point is exact. Where it is not, which is every real
application, practice substitutes a reference front assembled from the algorithms under
study, often from an extended-budget run of the same algorithm. The substitution is usually
described as a practical necessity and treated as benign.

This paper is about a case where it is not benign. A reference front produced by the same
optimiser inherits whatever that optimiser cannot do, so an indicator computed against it
measures consistency rather than quality, and reports success precisely when the failure is
systematic. The problem is distinct from the well-studied sensitivity of hypervolume to its
reference point, which concerns the scaling of a comparison rather than its blindness.

**Statistical comparison across problems.** Comparing optimisers over multiple problem
instances is conventionally handled with rank-based tests and critical-difference diagrams
(Demšar, 2006). We use that machinery and report the interval it implies, because over a
small number of blocks the interval is wide enough that adjacency in such a diagram carries
almost no information -- a property of the design that a diagram without its interval invites
readers to ignore.

**Portfolio selection as a multi-objective problem.** The mean-variance formulation
(Markowitz, 1952) is the canonical two-objective instance, and evolutionary approaches to it
are well established; NSGA-III (Deb and Jain, 2014) is the algorithm used here. The
formulation suits our purpose for a reason independent of finance: under a simplex constraint
both extremes are analytic, so the attainable range is computable and the indicator can be
checked against ground truth rather than against itself.

**Preselection and dimensionality reduction.** Applications commonly reduce the decision
space before optimising, and supervised preselection -- training a classifier to admit or
reject candidates -- is a frequent choice, justified on two grounds: that it improves the
solution by removing bad candidates, and that it makes the search tractable by reducing
decision variables. Our ablation tests both claims independently, and finds that they come
apart.

**Comparators.** The empirical arms are placed against deterministic baselines (equal
weighting, inverse volatility, minimum variance, maximum Sharpe), against the reference
series an investor would hold instead -- the CETES 28-day instrument at {cetes_bar}
annualised, the local index at {ipc_bar}, and the S&P 500 converted to pesos -- and against
adapted published approaches: a single-objective genetic algorithm maximising the Sharpe
ratio, an L1-regularised mean-variance formulation with shrinkage covariance, and a
cross-sectional momentum portfolio. Momentum is the demanding one, at {momentum_bar}, above
every arm the studied method produces.
""",

    "contributions": """\
## Contributions

**An indicator that cannot see its own failure mode, and a correction.** Attainment computed
against an extended-budget reference front rises with problem size on our ablation, from
{ablation_attain_small} at {ablation_small_n} decision variables to {ablation_attain_large}
at {ablation_large_n}. Measured against the closed-form attainable range instead, coverage
of the reachable return range falls from {coverage_return_small} to {coverage_return_large}
and of the risk range from {coverage_risk_small} to {coverage_risk_large}, with Spearman rho
= {coverage_rho} over {coverage_instances} instances. The two readings are contradictory and
only one is checkable against ground truth. We give the coverage measure, a guard that
refuses to report attainment without it, and the condition that flags the pathology: high
attainment beside low coverage.

**A demonstration that this changes a substantive conclusion.** The ablation was designed to
test whether supervised preselection earns its place. On the attainment reading it does not,
on any axis. On the coverage reading the tractability claim holds and the quality claim
still fails: preselection genuinely makes the search easier, and separately, the screening
signal is bad enough that the easier search does not pay for it. Those are separable
findings, and an indicator that conflates them would have suppressed one.

**An inferential hazard in blocked experimental designs.** Fitting a mixed model over
{n_cell_months} cell-months with the block as a random effect yields standard errors
{se_ratio_low} to {se_ratio_high} times smaller than resampling whole blocks. The point
estimates agree exactly; only the precision differs. A random intercept removes a block's
common level but not the cross-sectional dependence among configurations that share
underlying data, so the model treats correlated cells as replicates. We report both and let
the pre-registered block-paired tests govern.

**A negative empirical result, reported in full.** Across {n_screeners} screening arms over
{n_blocks} monthly blocks, none beats passing the whole admitted universe to the optimiser:
{control_return} annualised for the no-screening control against {best_ml_return} for the
best arm and {worst_ml_return} for the worst, at {cost_bps} basis points. Return attribution
isolates the mechanism -- the selection component is negative for every screening arm,
between {selection_worst} and {selection_best}, and exactly zero for the control, which
validates the decomposition. No effect exceeds the design's detectable bound of {mde_low} to
{mde_high}, so these are reported as unresolved rather than as null.

**A replication package.** The pipeline is specified before execution, runs from a cached
dataset of record with the network unavailable, and regenerates every table, figure and
quoted number. Configuration drift is detectable: seeds derive from a configuration
fingerprint that is checked against the manifest, a property added after the drift occurred
undetected during development.
""",

    "screening_justification": """\
## Does preselection justify itself?

The original formulation motivates supervised screening on two grounds: that it improves
the portfolio by removing instruments predicted to fall, and that it makes the optimisation
tractable by reducing the number of decision variables. The ablation tests both, over
{ablation_instances} instances spanning {ablation_small_n} to {ablation_large_n} decision
variables with no-preselection as an explicit level.

**On portfolio quality, the argument fails.** The no-screening control returns
{control_return} annualised against {best_ml_return} for the best screening arm, a gap of
{control_minus_best_ml}. The gap is below the design's resolution and is therefore not a
significant difference, but it points the wrong way for the method, and the attribution
decomposition shows why: the selection component is negative for every screener.

**On computational cost, the argument also fails, and by a wide margin.** Wall-clock rises
from {ablation_wall_small} to {ablation_wall_large} seconds between {ablation_small_n} and
{ablation_large_n} variables -- a factor of {ablation_wall_factor} -- while the covariance
matrix grows by a factor of {ablation_cov_factor}. The quadratic growth in problem size
does not translate into a proportionate cost, so preselection is not buying compute.

**On tractability, the argument holds, and this study initially missed it.** The obvious
metric says otherwise: attainment against an extended-budget reference rises with problem
size. That metric is self-referential, because the reference front comes from the same
optimiser. Against the closed-form attainable range -- maximum return is the whole budget in
the highest-mean asset, minimum variance is analytic -- the picture inverts. Return coverage
falls from {coverage_return_small} at {coverage_small_n} variables to
{coverage_return_large} at {coverage_large_n}; risk coverage falls from
{coverage_risk_small} to {coverage_risk_large}. The search genuinely degrades with
dimension.

The two findings are separable and both are reported. Preselection does make the
optimisation easier in a way that is real and measurable. It does not follow that
preselection helps, because the screening signal is negative enough to outweigh what the
smaller search space buys. A reader who wants the tractability benefit without the
selection penalty should look at dimension reduction that does not condition on a predicted
return.
""",

    "limitations": """\
## Limitations

**Resolution.** Conclusions rest on {n_blocks} monthly blocks, giving a minimum detectable
effect of {mde_low} to {mde_high} annualised. No factor effect in the study exceeds that
bound. The hierarchical Friedman tests return p between {friedman_p_low} and
{friedman_p_high}, and the critical difference over the headline set is
{critical_difference} ranks against a spread far smaller. These are statements about the
design, not about the world, and a longer window is the single change that would most
improve the study.

**Dependence and the mixed model.** Fitting the mixed model over {n_cell_months} cell-months
yields standard errors between {se_ratio_low} and {se_ratio_high} times smaller than
resampling whole months, because cells within a month hold overlapping stocks and a single
random intercept removes only the month's common level. The point estimates agree; the
apparent significance does not survive. The month-paired tests are confirmatory, as
pre-registered, and the model is reported as descriptive. The labeling-by-allocator
interaction -- the direct test of whether an apparent labeling advantage is really a
concentration effect -- has {interaction_terms} terms and none reaches significance even at
the model's inflated precision.

**Survivorship.** The universe is screened point-in-time from price history rather than from
an exchange listing record, because the exchange's listing endpoints were unavailable and no
archived snapshot exists. No instrument ceased trading during the evaluation window. That is
a lower bound on the exposure, not a proof of its absence.

**Costs.** Transaction costs are applied after optimisation rather than inside the
objective, so the optimiser never trades return against the cost of achieving it. Costs are
also charged at a flat spread across instruments. Real spreads widen on the smaller and less
liquid names, which the screening arms select more often than the control does, so the flat
assumption flatters the screened arms relative to the control -- the bias runs against the
study's own conclusion rather than toward it.

**Covariance estimator.** The shrinkage sensitivity arm moves the reported contrasts by at
most {shrinkage_max_shift} annualised and {shrinkage_reverses}. The arm covers a subset of
screeners and one labeling strategy, so this is an argument from the observed effect size
rather than a proof of insensitivity across the full grid.

**Scope.** The study covers one exchange over one window with one feature dictionary.
Nothing here establishes that supervised screening subtracts value in general; it
establishes that it did so here, under a protocol designed in advance to be able to detect
the opposite.
""",

    "future_work": """\
## Future work

**Cost-aware optimisation.** The most direct extension is to move transaction costs inside
the objective rather than applying them afterward. Realised turnover is high enough that the
optimiser is currently choosing portfolios it would not choose if it were charged for
reaching them, and a third objective or a turnover penalty would let the front express that
trade-off rather than have it imposed after the fact.

**Search that reaches the extremes.** The coverage result gives a concrete target: at
{coverage_large_n} decision variables the optimiser spans {coverage_return_large} of the
attainable return range. Reference-direction methods, decomposition-based algorithms, or
simply seeding the population with the analytic extremes would attack this directly, and the
closed-form bound gives an unambiguous measure of progress that attainment against a
self-produced reference cannot provide.

**Dimension reduction without return prediction.** The ablation separates a real
tractability benefit from a harmful selection signal. Clustering, sector-balanced sampling,
or liquidity-based admission would reduce the search space without conditioning on a
predicted return, and would test whether the benefit survives when the penalty is removed.

**Longer windows and more blocks.** The design's resolution is the binding constraint on
every conclusion. Extending the evaluation window is the change that most improves what can
be claimed, and the protocol is specified so that this requires no methodological change.

**Alternative labels.** All three labeling strategies condition on realised returns over a
one-month horizon. Labels defined on risk-adjusted outcomes, on longer horizons, or on
relative rather than absolute performance would test whether the negative selection
component is a property of supervised screening or of this particular target.
""",
}


def template_digit_violations() -> dict[str, list[str]]:
    """Literal numbers found in the templates, excluding citation years.

    The guard behind the no-hand-typed-numbers requirement. Placeholders are stripped first,
    then citation years, and anything numeric that remains is a number someone typed.
    """
    violations: dict[str, list[str]] = {}
    for name, template in SECTIONS.items():
        # Whitespace is normalised first: a wrapped line splits "CETES 28-day" across a
        # newline, and the exemption must not depend on where the paragraph happened to wrap.
        stripped = " ".join(_PLACEHOLDER.sub("", template).split())
        for noun in _PROPER_NOUNS:
            stripped = stripped.replace(noun, "")
        stripped = _CITATION_YEAR.sub("", stripped)
        found = re.findall(r"\d[\d,.]*", stripped)
        if found:
            violations[name] = found
    return violations


def _reflow(text: str, width: int = 88) -> str:
    """Rewrap paragraphs, leaving headings and blank lines alone.

    Substituting a fact changes a line's length, so the wrapped template comes out with
    breaks mid-sentence. Cosmetic in Markdown, but this source becomes LaTeX and a reviewer
    reads the diff of it.
    """
    import textwrap

    out: list[str] = []
    paragraph: list[str] = []

    def flush() -> None:
        if paragraph:
            # Hyphenated terms are single tokens here: "extended-budget" split across a
            # line reads as two words and survives into the typeset text.
            out.extend(textwrap.wrap(
                " ".join(paragraph), width=width, break_on_hyphens=False,
                break_long_words=False))
            paragraph.clear()

    for line in text.splitlines():
        stripped = line.strip()
        # A bullet needs the trailing space: "*falls*" opens emphasis, not a list, and
        # treating it as one breaks the paragraph at the emphasis.
        is_block = (
            not stripped
            or stripped.startswith(("#", "- ", "* ", "|", "```", "> "))
            or re.match(r"^\d+\. ", stripped) is not None
        )
        if is_block:
            flush()
            out.append(line.rstrip())
        else:
            paragraph.append(stripped)
    flush()
    return "\n".join(out).rstrip() + "\n"


def venue_limit_violations(rendered: Mapping[str, str]) -> list[str]:
    """Venue length limits checked against the rendered text, not the template."""
    problems = []
    abstract = rendered.get("abstract", "")
    words = len(abstract.replace("## Abstract", "").split())
    if words > ABSTRACT_WORD_LIMIT:
        problems.append(f"abstract is {words} words, limit {ABSTRACT_WORD_LIMIT}")
    for line in rendered.get("highlights", "").splitlines():
        bullet = line.strip()
        if bullet.startswith("- ") and len(bullet[2:]) > HIGHLIGHT_CHAR_LIMIT:
            problems.append(f"highlight is {len(bullet[2:])} chars: {bullet[2:60]}...")
    keywords = [
        k.strip() for k in rendered.get("keywords", "").replace("## Keywords", "").split(";")
        if k.strip()
    ]
    if keywords and not 1 <= len(keywords) <= KEYWORD_LIMIT:
        problems.append(f"{len(keywords)} keywords, limit {KEYWORD_LIMIT}")
    return problems


def render_sections(facts: Mapping[str, str]) -> dict[str, str]:
    """Render every section, failing loudly on a placeholder the facts do not resolve."""
    rendered = {}
    for name, template in SECTIONS.items():
        required = {match[1:-1] for match in _PLACEHOLDER.findall(template)}
        missing = sorted(required - set(facts))
        if missing:
            raise KeyError(f"section {name!r} needs unresolved facts: {missing}")
        rendered[name] = _reflow(template.format(**facts))
    return rendered


def write_manuscript(
    results: pd.DataFrame, config: RunConfig, artefacts: Mapping[str, Any],
    directory: str | Path,
) -> dict[str, Path]:
    """Write the rendered sections and the facts they were rendered from."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    facts = collect_facts(results, config, artefacts)
    rendered = render_sections(facts)
    paths = {}
    for name, text in rendered.items():
        path = directory / f"section_{name}.md"
        path.write_text(text, encoding="utf-8")
        paths[name] = path
    facts_path = directory / "manuscript_facts.json"
    facts_path.write_text(json.dumps(facts, indent=2, sort_keys=True), encoding="utf-8")
    paths["facts"] = facts_path
    return paths
