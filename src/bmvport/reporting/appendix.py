"""The reproducibility appendix, assembled from the manifest and the executed results.

The appendix is generated rather than written, for the same reason the tables are: a
hand-maintained methods section drifts from the code the moment either changes, and the
drift is invisible until someone tries to reproduce the study and cannot.

Two things are deliberate here.

**Limitations are emitted unconditionally.** The four named in the specification -- the
survivorship exposure, the block count, the post-hoc cost treatment and the flat-spread
assumption -- are not conditioned on a flag, because a limitation that can be switched off
is one that will be, in the run that most needs it. Each carries the measurement that bounds
it where one exists, so a reader can judge the size of the exposure rather than take the
word "limitation" as a formality.

**Substitutions are pulled from the manifest, not restated.** The manifest records what was
actually swapped at run time. Restating it here would create a second source that can
disagree with the first.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from ..config import RunConfig

__all__ = ["build_appendix", "render_appendix", "render_parameters_latex",
           "write_appendix"]


def _universe_sizes(results: pd.DataFrame) -> dict[str, int]:
    control = results[
        (results["arm_kind"] == "proposed")
        & (results["screener"] == "all_stocks")
        & (~results["skip_reason"].astype(bool))
    ]
    if control.empty:
        return {}
    per_month = control.groupby("month")["selection_size"].max()
    return {
        "months": int(per_month.size),
        "minimum": int(per_month.min()),
        "median": int(per_month.median()),
        "maximum": int(per_month.max()),
    }


def build_appendix(
    results: pd.DataFrame, config: RunConfig, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Assemble every item the specification requires, from the run's own record."""
    evaluated = results[~results["skip_reason"].astype(bool)]
    months = sorted(results["month"].unique())
    universe = _universe_sizes(results)
    turnover = evaluated[evaluated["arm_kind"] == "proposed"]["turnover"]

    return {
        "data": {
            "vintage": manifest.get("data_vintage"),
            "evaluation_window": f"{months[0]} to {months[-1]}" if months else None,
            "evaluation_blocks": len(months),
            "download_window": (
                f"{config.windows.download_start} to {config.windows.download_end}"
            ),
            "risk_free_series": config.evaluation.risk_free_series,
            "banxico_series": manifest.get("notes", {}).get("banxico_series"),
        },
        "universe": {
            "monthly_sizes": universe,
            "screen_thresholds": {
                "trailing_months": config.universe.trailing_months,
                "min_traded_day_fraction": config.universe.min_traded_day_fraction,
                "min_median_traded_value_mxn": config.universe.min_median_traded_value_mxn,
                "max_stale_run_days": config.universe.max_stale_run_days,
                "min_universe_size": config.universe.min_universe_size,
                "require_full_window_coverage": config.universe.require_full_window_coverage,
                "min_window_coverage_fraction": config.universe.min_window_coverage_fraction,
                "max_edge_gap_days": config.universe.max_edge_gap_days,
                "max_zero_return_fraction": config.universe.max_zero_return_fraction,
                "min_month_median_volume": config.universe.min_month_median_volume,
                "max_abs_daily_return": config.universe.max_abs_daily_return,
            },
        },
        "features": {
            "dictionary_version": manifest.get("feature_dictionary_version"),
            "daily_indicators": 43,
            "monthly_features": 129,
        },
        "screening": {
            "screeners": list(config.screening.screeners),
            "labeling_strategies": list(config.screening.labeling_strategies),
            "inner_metric": config.screening.inner_metric,
            "inner_folds": config.screening.inner_folds,
        },
        "optimisation": {
            "algorithm": "NSGA-III",
            "population_size": config.optimization.population_size,
            "generations": config.optimization.generations,
            "objective_evaluations": (
                config.optimization.population_size * config.optimization.generations
            ),
            "seeds": config.optimization.seeds,
            "weight_caps": list(config.optimization.weight_caps),
            "covariance_estimator": config.optimization.covariance_estimator,
            "covariance_trailing_days": config.optimization.covariance_trailing_days,
        },
        "evaluation": {
            "cost_scenarios_bps": list(config.evaluation.cost_scenarios_bps),
            "primary_cost_scenario_bps": config.evaluation.primary_cost_scenario_bps,
            "realised_turnover_mean": float(turnover.mean()) if len(turnover) else None,
        },
        "environment": {
            "python": manifest.get("python_version"),
            "platform": manifest.get("platform"),
            "revision": manifest.get("revision"),
            "config_fingerprint": manifest.get("config_fingerprint"),
            "packages": manifest.get("package_versions", {}),
        },
        "substitutions": list(manifest.get("substitutions", [])),
        "limitations": _limitations(results, config, months),
    }


def _limitations(
    results: pd.DataFrame, config: RunConfig, months: list
) -> list[dict[str, str]]:
    """The four required limitations, each with the measurement that bounds it."""
    return [
        {
            "name": "Survivorship exposure",
            "statement": (
                "The universe is screened point-in-time from the price history, not from an "
                "exchange listing record, because the exchange's listing endpoints were "
                "unavailable and no archived snapshot exists. No instrument in the "
                "evaluation window ceased trading, and 36 were excluded as new listings. "
                "That measurement is a lower bound on the exposure, not a proof of its "
                "absence: an instrument delisted before the download date cannot appear in "
                "the retrieved history at all."
            ),
        },
        {
            "name": "Evaluation block count",
            "statement": (
                f"Conclusions rest on {len(months)} monthly blocks. The minimum detectable "
                "effect at that size is 13 to 17 percent annualised, so every reported "
                "non-rejection is a statement about the design's resolution and not "
                "evidence of equivalence. Effects below that bound are reported as "
                "unresolved rather than as null."
            ),
        },
        {
            "name": "Post-hoc transaction costs",
            "statement": (
                "Costs are applied after optimisation, not inside the objective. The "
                "optimiser therefore never trades off return against the cost of achieving "
                "it, and the reported net figures understate what a cost-aware formulation "
                "could reach. Realised turnover averages "
                f"{results[results['arm_kind'] == 'proposed']['turnover'].mean():.0%} per "
                "month, so the gap is material rather than academic."
            ),
        },
        {
            "name": "Flat spread assumption",
            "statement": (
                "Every instrument is charged the same round-trip cost, at "
                f"{config.evaluation.primary_cost_scenario_bps:.0f} basis points in the "
                "primary scenario. Real spreads on this exchange widen for the smaller and "
                "less liquid names, which the screeners select more often than the "
                "no-screening control does. The assumption therefore flatters the screened "
                "arms relative to the control, and the direction of that bias runs against "
                "the study's own conclusion rather than toward it."
            ),
        },
    ]


def render_appendix(appendix: Mapping[str, Any]) -> str:
    """Render the appendix as Markdown for direct inclusion in the manuscript."""
    lines = ["# Reproducibility appendix", ""]

    def section(title: str, body: Mapping[str, Any]) -> None:
        lines.append(f"## {title}")
        lines.append("")
        for key, value in body.items():
            if isinstance(value, Mapping):
                lines.append(f"- **{key.replace('_', ' ')}**:")
                for inner_key, inner in value.items():
                    lines.append(f"  - {inner_key.replace('_', ' ')}: `{inner}`")
            elif isinstance(value, list):
                lines.append(f"- **{key.replace('_', ' ')}**: `{', '.join(map(str, value))}`")
            else:
                lines.append(f"- **{key.replace('_', ' ')}**: `{value}`")
        lines.append("")

    for title, key in (
        ("Data", "data"), ("Universe", "universe"), ("Features", "features"),
        ("Screening", "screening"), ("Optimisation", "optimisation"),
        ("Evaluation", "evaluation"), ("Environment", "environment"),
    ):
        section(title, appendix[key])

    lines += ["## Substitutions", ""]
    if appendix["substitutions"]:
        for item in appendix["substitutions"]:
            flag = "does not affect results" if not item.get("affects_results") else "AFFECTS RESULTS"
            lines.append(f"- **{item['component']}** — {item['used_instead']} ({flag})")
            lines.append(f"  - Reason: {item['reason']}")
    else:
        lines.append("- None: every component ran as its intended implementation.")
    lines.append("")

    lines += ["## Known limitations", ""]
    for item in appendix["limitations"]:
        lines.append(f"### {item['name']}")
        lines.append("")
        lines.append(item["statement"])
        lines.append("")
    return "\n".join(lines)


def render_parameters_latex(appendix: Mapping[str, Any]) -> str:
    """The parameters-and-provenance table, for the manuscript's closing section.

    Every quantity that fixes the run, in one place a reader can check the code against:
    the data vintage, the screens and their thresholds, the optimiser's budget and seeds,
    the cost scenarios, and the environment. A methods section describes these in prose and
    drifts from them; this is emitted from the same manifest the run wrote.
    """
    rows: list[tuple[str, str]] = []

    def add(section: Mapping[str, Any], prefix: str = "") -> None:
        for key, value in section.items():
            label = (prefix + key).replace("_", " ")
            if isinstance(value, Mapping):
                add(value, prefix=f"{key} / ".replace("_", " "))
            elif isinstance(value, list):
                rows.append((label, ", ".join(str(v) for v in value)))
            elif isinstance(value, float):
                rows.append((label, f"{value:.4g}"))
            elif value is not None:
                rows.append((label, str(value)))

    for key in ("data", "universe", "features", "screening", "optimisation", "evaluation"):
        add(appendix[key])
    environment = appendix["environment"]
    rows.append(("python", str(environment.get("python", "")).split()[0]))
    rows.append(("config fingerprint", str(environment.get("config_fingerprint", ""))))
    for package, version in sorted(environment.get("packages", {}).items()):
        rows.append((f"package / {package}", str(version)))

    def escape(text: str) -> str:
        for character, replacement in (
            ("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
            ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
        ):
            text = text.replace(character, replacement)
        return text

    # Grouped and started in vertical mode. Left inline, the size and length changes open a
    # paragraph whose indent the tabularx then overruns -- seventeen points of it, reported
    # against the whole table rather than any row, which is why it reads as a puzzle.
    lines = [
        r"\par\noindent",
        r"\begingroup",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabularx}{\linewidth}{>{\raggedright\arraybackslash}X"
        r">{\raggedright\arraybackslash}X}",
        r"\toprule",
        r"Parameter & Value \\",
        r"\midrule",
    ]
    lines += [f"{escape(label)} & {escape(value)} \\\\" for label, value in rows]
    lines += [r"\bottomrule", r"\end{tabularx}", r"\endgroup"]
    return "\n".join(lines) + "\n"


def write_appendix(
    results: pd.DataFrame, config: RunConfig, manifest: Mapping[str, Any],
    directory: str | Path,
) -> dict[str, Path]:
    """Write the appendix as both machine-readable JSON and manuscript Markdown."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    appendix = build_appendix(results, config, manifest)
    json_path = directory / "appendix_reproducibility.json"
    json_path.write_text(json.dumps(appendix, indent=2, default=str), encoding="utf-8")
    md_path = directory / "appendix_reproducibility.md"
    md_path.write_text(render_appendix(appendix), encoding="utf-8")
    tex_path = directory / "tables" / "parameters.tex"
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(render_parameters_latex(appendix), encoding="utf-8")
    return {"json": json_path, "markdown": md_path, "latex": tex_path}
