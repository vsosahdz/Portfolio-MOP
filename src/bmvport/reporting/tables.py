"""Manuscript tables, generated from the stored results.

Nothing here is typed by hand and nothing reads an intermediate stage: every number traces
to the tidy results table, so the limitations section and the contributions section are
assembled from the same rows and cannot quietly disagree.

Three conventions apply to every table and are enforced rather than remembered:

**The cost scenario is stated in the table**, not the caption. A performance figure without
its cost assumption is not interpretable, and captions get separated from tables.

**Gross figures are labelled as an upper bound.** They are reported because a reader should
be able to see the cost drag, not because they are achievable.

**A non-rejection carries its detectable bound.** "No significant difference" over eighteen
blocks says more about the design than the world, and a table that omits the bound invites
exactly that misreading.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

__all__ = [
    "to_latex",
    "COLUMN_LABELS",
    "performance_table",
    "comparator_table",
    "attribution_table",
    "profile_tables",
    "write_tables",
]

RETURN_METRICS = frozenset({"return_net", "return_gross"})

# Human column headings for the typeset tables. The CSV keeps the machine names, because a
# reader loading it into a dataframe wants stable identifiers; a reader of the paper wants a
# heading that fits and reads. Escaped underscores in a machine name also render as visible
# gaps, which is how the first typeset draft came out.
COLUMN_LABELS: dict[str, str] = {
    "screener": "Screening arm", "arm": "Arm", "tier": "Tier", "profile": "Profile",
    "metric": "Metric", "extraction_rule": "Extraction rule",
    "net_annualised": "Net",
    "gross_annualised_upper_bound": "Gross\\textsuperscript{a}",
    "median_sharpe": "Sharpe", "mean_turnover": "Turn.",
    "mean_selection_size": "Names", "positive_months": "Up", "n_blocks": "Blocks",
    "market_annualised": "Market", "selection_annualised": "Selection",
    "allocation_annualised": "Allocation", "fx_tilt_annualised": "FX tilt",
    "portfolio_sic_weight": "Portfolio SIC", "universe_sic_share": "Universe SIC",
    "max_abs_residual": "Max residual",
    "difference_vs_best_proposed": "Diff.",
    "p_adjusted": "$p$ adj.", "minimum_detectable_annualised": "MDE",
    "verdict": "Verdict", "value": "Value", "annualised": "Annualised",
}

# Columns rendered as percentages rather than as fractions, so the table agrees with the
# prose. Everything the study annualises is a rate; Sharpe and the counts are not.
PERCENT_COLUMNS = frozenset({
    "net_annualised", "gross_annualised_upper_bound", "mean_turnover",
    "market_annualised", "selection_annualised", "allocation_annualised",
    "fx_tilt_annualised", "portfolio_sic_weight", "universe_sic_share",
    "difference_vs_best_proposed", "minimum_detectable_annualised", "annualised",
})

# A verdict sentence is too wide for a table column and says the same thing every time.
VERDICT_SHORT = {
    "not rejected; effect below the design's resolution": "unresolved",
    "not tested": "--", "": "",
}


def _format_cell(column: str, value) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "--"
    if column == "verdict":
        return VERDICT_SHORT.get(str(value), str(value))
    if column in PERCENT_COLUMNS:
        return f"{float(value) * 100:.1f}\\%"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):d}"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value).replace("_", " ")


def _best_proposed_cell(results: pd.DataFrame, cost_bps: float) -> tuple[str, float]:
    """The single cell the comparator confrontation is measured against.

    Needed for the table's footnote. The confrontation compares each comparator with the
    best individual cell, while the table's return column shows screener-level aggregates,
    so a reader who subtracts the two visible columns gets a third number. Naming the
    reference is the difference between a table that is precise and one that misleads.
    """
    frame = results[
        (results["arm_kind"] == "proposed") & (results["cost_bps"] == cost_bps)
        & (~results["skip_reason"].astype(bool))
    ]
    per_cell = (
        frame.groupby(["labeling", "screener", "allocator", "weight_cap", "month"])
        ["return_net"].median()
        .groupby(level=[0, 1, 2, 3]).mean()
    )
    key = per_cell.idxmax()
    label = f"{key[1]}, {key[2]}, cap {float(key[3]):g}, {key[0]} labelling"
    return label.replace("_", " "), float(per_cell.max()) * 12


def to_latex(
    table: pd.DataFrame, *, drop: Sequence[str] = (), size: str | None = None,
    notes: Sequence[str] = (),
) -> str:
    """Render a table as a booktabs tabular sized to the text block.

    Columns whose value is constant down the table are dropped rather than repeated: the
    cost scenario belongs in the caption, and a column of identical numbers is width spent
    on nothing. The first typeset draft kept them and both tables ran off the page, losing
    their rightmost columns silently -- LaTeX does not warn when a tabular overflows into
    the margin.
    """
    frame = table.drop(columns=[c for c in drop if c in table.columns])
    # Constant among the rows that have a value at all. A column whose only distinct entry
    # is repeated down the table carries nothing per row and costs width the table does not
    # have; what it says goes into a footnote instead of being lost. Missing entries do not
    # make a column informative -- the verdict column read "unresolved" wherever it applied
    # and blank elsewhere, and kept the table over the text block until it was dropped.
    lifted: list[str] = []
    for column in list(frame.columns):
        distinct = {
            v for v in frame[column].tolist()
            if v is not None and str(v) != "" and not (isinstance(v, float) and pd.isna(v))
        }
        if len(distinct) <= 1 and len(frame) > 1:
            label = COLUMN_LABELS.get(column, column.replace("_", " "))
            if distinct:
                lifted.append(f"{label}: {_format_cell(column, distinct.pop())}")
            frame = frame.drop(columns=[column])
    notes = list(notes) + ([" -- ".join(lifted)] if lifted else [])

    # Tested on the dtype, not on equality with object: pandas stores strings in an Arrow
    # dtype here, so the object comparison silently right-aligns every text column.
    #
    # The first text column is an X column and the rest are fixed. tabularx then solves for
    # a table exactly \linewidth wide, absorbing the slack in the label. A plain tabular is
    # as wide as its content demands and runs into the margin without a warning a reader
    # would see -- the first typeset draft did so by up to 97pt while appearing to fit.
    kinds = ["r" if pd.api.types.is_numeric_dtype(frame[c]) else "l" for c in frame.columns]
    flexible = kinds.index("l") if "l" in kinds else 0
    # Ragged rather than justified: an X column justifies by stretching interword space, and
    # a label column of short entries stretches until a long one overflows instead.
    kinds[flexible] = r">{\raggedright\arraybackslash}X"
    alignment = "".join(kinds)
    header = " & ".join(
        COLUMN_LABELS.get(c, c.replace("_", " ").capitalize()) for c in frame.columns
    )
    body = [
        " & ".join(_format_cell(c, row[c]) for c in frame.columns) + r" \\"
        for _, row in frame.iterrows()
    ]
    # Notes sit after the tabular rather than in a \multicolumn row: a multicolumn cell is
    # as wide as the columns it spans and does not wrap, so a sentence-length note runs off
    # the page exactly like the columns this function exists to keep on it.
    # threeparttable is the idiom for table notes: it sets them to the table's own width and
    # keeps them inside the float, where a trailing minipage merely looked as though it did.
    # Sized by column count rather than fixed. At small, seven columns leave the label
    # column too narrow for its own entries and every cell overflows individually -- which
    # reads in the log as thirty-one overfull boxes and on the page as text crossing a rule.
    if size is None:
        size = "footnotesize" if len(frame.columns) >= 6 else "small"
    block = [
        r"\begin{threeparttable}",
        f"\\{size}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabularx}{\linewidth}{" + alignment + "}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabularx}",
    ]
    live = [note for note in notes if note]
    if live:
        block.append(r"\begin{tablenotes}[flushleft]\footnotesize")
        block += [r"  \item " + note for note in live]
        block.append(r"\end{tablenotes}")
    block.append(r"\end{threeparttable}")
    return "\n".join(block) + "\n"


def _annualise(monthly: float) -> float:
    return monthly * 12.0


def performance_table(results: pd.DataFrame, cost_bps: float) -> pd.DataFrame:
    """Per-screener performance, with the cost side beside the return side."""
    frame = results[
        (results["arm_kind"] == "proposed")
        & (results["cost_bps"] == cost_bps)
        & (~results["skip_reason"].astype(bool))
    ]
    rows = []
    for screener, group in frame.groupby("screener"):
        monthly = group.groupby("month")["return_net"].median()
        gross = group.groupby("month")["return_gross"].median()
        rows.append(
            {
                "screener": screener,
                "cost_bps": cost_bps,
                "net_annualised": _annualise(float(monthly.mean())),
                "gross_annualised_upper_bound": _annualise(float(gross.mean())),
                "median_sharpe": float(group["sharpe"].median()),
                "mean_turnover": float(group["turnover"].mean()),
                "mean_selection_size": float(group["selection_size"].mean()),
                "positive_months": int((monthly > 0).sum()),
                "n_blocks": int(monthly.size),
            }
        )
    return pd.DataFrame(rows).sort_values("net_annualised", ascending=False)


def comparator_table(results: pd.DataFrame, tests: Sequence, cost_bps: float) -> pd.DataFrame:
    """Every arm ranked, with the confrontation's verdict and detectable bound attached."""
    frame = results[
        (results["cost_bps"] == cost_bps) & (~results["skip_reason"].astype(bool))
    ]
    verdicts = {t.name: t for t in tests}
    rows = []
    for (arm, kind, tier), group in frame.groupby(["allocator", "arm_kind", "comparator_tier"]):
        if kind == "proposed":
            continue
        monthly = group.groupby("month")["return_net"].median()
        test = verdicts.get(arm)
        rows.append(
            {
                "arm": arm, "tier": tier, "cost_bps": cost_bps,
                "net_annualised": _annualise(float(monthly.mean())),
                "median_sharpe": float(group["sharpe"].median()),
                "mean_turnover": float(group["turnover"].mean()),
                "difference_vs_best_proposed": (
                    _annualise(test.effect) if test else float("nan")
                ),
                "p_adjusted": test.p_adjusted if test else float("nan"),
                "minimum_detectable_annualised": (
                    _annualise(test.minimum_detectable) if test else float("nan")
                ),
                "verdict": test.verdict if test else "not tested",
                "reference_series_uncosted": tier == "tier0",
            }
        )
    for screener, group in frame[frame["arm_kind"] == "proposed"].groupby("screener"):
        monthly = group.groupby("month")["return_net"].median()
        rows.append(
            {
                "arm": f"proposed:{screener}", "tier": "factorial", "cost_bps": cost_bps,
                "net_annualised": _annualise(float(monthly.mean())),
                "median_sharpe": float(group["sharpe"].median()),
                "mean_turnover": float(group["turnover"].mean()),
                "difference_vs_best_proposed": float("nan"),
                "p_adjusted": float("nan"),
                "minimum_detectable_annualised": float("nan"),
                "verdict": "", "reference_series_uncosted": False,
            }
        )
    return pd.DataFrame(rows).sort_values("net_annualised", ascending=False)


def attribution_table(results: pd.DataFrame, cost_bps: float) -> pd.DataFrame:
    """The decomposition that separates method skill from market and currency."""
    frame = results[
        (results["arm_kind"] == "proposed")
        & (results["cost_bps"] == cost_bps)
        & (~results["skip_reason"].astype(bool))
    ]
    rows = []
    for screener, group in frame.groupby("screener"):
        rows.append(
            {
                "screener": screener, "cost_bps": cost_bps,
                "market_annualised": _annualise(float(group["attribution_market"].mean())),
                "selection_annualised": _annualise(float(group["attribution_selection"].mean())),
                "allocation_annualised": _annualise(float(group["attribution_allocation"].mean())),
                "fx_tilt_annualised": _annualise(float(group["attribution_fx_tilt"].mean())),
                "portfolio_sic_weight": float(group["portfolio_sic_weight"].mean()),
                "universe_sic_share": float(group["universe_sic_share"].mean()),
                "max_abs_residual": float(group["attribution_residual"].abs().max()),
            }
        )
    return pd.DataFrame(rows).sort_values("selection_annualised", ascending=False)


def profile_tables(results: pd.DataFrame, cost_bps: float) -> pd.DataFrame:
    """The three investor profiles, each against its designated metric.

    The moderate profile uses the knee-point allocator, which replaces the original study's
    "average risk" pick; the extraction rule is named in the table so the change is visible
    where the numbers are, not only in the text.
    """
    frame = results[
        (results["arm_kind"] == "proposed")
        & (results["cost_bps"] == cost_bps)
        & (~results["skip_reason"].astype(bool))
    ]
    profiles = {
        "aggressive": ("return_net", "ga_max_return", "maximum-return end of the front"),
        "conservative": ("sigma_realized", "ga_min_risk", "minimum-risk end of the front"),
        "moderate": ("sharpe", "ga_knee", "knee point, replacing an index-based midpoint"),
    }
    rows = []
    for profile, (metric, allocator, rule) in profiles.items():
        subset = frame[frame["allocator"] == allocator]
        for screener, group in subset.groupby("screener"):
            monthly = group.groupby("month")[metric].median()
            rows.append(
                {
                    "profile": profile, "metric": metric, "extraction_rule": rule,
                    "screener": screener, "cost_bps": cost_bps,
                    "value": float(monthly.mean()),
                    "annualised": _annualise(float(monthly.mean()))
                    if metric in RETURN_METRICS else float("nan"),
                    "n_blocks": int(monthly.size),
                }
            )
    return pd.DataFrame(rows)


# Columns carried in the CSV for completeness but omitted from the typeset table, where
# width is the binding constraint. Nothing analytical is dropped: these are either recorded
# elsewhere in the paper or reconstructible from what remains.
DROPPED: dict[str, tuple[str, ...]] = {
    "comparators": ("reference_series_uncosted",),
    "attribution": ("max_abs_residual",),
    "profiles": ("extraction_rule",),
}


def write_tables(
    results: pd.DataFrame, tests: Sequence, cost_bps: float, directory: str | Path
) -> dict[str, Path]:
    """Write every manuscript table in machine-readable and typeset-ready form."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    built = {
        "performance": performance_table(results, cost_bps),
        "comparators": comparator_table(results, tests, cost_bps),
        "attribution": attribution_table(results, cost_bps),
        "profiles": profile_tables(results, cost_bps),
    }
    paths: dict[str, Path] = {}
    for name, table in built.items():
        csv_path = directory / f"table_{name}.csv"
        table.to_csv(csv_path, index=False)
        tex_path = directory / f"table_{name}.tex"
        notes = []
        if name == "comparators":
            label, value = _best_proposed_cell(results, cost_bps)
            notes.append(
                f"Differences are against the best individual cell "
                f"({label}, {value * 100:.1f}\\% annualised), not against the "
                f"screener-level figures in this table."
            )
        if name == "performance":
            notes.append(
                r"\textsuperscript{a}Gross is an upper bound, reported to show the cost drag."
            )
        tex_path.write_text(
            to_latex(table, drop=DROPPED.get(name, ()), notes=notes), encoding="utf-8")
        paths[name] = csv_path
    return paths
