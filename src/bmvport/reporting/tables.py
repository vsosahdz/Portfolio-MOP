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
    "performance_table",
    "comparator_table",
    "attribution_table",
    "profile_tables",
    "write_tables",
]

RETURN_METRICS = frozenset({"return_net", "return_gross"})


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
        tex_path.write_text(
            table.to_latex(index=False, float_format="%.4f", escape=True), encoding="utf-8"
        )
        paths[name] = csv_path
    return paths
