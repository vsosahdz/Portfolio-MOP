"""The tidy results table: one row per portfolio-month-seed-cost-scenario.

Every table and figure in the manuscript is generated from this one artefact. Nothing is
typed by hand, and nothing downstream reads an intermediate stage's output -- which is what
makes "same cache, same seeds, identical tables" a checkable claim rather than an aspiration.

**The schema is built so that a favourable and an unfavourable reading of the results are
equally easy to produce.** That is a deliberate design property, not a stylistic preference.
A results table that carries gross returns but not turnover, or returns but not the
attribution, permits only the flattering account: the reader cannot see the cost of the
rotation or ask how much of the gain was the currency. Here each of those is a column, so
the limitations section is assembled from the same rows as the contributions section and
cannot quietly disagree with it.

Concretely, every row carries:

- ``return_gross`` **and** ``return_net`` at the row's cost scenario, so the drag is visible
- ``turnover``, so the reader can see what the strategy had to trade to get there
- ``sigma_ex_ante`` **and** ``sigma_realized`` in separate columns, never conflated
- the attribution terms, so an active return can be traced to selection, allocation or a
  currency tilt
- ``cap_binding``, so a null concentration effect can be told apart from an inert constraint
- ``skip_reason``, so a missing cell is visibly missing rather than absent

**Skipped cells are rows.** A cell that produced no portfolio -- an empty selection, an
infeasible cap, a failed solve -- appears with its reason. Dropping them would make the grid
look complete and would silently change the denominator of every aggregate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

__all__ = ["ResultRow", "KEY_FIELDS", "build_results_table", "write_results_table",
           "completeness_report",
           "check_key_uniqueness"]

# The identifying key. Any two rows agreeing on all of these describe the same thing and
# one of them is a bug.
KEY_FIELDS = (
    "arm_kind",
    "month",
    "labeling",
    "screener",
    "allocator",
    "weight_cap",
    "covariance_estimator",
    "seed",
    "cost_bps",
)


@dataclass(frozen=True, slots=True)
class ResultRow:
    """One evaluated portfolio-month under one cost scenario."""

    # --- identity ---------------------------------------------------------------------
    arm_kind: str  # "proposed" | "comparator"
    month: str
    labeling: str
    screener: str
    allocator: str
    weight_cap: float
    covariance_estimator: str
    seed: int | None
    cost_bps: float
    comparator_tier: str = ""
    literature_reference: str = ""

    # --- outcome ----------------------------------------------------------------------
    return_gross: float = float("nan")
    return_net: float = float("nan")
    sigma_ex_ante: float = float("nan")
    sigma_realized: float = float("nan")
    sharpe: float = float("nan")
    risk_free: float = float("nan")
    turnover: float = float("nan")
    holdings: int = 0
    selection_size: int = 0

    # --- attribution ------------------------------------------------------------------
    attribution_market: float = float("nan")
    attribution_selection: float = float("nan")
    attribution_allocation: float = float("nan")
    attribution_fx_tilt: float = float("nan")
    attribution_residual: float = float("nan")
    portfolio_sic_weight: float = float("nan")
    universe_sic_share: float = float("nan")

    # --- diagnostics ------------------------------------------------------------------
    cap_binding: bool | None = None
    front_size: int = 0
    hypervolume: float = float("nan")
    dropped_at_entry: int = 0
    early_exits: int = 0
    skip_reason: str = ""

    @property
    def skipped(self) -> bool:
        return bool(self.skip_reason)


def build_results_table(rows: Iterable[ResultRow]) -> pd.DataFrame:
    """Assemble rows into the tidy table, ordered by key."""
    records = [asdict(r) for r in rows]
    if not records:
        return pd.DataFrame(columns=[*KEY_FIELDS])
    frame = pd.DataFrame(records)
    return frame.sort_values(list(KEY_FIELDS)).reset_index(drop=True)


def check_key_uniqueness(frame: pd.DataFrame) -> None:
    """Verify the identifying key selects at most one row.

    Raises:
        ValueError: naming the duplicated keys. A duplicate means two different portfolios
            are claiming the same identity, and every aggregate over that cell would double
            count one of them.
    """
    if frame.empty:
        return
    missing = [f for f in KEY_FIELDS if f not in frame.columns]
    if missing:
        raise ValueError(f"results table is missing key fields: {missing}")
    duplicated = frame.duplicated(subset=list(KEY_FIELDS), keep=False)
    if duplicated.any():
        offenders = (
            frame.loc[duplicated, list(KEY_FIELDS)]
            .astype(str)
            .agg(" | ".join, axis=1)
            .unique()
        )
        raise ValueError(
            f"{len(offenders)} duplicated result keys; the first is: {offenders[0]}"
        )


def write_results_table(frame: pd.DataFrame, directory: str | Path) -> dict[str, Path]:
    """Persist the tidy table and a completeness summary.

    The summary exists because a reader of the results has to be able to tell a grid that
    ran completely from one that ran partially. Reporting only the rows that succeeded would
    make every aggregate silently conditional on success.
    """
    import json

    check_key_uniqueness(frame)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    table_path = directory / "results.parquet"
    frame.to_parquet(table_path)

    skipped = frame[frame["skip_reason"].astype(bool)] if len(frame) else frame
    summary: dict[str, Any] = {
        "rows": int(len(frame)),
        "skipped_rows": int(len(skipped)),
        "skip_reasons": (
            skipped["skip_reason"].value_counts().to_dict() if len(skipped) else {}
        ),
        "months": sorted(frame["month"].unique().tolist()) if len(frame) else [],
        "cost_scenarios": sorted(frame["cost_bps"].unique().tolist()) if len(frame) else [],
        "arms": (
            frame.groupby("arm_kind").size().to_dict() if len(frame) else {}
        ),
    }
    if len(frame):
        evaluated = frame[~frame["skip_reason"].astype(bool)]
        if len(evaluated):
            # Turnover is summarised here and not only in the tables, because it is the
            # cost side of the result and it should be as hard to overlook as the return.
            summary["median_turnover"] = float(evaluated["turnover"].median())
            summary["median_holdings"] = float(evaluated["holdings"].median())
            binding = evaluated["cap_binding"].dropna()
            if len(binding):
                summary["cap_binding_share"] = float(binding.astype(bool).mean())

    summary_path = directory / "results_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    return {"table": table_path, "summary": summary_path}


GA_ALLOCATORS = ("ga_min_risk", "ga_knee", "ga_max_return")


def completeness_report(frame: pd.DataFrame, config) -> dict:
    """Every factorial cell must be a full set of results or carry a skip that explains it.

    A cell short of its expected rows is not by itself a defect. Two shortfalls are correct
    by construction: an infeasible weight cap is detected before any seed is drawn, so it
    records one skip per allocator rather than one per seed; and a degenerate front supplies
    fewer than three portfolios, recording a skip for each it cannot supply.

    What the check is looking for is the third case -- a shortfall with no skip behind it,
    which is indistinguishable from a run that silently never happened. An earlier pass
    counted only totals and reported a mismatch it could not attribute; this attributes it.
    """
    expected_rows = (
        len(GA_ALLOCATORS) * config.optimization.seeds
        * len(config.evaluation.cost_scenarios_bps)
    )
    proposed = frame[
        (frame["arm_kind"] == "proposed") & (frame["allocator"].isin(GA_ALLOCATORS))
    ]
    cells = proposed.groupby(["labeling", "screener", "month", "weight_cap"])
    unexplained, explained = [], []
    for key, group in cells:
        if len(group) == expected_rows:
            continue
        skips = group[group["skip_reason"].astype(bool)]
        record = {
            "labeling": key[0], "screener": key[1], "month": key[2], "weight_cap": key[3],
            "present": int(len(group)), "expected": expected_rows,
            "skip_reasons": sorted(set(skips["skip_reason"].astype(str))),
        }
        (explained if len(skips) else unexplained).append(record)
    return {
        "expected_rows_per_cell": expected_rows,
        "cells": int(cells.ngroups),
        "complete": int(sum(len(g) == expected_rows for _, g in cells)),
        "short_with_recorded_skip": explained,
        "short_without_any_skip": unexplained,
        "passes": not unexplained,
    }
