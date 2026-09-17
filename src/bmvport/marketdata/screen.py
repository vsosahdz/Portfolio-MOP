"""Monthly point-in-time universe screen.

Membership is decided in two stages, and keeping them separate matters:

1. **Static admission** over the evaluation window. An instrument must have traded
   throughout the months the study actually holds positions -- not throughout the download
   window, which reaches back to 2023 purely to feed model inputs. An instrument that listed
   in 2024 is perfectly holdable across 2025-2026 and is admitted; it simply has fewer early
   training observations.

2. **Monthly liquidity screen**, recomputed for each evaluation month from the trailing
   window and using *only* data dated on or before the last session of the preceding month.
   Liquidity is not a fixed property: an instrument can clear the bar in one month and fail
   it in the next, and the screen has to be able to say so.

The point-in-time discipline is the part that is easy to get wrong and impossible to detect
afterwards. A screen that peeks at the month it is selecting for produces a universe that
looks unusually tradeable in exactly the months it mattered, and every downstream result
inherits the flattery.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from ..config import RunConfig

__all__ = [
    "MonthlyMembership",
    "StaticAdmission",
    "admit_over_evaluation_window",
    "screen_month",
    "run_screen",
    "write_screen_outputs",
]


@dataclass(frozen=True, slots=True)
class StaticAdmission:
    """Verdict on whether an instrument is holdable across the whole evaluation window."""

    provider_symbol: str
    listing: str
    sessions: int
    coverage: float
    start_gap_days: int
    end_gap_days: int
    admitted: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MonthlyMembership:
    """Universe membership for one evaluation month, with its diagnostics."""

    month: str
    members: tuple[str, ...]
    evaluated: int
    exclusions: Mapping[str, int]
    traded_day_fraction: Mapping[str, float] = field(default_factory=dict)
    median_traded_value: Mapping[str, float] = field(default_factory=dict)
    below_minimum: bool = False

    @property
    def size(self) -> int:
        return len(self.members)


def _reference_calendar(frames: Mapping[str, pd.DataFrame], top: int = 25) -> pd.DatetimeIndex:
    """Exchange trading calendar, taken as the union of the best-covered series.

    Deriving it from the data rather than assuming a holiday schedule keeps the study
    independent of a calendar library whose Mexican holidays would be one more unverified
    dependency.
    """
    ranked = sorted(frames.items(), key=lambda kv: -len(kv[1]))[:top]
    if not ranked:
        return pd.DatetimeIndex([])
    dates: set[pd.Timestamp] = set()
    for _, frame in ranked:
        dates.update(frame.index)
    return pd.DatetimeIndex(sorted(dates))


def admit_over_evaluation_window(
    frames: Mapping[str, pd.DataFrame],
    listings: Mapping[str, str],
    config: RunConfig,
) -> list[StaticAdmission]:
    """Decide which instruments traded throughout the evaluation window.

    Returns one verdict per instrument, admitted or not, always with reasons when not.
    """
    months = config.windows.eval_months()
    window_start = pd.Timestamp(f"{months[0]}-01")
    calendar = _reference_calendar(frames)
    calendar = calendar[calendar >= window_start]
    total_sessions = len(calendar)
    if total_sessions == 0:
        raise ValueError("no trading sessions found inside the evaluation window")

    universe = config.universe
    verdicts: list[StaticAdmission] = []
    for symbol, frame in frames.items():
        close = frame["close"].dropna()
        close = close[close.index >= window_start]
        if close.empty:
            verdicts.append(
                StaticAdmission(symbol, listings.get(symbol, "?"), 0, 0.0, 0, 0, False,
                                ("no sessions in evaluation window",))
            )
            continue
        coverage = len(close) / total_sessions
        start_gap = (close.index.min() - calendar.min()).days
        end_gap = (calendar.max() - close.index.max()).days

        reasons: list[str] = []
        if universe.require_full_window_coverage:
            if coverage < universe.min_window_coverage_fraction:
                reasons.append(f"coverage={coverage:.2f}<{universe.min_window_coverage_fraction}")
            if start_gap > universe.max_edge_gap_days:
                reasons.append(f"listed_late_by={start_gap}d")
            if end_gap > universe.max_edge_gap_days:
                reasons.append(f"stopped_trading_early_by={end_gap}d")

        verdicts.append(
            StaticAdmission(
                provider_symbol=symbol,
                listing=listings.get(symbol, "?"),
                sessions=len(close),
                coverage=coverage,
                start_gap_days=start_gap,
                end_gap_days=end_gap,
                admitted=not reasons,
                reasons=tuple(reasons),
            )
        )
    return verdicts


def screen_month(
    month: str,
    frames: Mapping[str, pd.DataFrame],
    config: RunConfig,
) -> MonthlyMembership:
    """Compute membership for one evaluation month.

    Only data dated on or before the last session preceding the month is consulted. The
    cutoff is applied to every series before any statistic is computed, so a peek is not
    possible by construction rather than by discipline.
    """
    universe = config.universe
    month_start = pd.Timestamp(f"{month}-01")
    trailing_start = month_start - pd.DateOffset(months=universe.trailing_months)
    warmup_needed = config.features.warmup_days()

    members: list[str] = []
    exclusions: dict[str, int] = {}
    traded_fractions: dict[str, float] = {}
    traded_values: dict[str, float] = {}
    evaluated = 0

    for symbol, frame in frames.items():
        # The cutoff first: everything downstream sees only pre-month data.
        visible = frame[frame.index < month_start]
        if visible.empty:
            exclusions["no_prior_data"] = exclusions.get("no_prior_data", 0) + 1
            continue
        evaluated += 1

        close = visible["close"].dropna()
        if len(close) < warmup_needed:
            exclusions["insufficient_warmup"] = exclusions.get("insufficient_warmup", 0) + 1
            continue

        trailing = visible[visible.index >= trailing_start]
        trailing_close = trailing["close"].dropna()
        if trailing_close.empty:
            exclusions["no_trailing_data"] = exclusions.get("no_trailing_data", 0) + 1
            continue
        volume = trailing["volume"].fillna(0.0).reindex(trailing_close.index).fillna(0.0)

        traded_fraction = float((volume > 0).mean())
        traded_value = float((trailing_close * volume).median())
        traded_fractions[symbol] = traded_fraction
        traded_values[symbol] = traded_value

        diff = trailing_close.diff()
        stale_run = int((diff == 0).astype(int).groupby((diff != 0).cumsum()).sum().max() or 0)

        failed = False
        if traded_fraction < universe.min_traded_day_fraction:
            exclusions["traded_day_fraction"] = exclusions.get("traded_day_fraction", 0) + 1
            failed = True
        if traded_value < universe.min_median_traded_value_mxn:
            exclusions["median_traded_value"] = exclusions.get("median_traded_value", 0) + 1
            failed = True
        if stale_run > universe.max_stale_run_days:
            exclusions["stale_run"] = exclusions.get("stale_run", 0) + 1
            failed = True
        if not failed:
            members.append(symbol)

    return MonthlyMembership(
        month=month,
        members=tuple(sorted(members)),
        evaluated=evaluated,
        exclusions=dict(sorted(exclusions.items())),
        traded_day_fraction=traded_fractions,
        median_traded_value=traded_values,
        below_minimum=len(members) < universe.min_universe_size,
    )


def run_screen(
    frames: Mapping[str, pd.DataFrame],
    listings: Mapping[str, str],
    config: RunConfig,
) -> tuple[list[StaticAdmission], list[MonthlyMembership]]:
    """Run static admission and then the monthly screen over admitted instruments."""
    admissions = admit_over_evaluation_window(frames, listings, config)
    admitted = {a.provider_symbol for a in admissions if a.admitted}
    pool = {s: f for s, f in frames.items() if s in admitted}
    memberships = [screen_month(m, pool, config) for m in config.windows.eval_months()]
    return admissions, memberships


def write_screen_outputs(
    admissions: list[StaticAdmission],
    memberships: list[MonthlyMembership],
    directory: str | Path,
) -> dict[str, Path]:
    """Persist admission verdicts, monthly membership and the screen diagnostics."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    admission_path = directory / "screen_admission.csv"
    with admission_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["provider_symbol", "listing", "sessions", "coverage", "start_gap_days",
             "end_gap_days", "admitted", "reasons"]
        )
        for a in admissions:
            writer.writerow([a.provider_symbol, a.listing, a.sessions, f"{a.coverage:.4f}",
                             a.start_gap_days, a.end_gap_days, a.admitted, ";".join(a.reasons)])
    paths["admission"] = admission_path

    membership_path = directory / "screen_membership.csv"
    with membership_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["month", "provider_symbol"])
        for m in memberships:
            for symbol in m.members:
                writer.writerow([m.month, symbol])
    paths["membership"] = membership_path

    # Threshold distributions are recorded so the chosen cut-offs can be justified from the
    # observed data in the manuscript rather than asserted.
    diagnostics = []
    for m in memberships:
        values = np.array(list(m.median_traded_value.values()), dtype=float)
        fractions = np.array(list(m.traded_day_fraction.values()), dtype=float)
        diagnostics.append(
            {
                "month": m.month,
                "evaluated": m.evaluated,
                "admitted": m.size,
                "below_minimum": m.below_minimum,
                "exclusions": dict(m.exclusions),
                "traded_value_percentiles": {
                    str(p): float(np.percentile(values, p)) for p in (10, 25, 50, 75, 90)
                } if values.size else {},
                "traded_day_fraction_percentiles": {
                    str(p): float(np.percentile(fractions, p)) for p in (10, 25, 50, 75, 90)
                } if fractions.size else {},
            }
        )
    diagnostics_path = directory / "screen_diagnostics.json"
    diagnostics_path.write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    paths["diagnostics"] = diagnostics_path
    return paths
