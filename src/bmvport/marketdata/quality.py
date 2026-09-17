"""Data-quality gates over the cached price series.

The study's price source is fixed: no institutional feed is available, so there is nothing
to migrate to when a defect appears. Defects therefore have to be *detected and handled*
rather than escaped, and the handling has to be recorded so a referee can see what was
excluded and why.

**What the measurement found.** Across 91 liquid series over 2023-2026, 85 are clean in both
the training and evaluation periods, 6 carry defects confined to the pre-2025 training and
covariance period, and none carry defects inside the evaluation window. One name dominates:
`AMXB.MX` (América Móvil, the largest issuer on the exchange) shows 117 consecutive sessions
at a fixed 19.50 on a median volume of zero, from 2023-03-15 to 2023-08-31 -- the period in
which the B series was being created. Including those sessions understates its estimated
volatility by roughly 18%.

**Why a trading-day count does not catch it.** The exchange was open and prices were
published on every one of those days. What was absent was trading. A rule counting sessions
sees a full month; only a rule looking at return variation and volume sees an instrument that
is quoted but dormant.

**Where the damage actually lands.** The trailing covariance window is 252 sessions ending
before each evaluation month, so it reaches into 2023 for exactly one evaluation month
(2025-01), and then only by three sessions of late December -- months after the dormant
period ended. The risk objective is therefore effectively untouched. The exposure is in the
*training* observations: eight AMXB ticker-months in 2023 would otherwise enter the training
set with every momentum and ratio indicator collapsed to zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ..config import UniverseConfig

__all__ = [
    "StaleRun",
    "TickerMonthQuality",
    "find_stale_runs",
    "assess_ticker_months",
    "quality_report",
]


@dataclass(frozen=True, slots=True)
class StaleRun:
    """A maximal run of consecutive sessions at an unchanged closing price."""

    provider_symbol: str
    start: pd.Timestamp
    end: pd.Timestamp
    sessions: int
    price: float
    median_volume: float


@dataclass(frozen=True, slots=True)
class TickerMonthQuality:
    """Quality verdict for one ticker-month.

    ``usable`` gates entry into the feature matrix, and therefore into training and into
    universe membership. ``reasons`` is never empty when ``usable`` is False.
    """

    provider_symbol: str
    month: str
    sessions: int
    zero_return_fraction: float
    median_volume: float
    max_abs_return: float
    non_positive_prices: int
    usable: bool
    reasons: tuple[str, ...]


def find_stale_runs(
    prices: pd.Series, provider_symbol: str, *, min_sessions: int = 5
) -> list[StaleRun]:
    """Locate runs of consecutive sessions at an unchanged closing price.

    Args:
        prices: Closing prices indexed by date, ascending.
        provider_symbol: Symbol, carried into the returned records.
        min_sessions: Shortest run worth reporting. Runs of one or two sessions are
            unremarkable in any market and reporting them would bury the real defects.
    """
    prices = prices.dropna()
    if len(prices) < 2:
        return []
    changed = prices.diff() != 0
    group = changed.cumsum()
    runs: list[StaleRun] = []
    for _, index in prices.groupby(group).groups.items():
        index = pd.DatetimeIndex(index)
        # A group holds the session that set the price plus the sessions that repeated it.
        repeats = len(index) - 1
        if repeats >= min_sessions:
            runs.append(
                StaleRun(
                    provider_symbol=provider_symbol,
                    start=index.min(),
                    end=index.max(),
                    sessions=repeats,
                    price=float(prices.loc[index].iloc[0]),
                    median_volume=float("nan"),
                )
            )
    return sorted(runs, key=lambda r: -r.sessions)


def assess_ticker_months(
    frame: pd.DataFrame,
    provider_symbol: str,
    universe: UniverseConfig,
    *,
    min_trading_days: int,
) -> list[TickerMonthQuality]:
    """Judge each calendar month of one ticker's daily series.

    A month is rejected when it has too few sessions, when too large a fraction of its
    returns are exactly zero, when its median volume is below the floor, when any price is
    non-positive, or when a single-session return exceeds the plausibility bound.

    The zero-return and volume rules are the ones that matter: they are what separate an
    instrument that is quoted from an instrument that is traded.
    """
    if frame.empty or "close" not in frame:
        return []
    close = frame["close"].dropna()
    if close.empty:
        return []
    volume = frame.get("volume")
    volume = (
        pd.Series(0.0, index=close.index)
        if volume is None
        else volume.reindex(close.index).fillna(0.0)
    )
    returns = close.pct_change()

    results: list[TickerMonthQuality] = []
    for period, index in close.groupby(close.index.to_period("M")).groups.items():
        index = pd.DatetimeIndex(index)
        month_close = close.loc[index]
        month_returns = returns.loc[index].dropna()
        month_volume = volume.loc[index]

        sessions = len(month_close)
        zero_fraction = (
            float((month_returns == 0).mean()) if len(month_returns) else 1.0
        )
        median_volume = float(month_volume.median())
        max_abs_return = (
            float(month_returns.abs().max()) if len(month_returns) else 0.0
        )
        non_positive = int((month_close <= 0).sum())

        reasons: list[str] = []
        if sessions < min_trading_days:
            reasons.append(f"sessions={sessions}<{min_trading_days}")
        if zero_fraction > universe.max_zero_return_fraction:
            reasons.append(
                f"zero_return_fraction={zero_fraction:.2f}>"
                f"{universe.max_zero_return_fraction}"
            )
        if median_volume < universe.min_month_median_volume:
            reasons.append(
                f"median_volume={median_volume:.0f}<{universe.min_month_median_volume:.0f}"
            )
        if non_positive:
            reasons.append(f"non_positive_prices={non_positive}")
        if max_abs_return > universe.max_abs_daily_return:
            reasons.append(
                f"max_abs_daily_return={max_abs_return:.2f}>"
                f"{universe.max_abs_daily_return}"
            )

        results.append(
            TickerMonthQuality(
                provider_symbol=provider_symbol,
                month=str(period),
                sessions=sessions,
                zero_return_fraction=zero_fraction,
                median_volume=median_volume,
                max_abs_return=max_abs_return,
                non_positive_prices=non_positive,
                usable=not reasons,
                reasons=tuple(reasons),
            )
        )
    return results


def quality_report(
    assessments: Iterable[TickerMonthQuality],
    stale_runs: Iterable[StaleRun],
    directory: str | Path,
) -> tuple[Path, Path]:
    """Persist the per-ticker-month verdicts and the stale-run inventory.

    Both are written even when nothing was rejected: "we looked and found nothing" is a
    different claim from "we did not look", and the manuscript needs to be able to make the
    first one.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    months = pd.DataFrame(
        [
            {
                "provider_symbol": a.provider_symbol,
                "month": a.month,
                "sessions": a.sessions,
                "zero_return_fraction": a.zero_return_fraction,
                "median_volume": a.median_volume,
                "max_abs_return": a.max_abs_return,
                "non_positive_prices": a.non_positive_prices,
                "usable": a.usable,
                "reasons": ";".join(a.reasons),
            }
            for a in assessments
        ]
    )
    months_path = directory / "quality_ticker_months.csv"
    months.to_csv(months_path, index=False)

    runs = pd.DataFrame(
        [
            {
                "provider_symbol": r.provider_symbol,
                "start": r.start.date().isoformat(),
                "end": r.end.date().isoformat(),
                "sessions": r.sessions,
                "price": r.price,
            }
            for r in stale_runs
        ]
    )
    runs_path = directory / "quality_stale_runs.csv"
    runs.to_csv(runs_path, index=False)
    return months_path, runs_path
