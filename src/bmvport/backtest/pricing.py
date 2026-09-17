"""Executable entry and exit pricing.

Positions are entered at the **close of the first trading session of the holding month** and
exited at the close of the last. The original study enters "on the month's first day using
the previous month's last close price", which nobody can transact at: it hands the strategy
a day of hindsight on every position, every month, and it does so in the direction that
flatters returns.

Two absences are handled explicitly rather than by imputation:

*No entry price.* The instrument is dropped and the remaining weights renormalised. Holding
something that could not be bought is not a portfolio.

*No exit price.* The last available close inside the month is used, and the substitution is
recorded. This is the conservative reading of an instrument that stopped trading mid-month;
carrying the entry price forward would silently report a zero return for a position that may
have been worth considerably less.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd

__all__ = ["PricedPortfolio", "price_portfolio", "month_bounds"]


@dataclass(frozen=True, slots=True)
class PricedPortfolio:
    """A portfolio valued over one holding month."""

    month: str
    symbols: tuple[str, ...]
    weights: np.ndarray
    entry_prices: np.ndarray
    exit_prices: np.ndarray
    gross_return: float
    daily_returns: pd.Series
    dropped_at_entry: tuple[str, ...] = ()
    early_exits: tuple[str, ...] = ()
    notes: Mapping[str, str] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.symbols)


def month_bounds(index: pd.DatetimeIndex, month: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """First and last trading session inside ``month``.

    Raises:
        ValueError: if the calendar holds no session in that month.
    """
    sessions = index[index.to_period("M").astype(str) == month]
    if len(sessions) == 0:
        raise ValueError(f"no trading sessions in {month}")
    return sessions.min(), sessions.max()


def price_portfolio(
    month: str,
    symbols: list[str],
    weights: np.ndarray,
    prices: Mapping[str, pd.Series],
    calendar: pd.DatetimeIndex,
) -> PricedPortfolio:
    """Value a weighted portfolio over one holding month.

    Args:
        month: Holding month as ``YYYY-MM``.
        symbols: Held instruments, aligned with ``weights``.
        weights: Entry weights, summing to one.
        prices: Daily closing prices per symbol.
        calendar: Reference trading calendar.

    Returns:
        The priced portfolio, including its daily return series at fixed entry weights --
        which is what realised risk is computed from.

    Raises:
        ValueError: if no instrument survives entry pricing.
    """
    first, last = month_bounds(calendar, month)
    sessions = calendar[(calendar >= first) & (calendar <= last)]

    kept, kept_weights, entries, exits = [], [], [], []
    dropped, early = [], []
    for symbol, weight in zip(symbols, np.asarray(weights, dtype=float)):
        series = prices.get(symbol)
        if series is None:
            dropped.append(symbol)
            continue
        window = series[(series.index >= first) & (series.index <= last)].dropna()
        if window.empty or first not in window.index:
            # No price on the first session: the position could not have been opened.
            dropped.append(symbol)
            continue
        if last not in window.index:
            early.append(symbol)
        kept.append(symbol)
        kept_weights.append(weight)
        entries.append(float(window.loc[first]))
        exits.append(float(window.iloc[-1]))

    if not kept:
        raise ValueError(f"no instrument in the {month} portfolio could be priced at entry")

    w = np.asarray(kept_weights, dtype=float)
    total = w.sum()
    # Renormalise after drops so the budget constraint survives the repair.
    w = w / total if total > 1e-12 else np.full(len(w), 1.0 / len(w))
    entry = np.asarray(entries, dtype=float)
    exit_ = np.asarray(exits, dtype=float)
    gross = float(np.sum(w * (exit_ / entry - 1.0)))

    # Daily portfolio returns at fixed entry weights: the basis for realised risk.
    frame = pd.DataFrame(
        {s: prices[s].reindex(sessions) for s in kept}, index=sessions
    ).ffill()
    daily = frame.pct_change().fillna(0.0) @ w

    return PricedPortfolio(
        month=month,
        symbols=tuple(kept),
        weights=w,
        entry_prices=entry,
        exit_prices=exit_,
        gross_return=gross,
        daily_returns=daily,
        dropped_at_entry=tuple(dropped),
        early_exits=tuple(early),
        notes={
            "entry_session": str(first.date()),
            "exit_session": str(last.date()),
            "renormalised_after_drops": str(bool(dropped)),
        },
    )
