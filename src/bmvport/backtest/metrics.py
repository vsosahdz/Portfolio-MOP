"""Turnover, transaction costs, and the pinned performance metrics.

**Costs are the headline, not a footnote.** A strategy that rebalances its entire portfolio
every month pays for the privilege, and the original study's zero-cost assumption is exactly
the kind of omission that makes a backtest unfalsifiable. Net-of-cost results under several
round-trip scenarios are what the manuscript reports; the zero-cost case appears only as an
explicit upper bound.

Costs are applied post-hoc, so the optimiser never knew about them. That makes the net
figures **pessimistic** for the genetic arms relative to a cost-aware formulation, and the
manuscript says so rather than presenting the number as the last word. Turnover is reported
alongside, so a reader can see where the drag comes from.

**Ex-ante and realised risk never share a name.** The optimiser minimises an estimate from
trailing returns; a held portfolio realises a dispersion during the month. The original uses
one word for both, which makes its risk tables impossible to interpret. Here they are
``sigma_ex_ante`` and ``sigma_realized`` and they are stored in separate fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "PortfolioMetrics",
    "turnover",
    "apply_costs",
    "realised_risk",
    "sharpe_ratio",
    "evaluate_month",
]


@dataclass(frozen=True, slots=True)
class PortfolioMetrics:
    """Everything measured for one portfolio-month, stored at monthly frequency."""

    month: str
    gross_return: float
    turnover: float
    net_returns: Mapping[float, float]
    sigma_realized: float
    sigma_ex_ante: float
    sharpe: Mapping[float, float]
    risk_free: float
    holdings: int


def turnover(
    current: Mapping[str, float], previous: Mapping[str, float] | None
) -> float:
    """One-way turnover between consecutive months.

    Defined as half the sum of absolute weight changes, which is the fraction of the
    portfolio that had to be traded. The first month is full turnover from cash: the
    positions had to be opened, and pretending otherwise would hand the strategy a free
    entry.
    """
    if previous is None:
        return 1.0
    symbols = set(current) | set(previous)
    delta = sum(abs(current.get(s, 0.0) - previous.get(s, 0.0)) for s in symbols)
    return float(delta / 2.0)


def apply_costs(
    gross: float, turnover_value: float, cost_scenarios_bps: Sequence[float]
) -> dict[float, float]:
    """Net return under each round-trip cost scenario.

    A round trip is charged on the traded fraction: entering and leaving a position both
    cost, so the charge is ``2 x turnover x bps``.
    """
    out: dict[float, float] = {}
    for bps in cost_scenarios_bps:
        out[float(bps)] = float(gross - 2.0 * turnover_value * (bps / 10_000.0))
    return out


def realised_risk(daily_returns: pd.Series) -> float:
    """Standard deviation of the portfolio's daily returns during the holding month.

    Sample standard deviation, at the weights fixed on entry. Returns ``nan`` for a month
    with fewer than two sessions rather than a spurious zero.
    """
    values = pd.Series(daily_returns).dropna()
    if len(values) < 2:
        return float("nan")
    return float(values.std(ddof=1))


def sharpe_ratio(net_return: float, sigma: float, risk_free: float) -> float:
    """Monthly Sharpe ratio: excess return over realised dispersion.

    Undefined rather than infinite when the portfolio did not move: a constant portfolio has
    no risk-adjusted performance to report, and an infinity would dominate every aggregate
    it entered.
    """
    if not np.isfinite(sigma) or sigma <= 0:
        return float("nan")
    return float((net_return - risk_free) / sigma)


def evaluate_month(
    month: str,
    gross_return: float,
    daily_returns: pd.Series,
    current_weights: Mapping[str, float],
    previous_weights: Mapping[str, float] | None,
    *,
    sigma_ex_ante: float,
    risk_free: float,
    cost_scenarios_bps: Sequence[float],
) -> PortfolioMetrics:
    """Assemble every monthly metric for one portfolio."""
    turn = turnover(current_weights, previous_weights)
    net = apply_costs(gross_return, turn, cost_scenarios_bps)
    sigma = realised_risk(daily_returns)
    return PortfolioMetrics(
        month=month,
        gross_return=float(gross_return),
        turnover=turn,
        net_returns=net,
        sigma_realized=sigma,
        sigma_ex_ante=float(sigma_ex_ante),
        sharpe={bps: sharpe_ratio(r, sigma, risk_free) for bps, r in net.items()},
        risk_free=float(risk_free),
        holdings=len(current_weights),
    )
