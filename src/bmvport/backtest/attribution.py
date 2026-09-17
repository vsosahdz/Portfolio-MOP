"""Return attribution: which part of the result was the method, and which was the peso.

A headline return on this universe cannot be read as method skill without a decomposition,
because the universe mixes peso-denominated domestic issuers with SIC instruments whose
peso return is a foreign return plus a currency move. The original study reports a 70%
annualised figure over a window in which its best portfolio held 97% in a single US name
while the exchange rate moved substantially, and offers no way to separate the two. That is
not a small presentational gap: it is the difference between a claim about a method and a
claim about a currency.

**The identity is exact by construction.**

```
    r_portfolio  =  market  +  selection  +  allocation

    market      the equal-weight return of the month's admitted universe -- the passive
                alternative that required no screening and no optimisation at all
    selection   equal-weight return of the screened names, minus the universe's
                -- what choosing those instruments was worth, before any weighting
    allocation  the portfolio's own return, minus the equal-weight return of the same names
                -- what the weighting was worth, given the selection
```

Each term is a difference against the previous stage, so the three sum to the realised gross
return with no residual. Attribution schemes that leave an unexplained remainder invite the
suspicion that the remainder is where the result actually lives.

**Currency is carved out of the active return rather than added as a fourth term.** The FX
tilt is the portfolio's SIC weight in excess of the universe's, multiplied by the month's
peso move: the part of ``selection + allocation`` that came from being more exposed to the
dollar than the passive alternative was. Adding it as a separate additive component would
double-count, since that exposure is already inside the instruments' peso returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

__all__ = ["Attribution", "attribute_return"]

# Tolerance for the identity check. Wider than machine epsilon because the inputs are
# themselves aggregates of floating-point returns.
RECONCILIATION_TOLERANCE = 1e-9


@dataclass(frozen=True, slots=True)
class Attribution:
    """Decomposition of one portfolio-month's gross return."""

    month: str
    gross_return: float
    market: float
    selection: float
    allocation: float
    fx_tilt: float
    portfolio_sic_weight: float
    universe_sic_share: float
    fx_return: float
    residual: float

    @property
    def active_return(self) -> float:
        """Return in excess of the passive equal-weight universe."""
        return self.selection + self.allocation

    def reconciles(self, tolerance: float = RECONCILIATION_TOLERANCE) -> bool:
        return abs(self.residual) <= tolerance


def attribute_return(
    month: str,
    *,
    gross_return: float,
    portfolio_weights: Mapping[str, float],
    selected_returns: Mapping[str, float],
    universe_returns: Mapping[str, float],
    sic_symbols: Sequence[str],
    fx_return: float,
) -> Attribution:
    """Decompose a realised gross return into market, selection and allocation.

    Args:
        month: Holding month.
        gross_return: The portfolio's realised gross return.
        portfolio_weights: Entry weights of the held instruments.
        selected_returns: Realised monthly return of each screened instrument.
        universe_returns: Realised monthly return of each admitted universe member.
        sic_symbols: Instruments classified as SIC, for the currency carve-out.
        fx_return: The month's MXN/USD return, from Banxico's FIX.

    Returns:
        The decomposition, whose ``residual`` is zero up to floating point.

    Raises:
        ValueError: if the universe is empty; there would be no passive alternative to
            measure against.
    """
    if not universe_returns:
        raise ValueError(f"no universe returns for {month}; attribution has no baseline")

    universe_equal_weight = float(np.mean(list(universe_returns.values())))
    selected_equal_weight = (
        float(np.mean(list(selected_returns.values()))) if selected_returns
        else universe_equal_weight
    )

    market = universe_equal_weight
    selection = selected_equal_weight - universe_equal_weight
    allocation = gross_return - selected_equal_weight
    residual = gross_return - (market + selection + allocation)

    sic = set(sic_symbols)
    portfolio_sic = float(sum(w for s, w in portfolio_weights.items() if s in sic))
    universe_sic = (
        float(sum(1 for s in universe_returns if s in sic) / len(universe_returns))
    )
    # The currency component of the active return: exposure taken beyond the passive one.
    fx_tilt = (portfolio_sic - universe_sic) * float(fx_return)

    return Attribution(
        month=month,
        gross_return=float(gross_return),
        market=market,
        selection=selection,
        allocation=allocation,
        fx_tilt=fx_tilt,
        portfolio_sic_weight=portfolio_sic,
        universe_sic_share=universe_sic,
        fx_return=float(fx_return),
        residual=float(residual),
    )
