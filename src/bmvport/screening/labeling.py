"""The three binary labeling strategies, and the components they are built from.

Each strategy is a predicate over a ticker-month, using the month's own return ``r_t``, the
mean of the three preceding monthly returns ``MA3_t``, and the following month's return
``r_{t+1}``:

===================  =========================================
``historical``       ``r_t > 0`` and ``MA3_t > 0``
``t_plus_1``         ``r_{t+1} > 0``
``historical_and_t1``  both of the above
===================  =========================================

**Why the components are exposed and not just the labels.** Measured on this study's feature
matrix, a classifier reaches AUC 0.979 on ``historical`` and 0.470 on ``t_plus_1`` -- the
first because the monthly features contain the month's own return statistics, so predicting a
condition on that same month is close to evaluating an identity, and the second because there
is no exploitable directional signal. The conjunction inherits the first component's
inflation and scores 0.882, which would read as excellent screening and would be
meaningless.

Any metric computed on a label containing ``historical`` therefore has to be reported
alongside the same metric on ``t_plus_1`` alone. Only the latter speaks to predictive skill.
:func:`label_components` exists so that decomposition is always available and cannot be
forgotten.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

__all__ = [
    "LABELING_STRATEGIES",
    "FORWARD_COMPONENT",
    "LabelBalance",
    "label_components",
    "apply_labeling",
    "class_balance",
    "requires_forward_decomposition",
]

LABELING_STRATEGIES = ("historical", "t_plus_1", "historical_and_t1")

# The component that actually looks forward. Metrics on any label containing a
# present-describable conjunct are reported against this as well.
FORWARD_COMPONENT = "t_plus_1"

# Columns the labels are computed from. None may ever enter a screener's feature matrix.
_REQUIRED = ("monthly_return", "trailing_3m_mean_return", "next_month_return")


@dataclass(frozen=True, slots=True)
class LabelBalance:
    """Positive-class share for one labeling strategy in one month."""

    strategy: str
    month: str
    observations: int
    positives: int

    @property
    def positive_rate(self) -> float:
        return self.positives / self.observations if self.observations else float("nan")

    @property
    def imbalance_ratio(self) -> float:
        """Negatives per positive. Infinite when a month has no positives at all."""
        if self.positives == 0:
            return float("inf")
        return (self.observations - self.positives) / self.positives


def requires_forward_decomposition(strategy: str) -> bool:
    """Whether metrics on this label must be reported against ``t_plus_1`` as well.

    True for every strategy carrying a condition on the current month, because those are
    partly determined by the features and inflate any classification metric computed on
    them.
    """
    return strategy in ("historical", "historical_and_t1")


def label_components(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the boolean components each labeling strategy is assembled from.

    Raises:
        ValueError: if a required return column is absent. Silently producing a label from
            missing inputs would place unlabelable ticker-months in the negative class.
    """
    missing = [c for c in _REQUIRED if c not in frame.columns]
    if missing:
        raise ValueError(f"cannot build labels, missing columns: {missing}")
    return pd.DataFrame(
        {
            "current_month_positive": frame["monthly_return"] > 0,
            "trailing_mean_positive": frame["trailing_3m_mean_return"] > 0,
            "next_month_positive": frame["next_month_return"] > 0,
        },
        index=frame.index,
    )


def apply_labeling(frame: pd.DataFrame, strategy: str) -> pd.Series:
    """Compute one strategy's label over a monthly feature frame.

    Rows missing any required input are excluded rather than defaulted: an unlabelable
    ticker-month is not a negative example, and treating it as one would quietly bias every
    classifier towards the majority class.

    Returns:
        A boolean series indexed like ``frame``, with rows lacking inputs dropped.
    """
    if strategy not in LABELING_STRATEGIES:
        raise ValueError(f"unknown labeling strategy {strategy!r}")
    usable = frame.dropna(subset=list(_REQUIRED))
    components = label_components(usable)

    if strategy == "historical":
        labels = components["current_month_positive"] & components["trailing_mean_positive"]
    elif strategy == "t_plus_1":
        labels = components["next_month_positive"]
    else:
        labels = (
            components["current_month_positive"]
            & components["trailing_mean_positive"]
            & components["next_month_positive"]
        )
    return labels.rename(strategy)


def class_balance(
    frame: pd.DataFrame, strategy: str, *, month_column: str = "month"
) -> list[LabelBalance]:
    """Positive-class share per month, recorded for the manuscript.

    The original study reported 51% / 52% / 25% for the three strategies on its window.
    Whatever this window produces is a property of the period and is reported as measured,
    not compared against those figures.
    """
    labels = apply_labeling(frame, strategy)
    months = frame.loc[labels.index, month_column]
    out: list[LabelBalance] = []
    for month, group in labels.groupby(months):
        out.append(
            LabelBalance(
                strategy=strategy,
                month=str(month),
                observations=int(group.size),
                positives=int(group.sum()),
            )
        )
    return sorted(out, key=lambda b: b.month)
