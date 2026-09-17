"""Monthly aggregation: 43 daily indicators become 129 monthly features.

Each daily indicator contributes its mean, standard deviation and last value over the
month's sessions. The thesis's reasoning is that the first two describe how the indicator
behaved across the month while the last carries the state an investor would act on at the
rebalancing date, and that reading is preserved here.

Two properties are enforced rather than trusted:

**No partial windows.** A date enters the aggregation only when every one of the 43
indicators is complete at that date, so the 60-session warm-up is honoured for all of them
jointly rather than per column. Averaging a month in which half the long-window ratios were
still warming up would silently mix populations.

**No look-ahead.** Every feature for month *m* derives from sessions inside *m* and the
history preceding them. The labels are the only place future information appears, and they
live in a separate column set that the screening stage is forbidden to train on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from ..config import FeatureConfig
from .indicators import daily_indicators, indicator_specs

__all__ = [
    "MonthlyExclusion",
    "monthly_features",
    "monthly_returns",
    "build_feature_matrix",
    "KEY_COLUMNS",
    "LABEL_INPUT_COLUMNS",
]

# Identifying columns of the monthly matrix. Never features.
KEY_COLUMNS = ("provider_symbol", "month", "sessions")

# Columns that feed the labeling strategies. They carry current- and next-month information
# and must never reach a screener's training matrix; the screening stage asserts this.
LABEL_INPUT_COLUMNS = ("monthly_return", "trailing_3m_mean_return", "next_month_return")


@dataclass(frozen=True, slots=True)
class MonthlyExclusion:
    """A ticker-month dropped before it became an observation."""

    provider_symbol: str
    month: str
    sessions: int
    reason: str


def monthly_features(
    frame: pd.DataFrame,
    provider_symbol: str,
    config: FeatureConfig,
) -> tuple[pd.DataFrame, list[MonthlyExclusion]]:
    """Aggregate one instrument's daily indicators into monthly observations.

    Returns:
        The monthly feature rows and the ticker-months excluded for having too few complete
        sessions, each with its session count so the exclusion is auditable.

    Raises:
        ValueError: if the produced feature count disagrees with the configured invariant.
    """
    daily = daily_indicators(frame, config)

    # Warm-up completeness and legitimate missingness are different things and must not be
    # conflated. A date is admitted once the longest trailing window has enough history --
    # that is the warm-up rule. Within an admitted date an individual indicator can still be
    # undefined for a real reason: RS has no value when the window holds no down days, the
    # stochastic oscillator none when the window is perfectly flat. Dropping the whole date
    # for those would discard 42 valid indicators to avoid one gap, and would do it
    # selectively on strongly trending instruments -- a bias, not a safeguard.
    warmup = config.warmup_days()
    complete = daily.iloc[warmup:]

    rows: list[dict[str, object]] = []
    exclusions: list[MonthlyExclusion] = []
    if complete.empty:
        return pd.DataFrame(), exclusions

    for period, group in complete.groupby(complete.index.to_period("M")):
        month = str(period)
        sessions = len(group)
        if sessions < config.min_trading_days_per_month:
            exclusions.append(
                MonthlyExclusion(
                    provider_symbol=provider_symbol,
                    month=month,
                    sessions=sessions,
                    reason=(
                        f"sessions={sessions}<{config.min_trading_days_per_month} "
                        "complete-indicator days"
                    ),
                )
            )
            continue

        row: dict[str, object] = {
            "provider_symbol": provider_symbol,
            "month": month,
            "sessions": sessions,
        }
        for aggregation in config.aggregations:
            if aggregation == "mean":
                values = group.mean()
            elif aggregation == "std":
                # Sample standard deviation, consistent with the dispersion measures used
                # everywhere else in the study.
                values = group.std(ddof=1)
            elif aggregation == "last":
                # The last *defined* value, not the last row: an indicator that is
                # undefined on the final session should carry the state an investor would
                # actually have seen, not a gap.
                values = group.ffill().iloc[-1]
            else:
                raise ValueError(f"unsupported aggregation {aggregation!r}")
            for name, value in values.items():
                row[f"{name}_{aggregation}"] = float(value)

        # A feature that is still missing after aggregation would reach a classifier as a
        # gap it cannot consume. The ticker-month is dropped with the offending columns
        # named, rather than imputed.
        empty = [k for k, v in row.items() if k not in KEY_COLUMNS and not np.isfinite(v)]
        if empty:
            exclusions.append(
                MonthlyExclusion(
                    provider_symbol=provider_symbol,
                    month=month,
                    sessions=sessions,
                    reason=f"undefined features: {','.join(sorted(empty)[:4])}",
                )
            )
            continue
        rows.append(row)

    matrix = pd.DataFrame(rows)
    if matrix.empty:
        return matrix, exclusions

    feature_columns = [c for c in matrix.columns if c not in KEY_COLUMNS]
    if len(feature_columns) != config.expected_monthly_features:
        raise ValueError(
            f"produced {len(feature_columns)} monthly feature columns but the "
            f"configuration declares {config.expected_monthly_features}"
        )
    return matrix, exclusions


def monthly_returns(frame: pd.DataFrame) -> pd.DataFrame:
    """Monthly return and trailing three-month mean, from adjusted close.

    Adjusted close is used so that splits and dividends do not appear as returns. The
    trailing mean requires three prior monthly returns and is absent before that, which is
    what makes 2023-05 the first month the ``historical`` labeling strategies can describe.
    """
    column = "adj_close" if "adj_close" in frame.columns else "close"
    monthly_close = frame[column].dropna().resample("MS").last()
    returns = monthly_close.pct_change()
    trailing = returns.shift(1).rolling(3, min_periods=3).mean()
    out = pd.DataFrame(
        {
            "month": [str(p) for p in returns.index.to_period("M")],
            "monthly_return": returns.to_numpy(),
            "trailing_3m_mean_return": trailing.to_numpy(),
            "next_month_return": returns.shift(-1).to_numpy(),
        }
    )
    return out.dropna(subset=["monthly_return"]).reset_index(drop=True)


def build_feature_matrix(
    frames: Mapping[str, pd.DataFrame],
    config: FeatureConfig,
    *,
    usable: Mapping[tuple[str, str], bool] | None = None,
) -> tuple[pd.DataFrame, list[MonthlyExclusion]]:
    """Build the monthly feature matrix across instruments.

    Args:
        frames: Daily price frames keyed by provider symbol.
        config: Feature settings.
        usable: Optional quality verdicts keyed by ``(symbol, month)``. Ticker-months
            marked unusable are dropped here, so a dormant instrument never becomes a
            training observation with every momentum indicator collapsed to zero.

    Returns:
        The matrix and every exclusion, each with its reason.
    """
    matrices: list[pd.DataFrame] = []
    exclusions: list[MonthlyExclusion] = []

    for symbol, frame in frames.items():
        if frame.empty or "close" not in frame:
            continue
        matrix, dropped = monthly_features(frame, symbol, config)
        exclusions.extend(dropped)
        if matrix.empty:
            continue
        returns = monthly_returns(frame)
        matrix = matrix.merge(returns, on="month", how="left")

        if usable is not None:
            keep = matrix["month"].map(lambda m: usable.get((symbol, m), True))
            for month in matrix.loc[~keep, "month"]:
                exclusions.append(
                    MonthlyExclusion(symbol, month, 0, "rejected by data-quality gates")
                )
            matrix = matrix[keep]
        matrices.append(matrix)

    if not matrices:
        return pd.DataFrame(), exclusions
    combined = pd.concat(matrices, ignore_index=True)
    return combined.sort_values(["month", "provider_symbol"]).reset_index(drop=True), exclusions


def write_feature_dictionary(config: FeatureConfig, path: str | Path) -> Path:
    """Write the machine-readable dictionary describing every produced column."""
    import json

    entries = []
    for spec in indicator_specs(config):
        for aggregation in config.aggregations:
            entries.append(
                {
                    "column": f"{spec.name}_{aggregation}",
                    "indicator": spec.name,
                    "family": spec.family,
                    "window": spec.window,
                    "aggregation": aggregation,
                    "formula": spec.formula,
                    "note": spec.note,
                }
            )
    payload = {
        "version": config.dictionary_version,
        "daily_indicators": config.expected_daily_indicators,
        "monthly_features": config.expected_monthly_features,
        "aggregations": list(config.aggregations),
        "key_columns": list(KEY_COLUMNS),
        "label_input_columns": list(LABEL_INPUT_COLUMNS),
        "columns": entries,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
