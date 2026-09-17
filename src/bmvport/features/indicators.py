"""The 43 daily technical indicators of the thesis's Table A.1.

Written out rather than delegated to a technical-analysis library, for two reasons. Library
conventions differ on exactly the indicators this study uses -- most implementations apply
Wilder's smoothing to the RSI, while Table A.1 specifies a plain arithmetic mean -- and a
silent convention mismatch would make the published feature description wrong in a way
nobody could detect from the results. And the table itself is a lossy PDF conversion whose
ambiguities have to be resolved deliberately.

**Two ambiguities in the source, resolved here and recorded in the feature dictionary.**

*Price loss.* Table A.1 defines ``loss_t = change_t`` when ``change_t < 0``, i.e. a signed
quantity. Taken literally, ``avg_loss`` is then negative, ``RS = avg_gain / avg_loss`` is
negative, and ``RSI = 100 - 100/(1 + RS)`` diverges as ``RS`` approaches -1. The standard
definition uses the magnitude of the loss, and that is what is implemented; the signed
reading is not a variant, it is broken arithmetic.

*Stochastic oscillator.* The table's denominator renders as ``high_t - min(low over n)``
rather than the conventional ``max(high over n) - min(low over n)``. The conventional form
is implemented. A referee checking the formula against the literature will expect it, and a
bespoke variant would need a justification the thesis never offers -- the rendering is more
plausibly an artefact of the conversion than a deliberate departure.

Everything else follows the table exactly, including the plain-mean RSI.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import FeatureConfig

__all__ = [
    "IndicatorSpec",
    "indicator_specs",
    "daily_indicators",
    "REQUIRED_PRICE_COLUMNS",
]

REQUIRED_PRICE_COLUMNS = ("open", "high", "low", "close", "volume")


@dataclass(frozen=True, slots=True)
class IndicatorSpec:
    """One daily indicator column, for the versioned feature dictionary."""

    name: str
    family: str
    window: int | None
    formula: str
    note: str = ""


def indicator_specs(config: FeatureConfig) -> tuple[IndicatorSpec, ...]:
    """Describe all 43 daily columns, in the order they are produced."""
    specs: list[IndicatorSpec] = [
        IndicatorSpec("return", "return", 1, "close_t / close_{t-1} - 1"),
        IndicatorSpec("change", "return", 1, "close_t - close_{t-1}"),
        IndicatorSpec("gain", "rsi", 1, "change_t if change_t >= 0 else 0"),
        IndicatorSpec(
            "loss",
            "rsi",
            1,
            "|change_t| if change_t < 0 else 0",
            "Table A.1 writes the signed change; the magnitude is used, since a signed "
            "average makes RS negative and RSI divergent",
        ),
        IndicatorSpec("avg_gain", "rsi", config.rsi_window, "arithmetic mean of gain over n"),
        IndicatorSpec("avg_loss", "rsi", config.rsi_window, "arithmetic mean of loss over n"),
        IndicatorSpec(
            "rs",
            "rsi",
            config.rsi_window,
            "avg_gain / avg_loss",
            "infinite where avg_loss is zero, which maps to RSI = 100",
        ),
        IndicatorSpec(
            "rsi",
            "rsi",
            config.rsi_window,
            "100 - 100 / (1 + RS)",
            "plain arithmetic mean as specified, NOT Wilder smoothing",
        ),
    ]
    specs += [
        IndicatorSpec(f"ma_{n}", "moving_average", n, "mean of close over n")
        for n in config.ma_windows
    ]
    specs.append(IndicatorSpec("tp", "money_flow", 1, "(high + low + close) / 3"))
    specs += [
        IndicatorSpec("pmf", "money_flow", 1, "tp * volume if tp_t >= tp_{t-1} else 0"),
        IndicatorSpec("nmf", "money_flow", 1, "tp * volume if tp_t < tp_{t-1} else 0"),
        IndicatorSpec(
            "mfi",
            "money_flow",
            config.mfi_window,
            "100 - 100 / (1 + sum(pmf, n) / sum(nmf, n))",
            "maps to 100 where the negative money flow over the window is zero",
        ),
    ]
    specs.append(
        IndicatorSpec(
            "so",
            "stochastic",
            config.stochastic_window,
            "(close_t - min(low, n)) / (max(high, n) - min(low, n)) * 100",
            "Table A.1 renders the denominator as high_t - min(low, n); the conventional "
            "max(high, n) is used",
        )
    )
    for family, formula in (
        ("chr", "ln(close_t / high_{t-n})"),
        ("hor", "ln(high_t / open_{t-n})"),
        ("lor", "ln(low_t / open_{t-n})"),
        ("hlr", "ln(high_t / low_{t-n})"),
        ("ccr", "ln(close_t / close_{t-n})"),
    ):
        specs += [
            IndicatorSpec(f"{family}_{n}", family, n, formula) for n in config.ratio_windows
        ]
    return tuple(specs)


def _safe_log_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Log ratio that yields NaN rather than -inf on non-positive inputs.

    A non-positive price is a data defect, and the quality gates reject the month it falls
    in. Producing an infinity here would let it propagate into a monthly mean and quietly
    destroy the aggregate instead.
    """
    ratio = numerator / denominator
    ratio = ratio.where(np.isfinite(ratio) & (ratio > 0))
    return np.log(ratio)


def daily_indicators(frame: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """Compute the 43 daily indicator columns for one instrument.

    Rows whose trailing window is incomplete carry NaN for the affected indicators; the
    monthly stage drops dates that are not complete across all of them, so no indicator is
    ever computed from a partial window.

    Args:
        frame: Daily prices indexed by date, ascending, with the columns of
            :data:`REQUIRED_PRICE_COLUMNS`.
        config: Feature settings supplying the windows.

    Returns:
        A frame indexed like ``frame`` with exactly
        ``config.expected_daily_indicators`` columns.

    Raises:
        ValueError: if a required price column is missing, or if the produced column count
            disagrees with the configured invariant.
    """
    missing = [c for c in REQUIRED_PRICE_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"price frame is missing required columns: {missing}")

    close = frame["close"].astype(float)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    open_ = frame["open"].astype(float)
    volume = frame["volume"].astype(float).fillna(0.0)

    out: dict[str, pd.Series] = {}

    # --- return family -----------------------------------------------------------------
    out["return"] = close.pct_change()
    change = close.diff()
    out["change"] = change

    # --- RSI ---------------------------------------------------------------------------
    gain = change.where(change >= 0, 0.0)
    # Magnitude, not the signed change: see the module docstring.
    loss = (-change).where(change < 0, 0.0)
    out["gain"] = gain
    out["loss"] = loss
    n_rsi = config.rsi_window
    avg_gain = gain.rolling(n_rsi, min_periods=n_rsi).mean()
    avg_loss = loss.rolling(n_rsi, min_periods=n_rsi).mean()
    out["avg_gain"] = avg_gain
    out["avg_loss"] = avg_loss
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
    # A window with no down days makes RS infinite. That is arithmetically right and
    # useless as a feature: one infinity destroys the month's mean and standard deviation.
    # It is recorded as missing, and no information is lost -- RSI carries the same state as
    # its limiting value of 100. Observed rate on the real cache is roughly 0.02% of
    # sessions.
    rs = rs.replace([np.inf, -np.inf], np.nan)
    out["rs"] = rs
    # avg_loss == 0 means no down days in the window: RSI is 100 by definition, and the
    # limit of the formula agrees. Computing it directly avoids an inf/inf indeterminate.
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi = rsi.where(avg_loss != 0, 100.0)
    rsi = rsi.where(avg_gain.notna() & avg_loss.notna())
    out["rsi"] = rsi

    # --- moving averages ---------------------------------------------------------------
    for n in config.ma_windows:
        out[f"ma_{n}"] = close.rolling(n, min_periods=n).mean()

    # --- money flow --------------------------------------------------------------------
    tp = (high + low + close) / 3.0
    out["tp"] = tp
    raw_flow = tp * volume
    rising = tp >= tp.shift(1)
    pmf = raw_flow.where(rising, 0.0)
    nmf = raw_flow.where(~rising, 0.0)
    # The first row has no predecessor, so neither flow is defined there.
    pmf.iloc[:1] = np.nan
    nmf.iloc[:1] = np.nan
    out["pmf"] = pmf
    out["nmf"] = nmf
    n_mfi = config.mfi_window
    pmf_sum = pmf.rolling(n_mfi, min_periods=n_mfi).sum()
    nmf_sum = nmf.rolling(n_mfi, min_periods=n_mfi).sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        money_ratio = pmf_sum / nmf_sum
    mfi = 100.0 - 100.0 / (1.0 + money_ratio)
    mfi = mfi.where(nmf_sum != 0, 100.0)
    mfi = mfi.where(pmf_sum.notna() & nmf_sum.notna())
    out["mfi"] = mfi

    # --- stochastic oscillator ---------------------------------------------------------
    n_so = config.stochastic_window
    lowest = low.rolling(n_so, min_periods=n_so).min()
    highest = high.rolling(n_so, min_periods=n_so).max()
    span = highest - lowest
    so = (close - lowest) / span * 100.0
    # A flat window has no range; the oscillator is undefined rather than infinite.
    out["so"] = so.where(span > 0)

    # --- log ratio families ------------------------------------------------------------
    for n in config.ratio_windows:
        out[f"chr_{n}"] = _safe_log_ratio(close, high.shift(n))
        out[f"hor_{n}"] = _safe_log_ratio(high, open_.shift(n))
        out[f"lor_{n}"] = _safe_log_ratio(low, open_.shift(n))
        out[f"hlr_{n}"] = _safe_log_ratio(high, low.shift(n))
        out[f"ccr_{n}"] = _safe_log_ratio(close, close.shift(n))

    result = pd.DataFrame(out, index=frame.index)
    ordered = [spec.name for spec in indicator_specs(config)]
    result = result[ordered]

    if result.shape[1] != config.expected_daily_indicators:
        raise ValueError(
            f"produced {result.shape[1]} daily indicator columns but the configuration "
            f"declares {config.expected_daily_indicators}; the feature description in the "
            "manuscript and the pipeline have diverged"
        )
    return result
