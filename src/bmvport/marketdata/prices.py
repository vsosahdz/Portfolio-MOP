"""Cached daily price retrieval.

The price cache is the **dataset of record**. Every stage after this one reads only from
the cache, which is what makes the study reproducible against an upstream source that is
neither versioned nor reliable: retrieval emitted intermittent HTTP 401 responses during
development, and instrument symbols have changed since the original 2022 study (América
Móvil's L series was reclassified, so the 2022 list's ``AMXL`` no longer resolves).

Retrieval is resumable. A ticker already cached for the requested window is skipped, so an
interrupted download continues rather than restarting, and re-running the stage costs
nothing.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

__all__ = [
    "PRICE_COLUMNS",
    "RetrievalOutcome",
    "cache_path",
    "is_cached",
    "load_prices",
    "retrieve_prices",
    "write_retrieval_report",
]

# Columns retained from the provider. ``adj_close`` is what monthly returns use, so that
# splits and dividends are accounted for; ``close`` is what the backtest transacts at.
PRICE_COLUMNS = ("open", "high", "low", "close", "adj_close", "volume")

_PROVIDER_COLUMN_MAP = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
}

# Minimum rows for a cached series to be considered worth keeping. Below this the series
# cannot support a 60-day indicator warm-up plus any monthly aggregation at all.
MIN_USABLE_ROWS = 80


@dataclass(frozen=True, slots=True)
class RetrievalOutcome:
    """What happened for one ticker."""

    provider_symbol: str
    status: str  # "cached" | "downloaded" | "empty" | "too_short" | "error"
    rows: int
    first_date: str | None
    last_date: str | None
    detail: str = ""


def cache_path(cache_dir: str | Path, provider_symbol: str) -> Path:
    """Cache file for one ticker.

    Symbols contain characters that are awkward in filenames (``^MXX``, ``USDMXN=X``), so
    they are sanitised while keeping the mapping legible and collision-free.
    """
    safe = provider_symbol.replace("^", "_IDX_").replace("=", "_EQ_").replace("/", "_")
    return Path(cache_dir) / f"{safe}.parquet"


def is_cached(cache_dir: str | Path, provider_symbol: str, start: dt.date, end: dt.date) -> bool:
    """Whether the cache already covers the requested window for this ticker.

    Coverage is judged by the cached series' own extent, not by an exact date match: a
    ticker that stopped trading mid-window is fully cached even though its last row
    precedes ``end``. A sentinel written for unavailable tickers also counts as cached, so
    a known-empty symbol is not re-requested on every run.
    """
    path = cache_path(cache_dir, provider_symbol)
    if not path.exists():
        return False
    try:
        frame = pd.read_parquet(path)
    except Exception:  # noqa: BLE001 - a corrupt cache entry must be re-fetched
        return False
    if frame.empty:
        return True  # sentinel: provider had nothing for this symbol
    index = pd.DatetimeIndex(frame.index)
    # Allow a week of slack at the start: a late-listing instrument legitimately begins
    # after ``start`` and that is information, not a gap to re-request.
    return index.min().date() <= start + dt.timedelta(days=7)


def load_prices(cache_dir: str | Path, provider_symbol: str) -> pd.DataFrame:
    """Read one ticker's cached daily series.

    Raises:
        FileNotFoundError: if the ticker was never retrieved. Downstream stages must not
            silently treat an absent ticker as an empty one.
    """
    path = cache_path(cache_dir, provider_symbol)
    if not path.exists():
        raise FileNotFoundError(
            f"{provider_symbol} is not in the cache at {path}; run the prices stage first"
        )
    frame = pd.read_parquet(path)
    frame.index = pd.DatetimeIndex(frame.index, name="date")
    return frame


def _normalise(raw: pd.DataFrame) -> pd.DataFrame:
    """Reduce a provider frame to the canonical schema."""
    present = {src: dst for src, dst in _PROVIDER_COLUMN_MAP.items() if src in raw.columns}
    frame = raw[list(present)].rename(columns=present)
    for column in PRICE_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    frame = frame[list(PRICE_COLUMNS)].dropna(how="all")
    frame.index = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
    frame.index.name = "date"
    return frame[~frame.index.duplicated(keep="last")].sort_index()


def retrieve_prices(
    provider_symbols: Sequence[str],
    cache_dir: str | Path,
    start: dt.date,
    end: dt.date,
    *,
    batch_size: int = 40,
    progress: bool = True,
) -> list[RetrievalOutcome]:
    """Retrieve daily prices for ``provider_symbols`` into the cache.

    Args:
        provider_symbols: Symbols in the provider's convention.
        cache_dir: Cache directory; one parquet file per ticker.
        start: First date requested, inclusive.
        end: Last date requested, inclusive.
        batch_size: Symbols per provider request. Batching is a throughput tradeoff, not a
            correctness one.
        progress: Print per-batch progress. Retrieval of a full universe takes long enough
            that silence is indistinguishable from a hang.

    Returns:
        One outcome per requested symbol, including failures. Failures are reported rather
        than raised: an unavailable instrument is a fact about the universe, and the
        retrieval report is how that fact reaches the manuscript.
    """
    import yfinance as yf

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    outcomes: list[RetrievalOutcome] = []
    pending: list[str] = []
    for symbol in provider_symbols:
        if is_cached(cache_dir, symbol, start, end):
            try:
                cached = load_prices(cache_dir, symbol)
            except Exception:  # noqa: BLE001
                pending.append(symbol)
                continue
            outcomes.append(
                RetrievalOutcome(
                    provider_symbol=symbol,
                    status="cached",
                    rows=len(cached),
                    first_date=str(cached.index.min().date()) if len(cached) else None,
                    last_date=str(cached.index.max().date()) if len(cached) else None,
                )
            )
        else:
            pending.append(symbol)

    if progress:
        print(
            f"prices: {len(outcomes)} already cached, {len(pending)} to retrieve "
            f"({start} .. {end})",
            flush=True,
        )

    # The provider appends one day exclusive of ``end``.
    request_end = end + dt.timedelta(days=1)

    for offset in range(0, len(pending), batch_size):
        batch = pending[offset : offset + batch_size]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = yf.download(
                    batch,
                    start=start.isoformat(),
                    end=request_end.isoformat(),
                    auto_adjust=False,
                    progress=False,
                    threads=True,
                    group_by="ticker",
                )
        except Exception as exc:  # noqa: BLE001 - a failed batch must not abort the run
            for symbol in batch:
                outcomes.append(
                    RetrievalOutcome(symbol, "error", 0, None, None, f"batch: {exc}")
                )
            if progress:
                print(f"  batch {offset // batch_size + 1}: FAILED ({exc})", flush=True)
            continue

        for symbol in batch:
            try:
                sub = raw[symbol] if isinstance(raw.columns, pd.MultiIndex) else raw
                frame = _normalise(sub)
            except Exception as exc:  # noqa: BLE001
                outcomes.append(
                    RetrievalOutcome(symbol, "error", 0, None, None, str(exc)[:200])
                )
                continue

            if frame.empty:
                # Sentinel: record the absence so the symbol is not re-requested.
                frame.to_parquet(cache_path(cache_dir, symbol))
                outcomes.append(
                    RetrievalOutcome(symbol, "empty", 0, None, None, "provider returned no rows")
                )
                continue

            frame.to_parquet(cache_path(cache_dir, symbol))
            status = "downloaded" if len(frame) >= MIN_USABLE_ROWS else "too_short"
            outcomes.append(
                RetrievalOutcome(
                    provider_symbol=symbol,
                    status=status,
                    rows=len(frame),
                    first_date=str(frame.index.min().date()),
                    last_date=str(frame.index.max().date()),
                    detail="" if status == "downloaded" else f"only {len(frame)} rows",
                )
            )

        if progress:
            done = min(offset + batch_size, len(pending))
            ok = sum(1 for o in outcomes if o.status in ("downloaded", "cached"))
            print(
                f"  batch {offset // batch_size + 1}: {done}/{len(pending)} requested, "
                f"{ok} usable so far",
                flush=True,
            )

    return outcomes


def write_retrieval_report(
    outcomes: Iterable[RetrievalOutcome], directory: str | Path
) -> tuple[Path, Path]:
    """Write the per-ticker retrieval report and its summary."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / "retrieval_report.csv"
    json_path = directory / "retrieval_summary.json"

    outcomes = list(outcomes)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["provider_symbol", "status", "rows", "first_date", "last_date", "detail"]
        )
        for outcome in outcomes:
            writer.writerow(
                [
                    outcome.provider_symbol,
                    outcome.status,
                    outcome.rows,
                    outcome.first_date or "",
                    outcome.last_date or "",
                    outcome.detail,
                ]
            )

    counts: dict[str, int] = {}
    for outcome in outcomes:
        counts[outcome.status] = counts.get(outcome.status, 0) + 1
    summary = {
        "requested": len(outcomes),
        "by_status": counts,
        "usable": counts.get("downloaded", 0) + counts.get("cached", 0),
        "written_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return csv_path, json_path
