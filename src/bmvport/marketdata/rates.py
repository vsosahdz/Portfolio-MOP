"""Banxico SIE series: risk-free rate and the official exchange rate.

Two series come from Banxico rather than the equity price provider, because for a paper
about the Mexican market these are the canonical sources:

- **CETES 28 days** (`SF43936`) is the risk-free rate the Sharpe ratio is measured against
  and the Tier 0 comparator floor.
- **FIX** (`SF43718`) is Banxico's official MXN/USD reference rate, used for the foreign
  exchange component of return attribution and to express the S&P 500 in pesos.

Everything below was verified against the live API rather than assumed:

``SF43936`` is titled "Valores gubernamentales — Resultados de la subasta semanal — Tasa de
rendimiento — Cetes a 28 días". Its metadata declares daily periodicity but the observations
are **weekly auction results**, dated by settlement (Thursdays). ``SF60633`` carries the same
auctions dated by auction day (Tuesdays) instead; settlement dating is used here because that
is when the instrument exists to be held. ``SF282`` is a monthly average and is deliberately
not used: averaging across the month folds in information that was not available at the point
the portfolio was formed.

Values are annualised percentages on a 360-day basis and dates arrive as ``DD/MM/YYYY``.

**Rate limits.** Banxico caps historical queries at 200 per 5 minutes and 10,000 per day, and
blocks the token for the remainder of the window when exceeded. This module stays far below
that -- the whole study needs two historical requests -- but it caches to disk anyway, which
is what Banxico's own guidance recommends, and it parses the ``Bmx-secondsToReset`` header so
a block produces an actionable message instead of an opaque HTTP 400.
"""

from __future__ import annotations

import calendar
import datetime as dt
import json
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

__all__ = [
    "SIE_SERIES",
    "TOKEN_ENV_VAR",
    "SIEError",
    "SIERateLimited",
    "load_env_file",
    "get_token",
    "fetch_series",
    "load_series",
    "monthly_risk_free",
]

SIE_BASE = "https://www.banxico.org.mx/SieAPIRest/service/v1"
TOKEN_ENV_VAR = "BANXICO_SIE_TOKEN"

# Banxico's day-count convention for CETES yields.
DAY_COUNT_BASIS = 360


@dataclass(frozen=True, slots=True)
class SeriesSpec:
    """A Banxico series, with the metadata verified against the live API."""

    series_id: str
    label: str
    title: str
    observation_frequency: str
    units: str


SIE_SERIES: dict[str, SeriesSpec] = {
    "cetes_28d": SeriesSpec(
        series_id="SF43936",
        label="cetes_28d",
        title=(
            "Valores gubernamentales - Resultados de la subasta semanal - "
            "Tasa de rendimiento - Cetes a 28 dias"
        ),
        observation_frequency="weekly (settlement-dated)",
        units="annualised percent, 360-day basis",
    ),
    "fx_fix": SeriesSpec(
        series_id="SF43718",
        label="fx_fix",
        title=(
            "Tipo de cambio pesos por dolar E.U.A. para solventar obligaciones "
            "denominadas en moneda extranjera - Fecha de determinacion (FIX)"
        ),
        observation_frequency="daily",
        units="MXN per USD",
    ),
}


class SIEError(RuntimeError):
    """A Banxico SIE request failed."""


class SIERateLimited(SIEError):
    """The token is blocked for exceeding a query limit."""

    def __init__(self, message: str, seconds_to_reset: int | None) -> None:
        super().__init__(message)
        self.seconds_to_reset = seconds_to_reset


def load_env_file(path: str | Path = ".env") -> dict[str, str]:
    """Read a minimal ``KEY=value`` env file.

    Written inline rather than adding a dependency for ten lines. Values are returned, not
    injected into ``os.environ``, so a caller decides the precedence.
    """
    path = Path(path)
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip("'\"")
    return values


def get_token(env_file: str | Path = ".env") -> str:
    """Resolve the SIE token from the environment, falling back to ``.env``.

    The token is never logged, echoed or written into any artefact. Only its presence is
    recorded, in the run manifest.

    Raises:
        SIEError: if no token is configured, naming both places it is looked for.
    """
    token = os.environ.get(TOKEN_ENV_VAR) or load_env_file(env_file).get(TOKEN_ENV_VAR)
    if not token:
        raise SIEError(
            f"no Banxico SIE token: set {TOKEN_ENV_VAR} in the environment or in {env_file}. "
            "A free token is issued at "
            "https://www.banxico.org.mx/SieAPIRest/service/v1/token"
        )
    return token


def _raise_for_response(response: requests.Response) -> None:
    """Turn a failed SIE response into a specific exception.

    Banxico signals a rate-limit block with HTTP 400 and a body naming when the token
    unblocks, rather than with 429. Surfacing that distinctly is the difference between a
    caller that can wait and one that just sees a bad request.
    """
    if response.status_code == 200:
        return
    seconds: int | None = None
    detail = response.text[:300]
    header_seconds = response.headers.get("Bmx-secondsToReset")
    try:
        payload = response.json().get("error", {})
        detail = payload.get("detalle") or payload.get("mensaje") or detail
        seconds = payload.get("secondsToReset")
    except (ValueError, AttributeError):
        pass
    if seconds is None and header_seconds is not None:
        try:
            seconds = int(header_seconds)
        except ValueError:
            seconds = None
    if seconds is not None:
        raise SIERateLimited(
            f"Banxico SIE token is rate limited; unblocks in {seconds}s. {detail}", seconds
        )
    raise SIEError(f"Banxico SIE returned HTTP {response.status_code}: {detail}")


def _cache_path(cache_dir: str | Path, label: str) -> Path:
    return Path(cache_dir) / f"banxico_{label}.parquet"


def fetch_series(
    label: str,
    start: dt.date,
    end: dt.date,
    cache_dir: str | Path,
    *,
    token: str | None = None,
    force: bool = False,
    timeout: float = 60.0,
) -> pd.DataFrame:
    """Retrieve one Banxico series into the cache and return it.

    Args:
        label: Key in :data:`SIE_SERIES`.
        start: First date requested, inclusive.
        end: Last date requested, inclusive.
        cache_dir: Where the parquet cache lives.
        token: Overrides the configured token; resolved from the environment otherwise.
        force: Re-request even when the cache already covers the window.

    Returns:
        A frame indexed by date with a single ``value`` column of floats.

    Raises:
        SIERateLimited: if the token is blocked, carrying the seconds until it unblocks.
        SIEError: on any other API failure, or if the series returns no observations.
    """
    if label not in SIE_SERIES:
        raise SIEError(f"unknown series {label!r}; known: {sorted(SIE_SERIES)}")
    spec = SIE_SERIES[label]
    path = _cache_path(cache_dir, label)

    if path.exists() and not force:
        cached = pd.read_parquet(path)
        if not cached.empty:
            index = pd.DatetimeIndex(cached.index)
            if index.min().date() <= start + dt.timedelta(days=14):
                return cached

    url = f"{SIE_BASE}/series/{spec.series_id}/datos/{start.isoformat()}/{end.isoformat()}"
    response = requests.get(
        url,
        headers={"Bmx-Token": token or get_token(), "Accept": "application/json"},
        timeout=timeout,
    )
    _raise_for_response(response)

    series = response.json()["bmx"]["series"][0]
    observations = series.get("datos") or []
    if not observations:
        raise SIEError(
            f"series {spec.series_id} ({label}) returned no observations for "
            f"{start}..{end}; the window or the identifier is wrong"
        )

    frame = pd.DataFrame(observations)
    # Banxico dates are DD/MM/YYYY and values may carry 'N/E' for unavailable observations.
    frame["date"] = pd.to_datetime(frame["fecha"], format="%d/%m/%Y")
    frame["value"] = pd.to_numeric(
        frame["dato"].astype(str).str.replace(",", "", regex=False), errors="coerce"
    )
    frame = (
        frame.dropna(subset=["value"])
        .set_index("date")[["value"]]
        .sort_index()
    )
    frame = frame[~frame.index.duplicated(keep="last")]

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)
    meta = {
        "series_id": spec.series_id,
        "label": spec.label,
        "title": spec.title,
        "observation_frequency": spec.observation_frequency,
        "units": spec.units,
        "requested_start": start.isoformat(),
        "requested_end": end.isoformat(),
        "observations": int(len(frame)),
        "first_observation": str(frame.index.min().date()),
        "last_observation": str(frame.index.max().date()),
        "retrieved_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    path.with_suffix(".json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return frame


def load_series(label: str, cache_dir: str | Path) -> pd.DataFrame:
    """Read a cached Banxico series.

    Raises:
        FileNotFoundError: if the series was never retrieved. Downstream stages must not
            silently proceed without the risk-free rate.
    """
    path = _cache_path(cache_dir, label)
    if not path.exists():
        raise FileNotFoundError(
            f"{label} is not cached at {path}; run the rates stage before any stage that "
            "needs it"
        )
    frame = pd.read_parquet(path)
    frame.index = pd.DatetimeIndex(frame.index, name="date")
    return frame


def monthly_risk_free(
    months: list[str],
    cache_dir: str | Path,
    *,
    label: str = "cetes_28d",
) -> pd.Series:
    """Convert the annualised CETES quote into a holding-month rate for each month.

    For month ``t`` the rate used is the **last auction result published on or before the
    month's first calendar day** -- the rate an investor could actually have locked in when
    the portfolio was formed. Using a within-month average, or the month's own later
    auctions, would fold in information the portfolio did not have.

    The conversion is simple interest on Banxico's 360-day basis:

        r_month = (annual_percent / 100) x (calendar_days_in_month / 360)

    Args:
        months: Months as ``YYYY-MM``.
        cache_dir: Cache holding the retrieved series.
        label: Series key; defaults to the 28-day CETES.

    Returns:
        Holding-month rates indexed by month string.

    Raises:
        SIEError: if any requested month has no quote at or before its first day. A missing
            risk-free rate is never imputed: it would silently corrupt every Sharpe ratio
            for that month.
    """
    series = load_series(label, cache_dir)["value"]
    rates: dict[str, float] = {}
    missing: list[str] = []
    for month in months:
        year, month_number = (int(part) for part in month.split("-"))
        first_day = pd.Timestamp(year=year, month=month_number, day=1)
        available = series.loc[series.index <= first_day]
        if available.empty:
            missing.append(month)
            continue
        days_in_month = calendar.monthrange(year, month_number)[1]
        rates[month] = float(available.iloc[-1]) / 100.0 * days_in_month / DAY_COUNT_BASIS
    if missing:
        raise SIEError(
            f"no {label} quote on or before the first day of: {missing}. Extend the "
            "retrieval window; the risk-free rate is never imputed."
        )
    return pd.Series(rates, name=f"{label}_monthly")
