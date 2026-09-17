"""Candidate ticker universe assembly.

The BMV's own issuer directory is a JavaScript application over a JBoss backend whose
search and export endpoints reject programmatic calls, and the Internet Archive holds no
snapshots of its listing pages. So the candidate list is assembled from a public listing of
BMV-traded instruments and converted to the price provider's symbol convention.

This is a **stated selection rule**, not a reconstruction of historical exchange membership.
It is reproducible and its provenance is recorded, but it cannot recover issuers that were
already absent from the source when it was retrieved. That limitation is quantified by
:mod:`bmvport.marketdata.coverage` and must be declared in the manuscript.

Symbol convention: the public listing writes national series as ``CLAVE.SERIE``
(``AMX.B``, ``FEMSA.UBD``), while the price provider concatenates them and appends the
exchange suffix (``AMXB.MX``, ``FEMSAUBD.MX``). SIC instruments carry no series and take the
suffix directly (``AAPL`` -> ``AAPL.MX``).
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests

__all__ = [
    "Candidate",
    "PUBLIC_LISTING_URL",
    "to_provider_symbol",
    "fetch_candidates",
    "write_candidates",
    "read_candidates",
]

PUBLIC_LISTING_URL = "https://stockanalysis.com/list/mexican-stock-exchange/"

_USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"

# Curated national issuers, unioned with the public listing.
#
# This exists because the public listing is overwhelmingly composed of SIC instruments and
# resolves only nine Mexican issuers. Using it alone yields a universe of roughly 55 foreign
# instruments and 9 domestic ones, which is not the Mexican Stock Exchange in any meaningful
# sense -- it is US equities quoted in pesos, and a study built on it would be measuring an
# FX overlay on US returns rather than the market the paper is about.
#
# Every symbol below was verified to resolve at the price provider with a full daily history
# over 2022-2026 before being included. Series suffixes follow the provider's concatenated
# convention. This is a stated selection rule and must be reported as such: it is a curated
# list of liquid domestic issuers, not exchange membership.
NATIONAL_SEED: tuple[str, ...] = (
    "AC", "AGUA", "ALPEKA", "ALSEA", "AMXB", "ASURB", "AXTELCPO", "BBAJIOO", "BIMBOA",
    "BOLSAA", "CEMEXCPO", "CHDRAUIB", "CUERVO", "DANHOS13", "FEMSAUBD", "FIBRAPL14",
    "FIHO12", "FMTY14", "FUNO11", "GAPB", "GCARSOA1", "GCC", "GENTERA", "GFINBURO",
    "GFNORTEO", "GMEXICOB", "GRUMAB", "KIMBERA", "KOFUBL", "LABB", "LACOMERUBC",
    "LIVEPOLC-1", "MEGACPO", "NEMAKA", "OMAB", "ORBIA", "PINFRA", "Q", "RA", "VESTA",
    "VOLARA", "WALMEX",
)

# Auxiliary series: domestic index, foreign index, and the exchange rate. Needed for the
# benchmark comparators and for the FX component of return attribution.
AUXILIARY_SERIES: tuple[str, ...] = ("^MXX", "^GSPC", "USDMXN=X")

# Symbols appearing in the listing payload that are not tradeable instruments.
_SYMBOL_PATTERN = re.compile(r'"s"\s*:\s*"([A-Z0-9][A-Z0-9&.\-*]{0,14})"')
_QUOTE_PATTERN = re.compile(r"/quote/bmv/([A-Za-z0-9&.\-*]{1,15})")


@dataclass(frozen=True, slots=True)
class Candidate:
    """One candidate instrument.

    Attributes:
        source_symbol: Symbol as written by the listing source.
        provider_symbol: Symbol in the price provider's convention.
        listing: ``"national"`` when the source symbol carries a share series, otherwise
            ``"sic_or_unseried"``. The distinction drives the FX component of return
            attribution, so it is recorded rather than inferred later.
    """

    source_symbol: str
    provider_symbol: str
    listing: str


def to_provider_symbol(source_symbol: str, suffix: str = ".MX") -> str:
    """Convert a listing symbol to the price provider's convention.

    Verified against the provider on a 27-symbol sample covering national series and SIC
    instruments; 26 resolved, the exception being a recently listed SIC name with a short
    history that the liquidity screen rejects anyway.
    """
    return source_symbol.replace(".", "").replace("&", "") + suffix


def _classify(source_symbol: str) -> str:
    return "national" if "." in source_symbol else "sic_or_unseried"


def fetch_candidates(
    url: str = PUBLIC_LISTING_URL, *, timeout: float = 60.0
) -> tuple[list[Candidate], dict[str, object]]:
    """Retrieve and parse the public listing.

    Returns:
        The candidates and a provenance record (source URL, retrieval timestamp, raw
        symbol count) to be written alongside them.

    Raises:
        RuntimeError: if the listing cannot be retrieved or yields no symbols. A partial
            candidate list is never returned: it would silently shrink the universe.
    """
    response = requests.get(url, headers={"User-Agent": _USER_AGENT}, timeout=timeout)
    if response.status_code != 200:
        raise RuntimeError(
            f"listing source returned HTTP {response.status_code} for {url}"
        )
    html = response.text

    symbols: set[str] = set()
    symbols.update(_SYMBOL_PATTERN.findall(html))
    symbols.update(match.upper() for match in _QUOTE_PATTERN.findall(html))
    symbols = {s for s in symbols if not s.startswith("BMV")}

    if not symbols:
        raise RuntimeError(
            f"listing source {url} yielded no symbols; its markup has probably changed "
            "and the parser must be revisited before the universe can be trusted"
        )

    candidates = [
        Candidate(
            source_symbol=symbol,
            provider_symbol=to_provider_symbol(symbol),
            listing=_classify(symbol),
        )
        for symbol in sorted(symbols)
    ]

    # Union in the curated national issuers the public listing omits.
    present = {c.provider_symbol for c in candidates}
    seeded = 0
    for symbol in NATIONAL_SEED:
        provider_symbol = f"{symbol}.MX"
        if provider_symbol in present:
            # Reclassify: the public listing may carry it without a series suffix.
            candidates = [
                Candidate(c.source_symbol, c.provider_symbol, "national")
                if c.provider_symbol == provider_symbol
                else c
                for c in candidates
            ]
            continue
        candidates.append(Candidate(symbol, provider_symbol, "national"))
        present.add(provider_symbol)
        seeded += 1

    candidates.sort(key=lambda c: c.provider_symbol)
    provenance = {
        "source_url": url,
        "retrieved_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "symbol_count": len(candidates),
        "national_count": sum(1 for c in candidates if c.listing == "national"),
        "public_listing_symbols": len(symbols),
        "national_seed_symbols": len(NATIONAL_SEED),
        "national_seed_added": seeded,
        "note": (
            "union of a public BMV listing and a curated list of liquid domestic issuers. "
            "A stated selection rule, not historical exchange membership: the public "
            "listing resolves only nine domestic issuers, and issuers already absent from "
            "it at retrieval time cannot be recovered."
        ),
    }
    return candidates, provenance


def write_candidates(
    candidates: Iterable[Candidate],
    provenance: dict[str, object],
    directory: str | Path,
) -> tuple[Path, Path]:
    """Write the candidate list and its provenance record."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / "candidates.csv"
    json_path = directory / "candidates_provenance.json"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_symbol", "provider_symbol", "listing"])
        for candidate in candidates:
            writer.writerow(
                [candidate.source_symbol, candidate.provider_symbol, candidate.listing]
            )

    json_path.write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return csv_path, json_path


def read_candidates(directory: str | Path) -> list[Candidate]:
    """Read a previously written candidate list."""
    path = Path(directory) / "candidates.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run the universe stage before any stage that needs it"
        )
    with path.open(encoding="utf-8") as handle:
        return [
            Candidate(
                source_symbol=row["source_symbol"],
                provider_symbol=row["provider_symbol"],
                listing=row["listing"],
            )
            for row in csv.DictReader(handle)
        ]
