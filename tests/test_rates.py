"""Banxico SIE client: token handling, rate-limit reporting, and the risk-free conversion.

These tests never touch the network. The conversion tests build a synthetic cache so the
arithmetic and the point-in-time rule are pinned independently of what Banxico happens to
have published.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from bmvport.marketdata import rates

ROOT = Path(__file__).resolve().parents[1]


class _FakeResponse:
    """Minimal stand-in for the pieces of ``requests.Response`` the client reads."""

    def __init__(self, status_code: int, payload: dict | None, headers: dict | None = None):
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}
        self.text = str(payload)

    def json(self) -> dict:
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


def test_verified_series_identifiers() -> None:
    """The identifiers were confirmed against the live API; pin them so drift is visible."""
    assert rates.SIE_SERIES["cetes_28d"].series_id == "SF43936"
    assert rates.SIE_SERIES["fx_fix"].series_id == "SF43718"
    assert "Cetes a 28 dias" in rates.SIE_SERIES["cetes_28d"].title
    # Metadata declares daily periodicity but the observations are weekly auctions.
    assert "weekly" in rates.SIE_SERIES["cetes_28d"].observation_frequency


def test_env_file_parsing(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text(
        "# comment\n\nBANXICO_SIE_TOKEN='abc123'\nOTHER=plain\nMALFORMED\n", encoding="utf-8"
    )
    values = rates.load_env_file(path)
    assert values["BANXICO_SIE_TOKEN"] == "abc123"
    assert values["OTHER"] == "plain"
    assert "MALFORMED" not in values


def test_missing_token_names_both_lookup_locations(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv(rates.TOKEN_ENV_VAR, raising=False)
    with pytest.raises(rates.SIEError, match="no Banxico SIE token"):
        rates.get_token(tmp_path / "absent.env")


def test_environment_takes_precedence_over_env_file(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / ".env"
    path.write_text(f"{rates.TOKEN_ENV_VAR}=from_file\n", encoding="utf-8")
    monkeypatch.setenv(rates.TOKEN_ENV_VAR, "from_environment")
    assert rates.get_token(path) == "from_environment"


def test_rate_limit_is_distinguished_from_a_generic_failure() -> None:
    """Banxico signals a block with HTTP 400, not 429; the caller must be able to tell."""
    response = _FakeResponse(
        400,
        {
            "error": {
                "mensaje": "Límite de consultas superado.",
                "detalle": "Podrá volver a consultar a las 17:06:27",
                "secondsToReset": 54,
            }
        },
    )
    with pytest.raises(rates.SIERateLimited) as info:
        rates._raise_for_response(response)
    assert info.value.seconds_to_reset == 54


def test_rate_limit_falls_back_to_the_header() -> None:
    response = _FakeResponse(400, None, headers={"Bmx-secondsToReset": "42"})
    with pytest.raises(rates.SIERateLimited) as info:
        rates._raise_for_response(response)
    assert info.value.seconds_to_reset == 42


def test_other_failures_are_not_reported_as_rate_limits() -> None:
    response = _FakeResponse(500, None)
    with pytest.raises(rates.SIEError) as info:
        rates._raise_for_response(response)
    assert not isinstance(info.value, rates.SIERateLimited)


def _synthetic_cache(tmp_path: Path, observations: dict[str, float]) -> Path:
    frame = pd.DataFrame(
        {"value": list(observations.values())},
        index=pd.DatetimeIndex([pd.Timestamp(d) for d in observations], name="date"),
    )
    frame.to_parquet(tmp_path / "banxico_cetes_28d.parquet")
    return tmp_path


def test_monthly_conversion_uses_simple_interest_on_a_360_day_basis(tmp_path: Path) -> None:
    cache = _synthetic_cache(tmp_path, {"2024-12-26": 10.0})
    result = rates.monthly_risk_free(["2025-01"], cache)
    # 10% annual, 31 calendar days, 360-day basis
    assert result["2025-01"] == pytest.approx(0.10 * 31 / 360)


def test_month_length_changes_the_rate(tmp_path: Path) -> None:
    cache = _synthetic_cache(tmp_path, {"2024-12-26": 12.0})
    january = rates.monthly_risk_free(["2025-01"], cache)["2025-01"]
    february = rates.monthly_risk_free(["2025-02"], cache)["2025-02"]
    assert january == pytest.approx(0.12 * 31 / 360)
    assert february == pytest.approx(0.12 * 28 / 360)


def test_rate_is_point_in_time(tmp_path: Path) -> None:
    """The rate used is the last auction on or before the month's first day.

    A later auction inside the same month must not be picked up: the portfolio was formed
    before it happened.
    """
    cache = _synthetic_cache(
        tmp_path,
        {"2024-12-19": 9.0, "2024-12-26": 10.0, "2025-01-09": 99.0, "2025-01-30": 88.0},
    )
    result = rates.monthly_risk_free(["2025-01"], cache)
    assert result["2025-01"] == pytest.approx(0.10 * 31 / 360)


def test_quote_exactly_on_the_first_day_is_eligible(tmp_path: Path) -> None:
    cache = _synthetic_cache(tmp_path, {"2025-01-01": 8.0})
    assert rates.monthly_risk_free(["2025-01"], cache)["2025-01"] == pytest.approx(
        0.08 * 31 / 360
    )


def test_missing_rate_raises_rather_than_imputing(tmp_path: Path) -> None:
    """An imputed risk-free rate would silently corrupt every Sharpe ratio that month."""
    cache = _synthetic_cache(tmp_path, {"2025-06-05": 8.0})
    with pytest.raises(rates.SIEError, match="never imputed"):
        rates.monthly_risk_free(["2025-01", "2025-02"], cache)


def test_load_series_refuses_to_proceed_when_absent(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="run the rates stage"):
        rates.load_series("cetes_28d", tmp_path)


def test_unknown_series_label_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(rates.SIEError, match="unknown series"):
        rates.fetch_series("tiie_28d", dt.date(2025, 1, 1), dt.date(2025, 2, 1), tmp_path)


@pytest.mark.skipif(
    not (ROOT / "data" / "cache" / "banxico_cetes_28d.parquet").exists(),
    reason="Banxico series not retrieved in this environment",
)
def test_retrieved_series_are_plausible() -> None:
    """Sanity bounds on the real cached data, not exact values."""
    cache = ROOT / "data" / "cache"
    cetes = rates.load_series("cetes_28d", cache)["value"]
    fx = rates.load_series("fx_fix", cache)["value"]
    assert 3.0 < cetes.min() and cetes.max() < 20.0  # plausible policy-rate band
    assert 12.0 < fx.min() and fx.max() < 30.0  # plausible MXN/USD band
    assert cetes.index.is_monotonic_increasing
    assert fx.index.is_monotonic_increasing
