"""Tests for fx.rates — happy path and failure/fallback path.

HTTP calls are mocked with respx so no network access is required.
The fx_clean_cache fixture (defined in conftest.py) is applied to every
test to keep in-memory and on-disk state isolated.
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal

import httpx
import pytest
import respx

from fx import rates

pytestmark = pytest.mark.usefixtures("fx_clean_cache")

_SETTLEMENT_DATE = date(2024, 3, 20)
_FROM = "MYR"
_TO = "USD"
_RATE_STR = "0.2134"
_RATE = Decimal(_RATE_STR)

_HISTORICAL_URL = f"https://api.frankfurter.app/{_SETTLEMENT_DATE.isoformat()}"
_LATEST_URL = "https://api.frankfurter.app/latest"


# ---------------------------------------------------------------------------
# Test 1 — Happy path: historical rate returned and cached
# ---------------------------------------------------------------------------
@respx.mock
def test_get_rate_historical_success(fx_clean_cache) -> None:
    """Successful historical lookup returns a Decimal rate and caches it."""
    fake_cache = fx_clean_cache  # Path to the temp cache file from fixture

    respx.get(_HISTORICAL_URL).mock(
        return_value=httpx.Response(
            200,
            json={"amount": 1, "base": _FROM, "date": "2024-03-20", "rates": {_TO: float(_RATE)}},
        )
    )

    result = rates.get_rate(_FROM, _TO, _SETTLEMENT_DATE)

    assert isinstance(result, Decimal)
    assert result == _RATE

    # Verify it was written to the on-disk cache
    assert fake_cache.exists()
    cached = rates._cache[rates._cache_key(_FROM, _TO, _SETTLEMENT_DATE)]
    assert Decimal(cached) == _RATE

    # Second call must NOT hit the network (served from cache); respx would
    # raise if an unexpected request went out.
    result2 = rates.get_rate(_FROM, _TO, _SETTLEMENT_DATE)
    assert result2 == result


# ---------------------------------------------------------------------------
# Test 2 — Failure/fallback path: 404 on historical date → /latest used
# ---------------------------------------------------------------------------
@respx.mock
def test_get_rate_fallback_to_latest_on_404(fx_clean_cache) -> None:
    """404 on the historical endpoint triggers a /latest fallback."""
    # Historical endpoint returns 404
    respx.get(_HISTORICAL_URL).mock(
        return_value=httpx.Response(404, json={"message": "date not found"})
    )
    # /latest returns a valid rate
    respx.get(_LATEST_URL).mock(
        return_value=httpx.Response(
            200,
            json={"amount": 1, "base": _FROM, "date": "2024-03-20", "rates": {_TO: float(_RATE)}},
        )
    )

    result = rates.get_rate(_FROM, _TO, _SETTLEMENT_DATE)

    assert isinstance(result, Decimal)
    assert result == _RATE

    # Both routes must have been called
    assert respx.calls.call_count == 2

    # convert() should also work end-to-end
    converted = rates.convert(Decimal("1000"), _FROM, _TO, _SETTLEMENT_DATE)
    assert converted == (Decimal("1000") * _RATE).quantize(Decimal("0.0001"))
