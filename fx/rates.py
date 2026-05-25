"""Synchronous FX rate lookups via Frankfurter API with disk caching.

Cache file: .cache/fx_rates.json
Keys:       "FROM:TO:YYYY-MM-DD" -> Decimal-string rate

On a 404 or timeout for a historical date, falls back to the /latest
endpoint and logs a warning.  All monetary arithmetic uses Decimal.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_BASE: str = os.getenv("FRANKFURTER_BASE", "https://api.frankfurter.app")
_CACHE_PATH: Path = Path(".cache/fx_rates.json")
_TIMEOUT: float = 10.0

# In-memory cache: "FROM:TO:DATE" -> rate as str (preserved from JSON)
_cache: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------
def _load_cache() -> None:
    """Read the on-disk cache into memory, silently ignoring corruption."""
    global _cache
    if _CACHE_PATH.exists():
        try:
            _cache = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("FX cache unreadable, starting empty: %s", exc)
            _cache = {}


def _save_cache() -> None:
    """Persist the current in-memory cache to disk."""
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_PATH.write_text(
            json.dumps(_cache, indent=2, sort_keys=True), encoding="utf-8"
        )
    except OSError as exc:
        logger.warning("Could not write FX cache: %s", exc)


def _cache_key(from_ccy: str, to_ccy: str, d: date) -> str:
    return f"{from_ccy}:{to_ccy}:{d.isoformat()}"


# Load cache on module import
_load_cache()


# ---------------------------------------------------------------------------
# Private fetch helpers
# ---------------------------------------------------------------------------
def _fetch_historical(from_ccy: str, to_ccy: str, d: date) -> Decimal:
    """Fetch the rate for a specific historical date from Frankfurter."""
    url = f"{_BASE}/{d.isoformat()}"
    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            resp = client.get(url, params={"from": from_ccy, "to": to_ccy})
            if resp.status_code == 404:
                raise httpx.HTTPStatusError(
                    f"404 for {d.isoformat()}",
                    request=resp.request,
                    response=resp,
                )
            resp.raise_for_status()
            data = resp.json()
            return Decimal(str(data["rates"][to_ccy]))
    except (httpx.HTTPStatusError, httpx.TimeoutException) as exc:
        logger.warning(
            "Historical FX lookup failed for %s (%s→%s): %s — falling back to /latest",
            d.isoformat(),
            from_ccy,
            to_ccy,
            exc,
        )
        return _fetch_latest(from_ccy, to_ccy)


def _fetch_latest(from_ccy: str, to_ccy: str) -> Decimal:
    """Fetch the current rate from /latest."""
    url = f"{_BASE}/latest"
    with httpx.Client(timeout=_TIMEOUT) as client:
        resp = client.get(url, params={"from": from_ccy, "to": to_ccy})
        resp.raise_for_status()
        data = resp.json()
        return Decimal(str(data["rates"][to_ccy]))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_rate(from_ccy: str, to_ccy: str, settlement_date: date) -> Decimal:
    """Return the exchange rate from_ccy → to_ccy on settlement_date.

    Results are cached to ``_CACHE_PATH`` (default: .cache/fx_rates.json).
    Cache is keyed by "FROM:TO:YYYY-MM-DD" and persists across process
    restarts.

    If the historical lookup returns 404 or times out, /latest is used
    instead and a warning is logged.

    Args:
        from_ccy: Source ISO 4217 currency code (case-insensitive).
        to_ccy: Target ISO 4217 currency code (case-insensitive).
        settlement_date: Date for the historical rate.

    Returns:
        Exact Decimal exchange rate.
    """
    from_ccy = from_ccy.upper()
    to_ccy = to_ccy.upper()

    if from_ccy == to_ccy:
        return Decimal("1")

    key = _cache_key(from_ccy, to_ccy, settlement_date)
    if key in _cache:
        return Decimal(_cache[key])

    rate = _fetch_historical(from_ccy, to_ccy, settlement_date)

    _cache[key] = str(rate)
    _save_cache()

    return rate


def convert(
    amount: Decimal,
    from_ccy: str,
    to_ccy: str,
    settlement_date: date,
) -> Decimal:
    """Convert amount from from_ccy to to_ccy using the rate on settlement_date.

    Uses Decimal arithmetic throughout; result is quantized to 4 decimal
    places with ROUND_HALF_UP.

    Args:
        amount: Source amount as Decimal.
        from_ccy: Source ISO 4217 currency code.
        to_ccy: Target ISO 4217 currency code.
        settlement_date: Date for the exchange rate.

    Returns:
        Converted amount quantized to 4 d.p.
    """
    rate = get_rate(from_ccy, to_ccy, settlement_date)
    result = amount * rate
    return result.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
