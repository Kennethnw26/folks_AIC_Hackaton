from __future__ import annotations

import logging

import diskcache
import httpx

logger = logging.getLogger(__name__)

_MEM_CACHE: dict[str, float] = {}
_DISK_CACHE = diskcache.Cache("./fx_cache")
SUPPORTED: set[str] = set()

_FRANKFURTER = "https://api.frankfurter.dev/v1"
_FALLBACK = "https://open.er-api.com/v6/latest"


async def load_supported() -> None:
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get(f"{_FRANKFURTER}/currencies")
            r.raise_for_status()
            SUPPORTED.update(r.json().keys())
    except Exception as e:
        logger.warning("Could not load Frankfurter currencies: %s", e)


async def _fetch_frankfurter(from_curr: str, to_curr: str, date_str: str) -> float | None:
    try:
        url = f"{_FRANKFURTER}/{date_str}?from={from_curr}&to={to_curr}"
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get(url)
            r.raise_for_status()
            return float(r.json()["rates"][to_curr])
    except Exception:
        return None


async def _fetch_fallback(from_curr: str, to_curr: str) -> float | None:
    # open.er-api.com: free, no key, supports CNY and most currencies
    # Returns latest rates only (no historical), acceptable as fallback
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get(f"{_FALLBACK}/{from_curr}")
            r.raise_for_status()
            return float(r.json()["rates"][to_curr])
    except Exception as e:
        logger.warning("Fallback FX lookup failed (%s→%s): %s", from_curr, to_curr, e)
        return None


async def get_rate(from_curr: str, to_curr: str, date_str: str) -> float | None:
    if from_curr == to_curr:
        return 1.0

    key = f"{from_curr}_{to_curr}_{date_str}"
    if key in _MEM_CACHE:
        return _MEM_CACHE[key]
    if key in _DISK_CACHE:
        _MEM_CACHE[key] = _DISK_CACHE[key]
        return _MEM_CACHE[key]

    rate = await _fetch_frankfurter(from_curr, to_curr, date_str)

    if rate is None:
        logger.info("Frankfurter missing %s→%s, trying fallback", from_curr, to_curr)
        rate = await _fetch_fallback(from_curr, to_curr)

    if rate is not None:
        _MEM_CACHE[key] = rate
        _DISK_CACHE[key] = rate
    else:
        logger.warning("All FX sources failed for %s→%s on %s", from_curr, to_curr, date_str)

    return rate
