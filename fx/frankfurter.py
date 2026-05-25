"""Thin async client for the Frankfurter FX API."""
from __future__ import annotations

import os
from datetime import date

import httpx
from dotenv import load_dotenv

load_dotenv()

FRANKFURTER_BASE = os.getenv("FRANKFURTER_BASE", "https://api.frankfurter.app")

async def get_rate(on: date, base: str, quote: str) -> float:
    """Return the `base -> quote` exchange rate on the given date."""
    base = base.upper()
    quote = quote.upper()
    if base == quote:
        return 1.0
    url = f"{FRANKFURTER_BASE}/{on.isoformat()}"
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(url, params={"from": base, "to": quote})
        resp.raise_for_status()
        data = resp.json()
    return float(data["rates"][quote])
