from __future__ import annotations

import logging

import httpx

from config import settings

logger = logging.getLogger(__name__)

_GRAPH_URL = "https://graph.facebook.com/v19.0"


async def send_text(to: str, text: str) -> None:
    url = f"{_GRAPH_URL}/{settings.meta_phone_id}/messages"
    headers = {"Authorization": f"Bearer {settings.meta_access_token}"}
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": text},
    }
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.post(url, headers=headers, json=payload)
            r.raise_for_status()
    except Exception as e:
        logger.error("WhatsApp send_text failed: %s", e)


async def send_interactive(to: str, message: dict) -> None:
    url = f"{_GRAPH_URL}/{settings.meta_phone_id}/messages"
    headers = {"Authorization": f"Bearer {settings.meta_access_token}"}
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        **message,
    }
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.post(url, headers=headers, json=payload)
            r.raise_for_status()
    except Exception as e:
        logger.error("WhatsApp send_interactive failed: %s", e)


async def download_media(media_id: str) -> bytes:
    headers = {"Authorization": f"Bearer {settings.meta_access_token}"}
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.get(f"{_GRAPH_URL}/{media_id}", headers=headers)
        r.raise_for_status()
        media_url: str = r.json()["url"]
        r2 = await c.get(media_url, headers=headers)
        r2.raise_for_status()
        return r2.content
