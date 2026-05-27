from __future__ import annotations

import hashlib
import hmac
import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import PlainTextResponse

from config import settings
from orchestrator import handle_message

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/webhooks/whatsapp")
async def verify_webhook(request: Request) -> PlainTextResponse:
    params = request.query_params
    if (
        params.get("hub.mode") == "subscribe"
        and params.get("hub.verify_token") == settings.meta_verify_token
    ):
        return PlainTextResponse(params.get("hub.challenge", ""))
    raise HTTPException(status_code=403, detail="Verification failed")


@router.post("/webhooks/whatsapp")
async def receive_webhook(request: Request, background_tasks: BackgroundTasks) -> dict:
    body = await request.body()

    sig_header = request.headers.get("X-Hub-Signature-256", "")
    expected = "sha256=" + hmac.new(
        settings.meta_app_secret.encode(), body, hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(expected, sig_header):
        raise HTTPException(status_code=403, detail="Invalid signature")

    try:
        data = await request.json()
        entry = data.get("entry", [{}])[0]
        changes = entry.get("changes", [{}])[0]
        value = changes.get("value", {})
        messages = value.get("messages", [])
        if not messages:
            return {"status": "ok"}

        msg = messages[0]
        envelope = {
            "message_id": msg.get("id", ""),
            "from": msg.get("from", ""),
            "type": msg.get("type", ""),
            "text": msg.get("text", {}).get("body", "") if msg.get("type") == "text" else "",
            "image": msg.get("image", {}) if msg.get("type") == "image" else {},
            "interactive": msg.get("interactive", {}) if msg.get("type") == "interactive" else {},
        }
        background_tasks.add_task(handle_message, envelope)
    except Exception as e:
        logger.error("Webhook parse error: %s", e)

    return {"status": "ok"}
