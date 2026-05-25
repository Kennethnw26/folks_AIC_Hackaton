"""FastAPI entrypoint for the Global Treasury Agent.

Run:
    uvicorn main:app --reload --port 8000
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from db.session import init_db

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Global Treasury Agent",
    version="0.1.0",
    description="Invoice/payment reconciliation, FX, fraud, and WhatsApp ops.",
    lifespan=lifespan,
)


@app.get("/health", tags=["meta"])
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "treasury-agent",
        "version": app.version,
    }


# ---------------------------------------------------------------------------
# WhatsApp webhook (Meta Cloud API)
#   GET  -> verification handshake
#   POST -> inbound message / status events (stubbed for now)
# ---------------------------------------------------------------------------
@app.get("/webhook/whatsapp", tags=["whatsapp"])
def whatsapp_verify(
    hub_mode: str = Query(default="", alias="hub.mode"),
    hub_challenge: str = Query(default="", alias="hub.challenge"),
    hub_verify_token: str = Query(default="", alias="hub.verify_token"),
):
    expected = os.getenv("WHATSAPP_VERIFY_TOKEN", "")
    if hub_mode == "subscribe" and hub_verify_token and hub_verify_token == expected:
        return PlainTextResponse(content=hub_challenge, status_code=200)
    return JSONResponse(
        status_code=403,
        content={"error": "verification_failed"},
    )


@app.post("/webhook/whatsapp", tags=["whatsapp"])
async def whatsapp_inbound(request: Request) -> dict[str, Any]:
    payload = await request.json()
    # TODO: dispatch to whatsapp.handlers in a later prompt.
    return {"received": True, "payload_keys": list(payload.keys())}
