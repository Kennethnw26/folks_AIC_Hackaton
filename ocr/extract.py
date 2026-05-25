"""Payment-proof OCR extraction using Chutes vision-capable models.

Tries NousResearch/Hermes-4-405B first; falls back to Qwen2.5-VL-72B if
the primary model is unavailable. On JSON parse failure the model is asked
to repair the output -- up to _MAX_RETRIES times before raising.
"""
from __future__ import annotations

import base64
import json
import logging
import os
from datetime import date
from decimal import Decimal
from typing import Any, Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------
_CHUTES_BASE: str = os.getenv("CHUTES_BASE_URL", "https://llm.chutes.ai/v1")
_CHUTES_KEY: str = os.getenv("CHUTES_API_KEY", "")
_PRIMARY_MODEL: str = os.getenv("CHUTES_VISION_MODEL", "NousResearch/Hermes-4-405B")
_FALLBACK_MODEL: str = "Qwen/Qwen2.5-VL-72B-Instruct"
_MAX_RETRIES: int = 2


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------
class PaymentExtract(BaseModel):
    """Structured data extracted from a bank payment screenshot."""

    amount: Decimal = Field(description="Transaction amount -- always Decimal, never float")
    currency: str = Field(description="ISO 4217 currency code, e.g. MYR, USD, GBP")
    payment_date: date = Field(description="Date the payment was executed")
    sender_name: str = Field(description="Full name of the sending party")
    reference: Optional[str] = Field(
        default=None,
        description="Payment reference, transaction ID, or description -- null if absent",
    )
    destination_account: Optional[str] = Field(
        default=None,
        description="Recipient account number or IBAN -- null if absent",
    )
    raw_text: str = Field(description="Verbatim text content of the receipt image")
    extraction_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score 0-1 for the overall extraction quality",
    )


# JSON schema sent to the model as response_format
_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "amount": {
            "type": "string",
            "description": "Decimal string -- no thousand separators, e.g. '1234.56'",
        },
        "currency": {
            "type": "string",
            "description": "ISO 4217 three-letter code, e.g. MYR",
        },
        "payment_date": {
            "type": "string",
            "description": "ISO 8601 date YYYY-MM-DD",
        },
        "sender_name": {"type": "string"},
        "reference": {"type": ["string", "null"]},
        "destination_account": {"type": ["string", "null"]},
        "raw_text": {"type": "string"},
        "extraction_confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
        },
    },
    "required": [
        "amount",
        "currency",
        "payment_date",
        "sender_name",
        "raw_text",
        "extraction_confidence",
    ],
    "additionalProperties": False,
}

# System prompt with bank-specific guidance and two few-shot examples

_SYSTEM_PROMPT = (
    "You are a payment receipt OCR extractor. Given an image of a bank transfer\n"
    "screenshot, extract structured fields as JSON. Handle these formats:\n\n"
    "  * Maybank (Malaysia) -- 'Maybank2u' / 'MAE' app. Shows 'Transfer Successful',\n"
    "    amount in MYR, recipient name, account number, reference, and date.\n\n"
    "  * CIMB (Malaysia) -- 'CIMB Clicks' / 'CIMB OCTO'. Shows recipient name,\n"
    "    masked or full account, FPX/IBG reference, and timestamp.\n\n"
    "  * Public Bank (Malaysia) -- 'PBe' app. Shows transaction type (FPX / IBG),\n"
    "    beneficiary name, account number, reference, and date.\n\n"
    "  * Wise (international) -- Multi-currency. Shows 'You sent X CCY', recipient\n"
    "    full name, IBAN or account number, reference, and transfer date.\n\n"
    "Extract these fields exactly:\n"
    "  amount               -- numeric only, as decimal string, no thousand-separators\n"
    "  currency             -- ISO 4217 (MYR / USD / GBP / EUR / SGD / etc.)\n"
    "  payment_date         -- YYYY-MM-DD; infer year from context if the image omits it\n"
    "  sender_name          -- full name of the person / entity who sent the payment\n"
    "  reference            -- transaction ID, reference number, or memo; null if absent\n"
    "  destination_account  -- recipient account number or IBAN; null if absent\n"
    "  raw_text             -- verbatim full text of the receipt (line breaks as spaces)\n"
    "  extraction_confidence -- float 0-1: confidence in the extracted fields\n\n"
    "--- FEW-SHOT EXAMPLE 1 (Maybank MAE) ---\n"
    "Image text:\n"
    '  "MAE Transfer Successful\n'
    "   Amount  RM 2,500.00\n"
    "   To      Ahmad Bin Ibrahim\n"
    "   Account 5141 2345 6789\n"
    "   Ref     20240315001\n"
    '   Date    15 Mar 2024"\n\n'
    "Expected output:\n"
    "{\n"
    '  "amount": "2500.00",\n'
    '  "currency": "MYR",\n'
    '  "payment_date": "2024-03-15",\n'
    '  "sender_name": "Ahmad Bin Ibrahim",\n'
    '  "reference": "20240315001",\n'
    '  "destination_account": "51412345 6789",\n'
    '  "raw_text": "MAE Transfer Successful Amount RM 2,500.00 To Ahmad Bin Ibrahim Account 5141 2345 6789 Ref 20240315001 Date 15 Mar 2024",\n'
    '  "extraction_confidence": 0.95\n'
    "}\n\n"
    "--- FEW-SHOT EXAMPLE 2 (Wise international) ---\n"
    "Image text:\n"
    '  "You sent 500.00 GBP\n'
    "   Recipient  Siti Rahimah Binti Yusof\n"
    "   IBAN       GB29NWBK60161331926819\n"
    "   Reference  INV-2024-0042\n"
    "   Transfer date  2024-03-18\n"
    '   Transfer ID    WISE-TRF-9912345"\n\n'
    "Expected output:\n"
    "{\n"
    '  "amount": "500.00",\n'
    '  "currency": "GBP",\n'
    '  "payment_date": "2024-03-18",\n'
    '  "sender_name": "Siti Rahimah Binti Yusof",\n'
    '  "reference": "INV-2024-0042",\n'
    '  "destination_account": "GB29NWBK60161331926819",\n'
    '  "raw_text": "You sent 500.00 GBP Recipient Siti Rahimah Binti Yusof IBAN GB29NWBK60161331926819 Reference INV-2024-0042 Transfer date 2024-03-18 Transfer ID WISE-TRF-9912345",\n'
    '  "extraction_confidence": 0.97\n'
    "}\n\n"
    "Return ONLY valid JSON. No markdown, no code fences, no explanation."
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _build_client() -> AsyncOpenAI:
    # Use a placeholder when the key is absent so unit tests can run without
    # credentials -- the real key is enforced server-side by Chutes.
    return AsyncOpenAI(base_url=_CHUTES_BASE, api_key=_CHUTES_KEY or "placeholder")


def _parse_extract(raw: str) -> PaymentExtract:
    """Parse a raw JSON string into PaymentExtract.

    Converts the model's string amount to Decimal and the ISO date string
    to a date object before Pydantic validation.
    """
    data: dict[str, Any] = json.loads(raw)
    # Guarantee Decimal -- model may return a number literal
    data["amount"] = Decimal(str(data["amount"]))
    if isinstance(data.get("payment_date"), str):
        data["payment_date"] = date.fromisoformat(data["payment_date"])
    return PaymentExtract(**data)


async def _call_model(client: AsyncOpenAI, model: str, image_b64: str) -> str:
    """Send the image to the vision model and return the raw response text."""
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}",
                            "detail": "high",
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "Extract all payment fields from this receipt image "
                            "and return JSON only."
                        ),
                    },
                ],
            },
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "PaymentExtract",
                "schema": _JSON_SCHEMA,
                "strict": True,
            },
        },
        temperature=0.1,
        max_tokens=1024,
    )
    return response.choices[0].message.content or ""


async def _repair_json(client: AsyncOpenAI, model: str, broken: str) -> str:
    """Ask the model to repair malformed JSON output."""
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": (
                    "Fix this JSON, return only valid JSON conforming to the schema:\n\n"
                    f"{broken}\n\n"
                    f"Required schema:\n{json.dumps(_JSON_SCHEMA, indent=2)}"
                ),
            }
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
        max_tokens=1024,
    )
    return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
async def extract_payment_proof(image_bytes: bytes) -> PaymentExtract:
    """Extract structured payment data from a bank receipt image.

    Algorithm:
    1. Encode image as base64.
    2. Try _PRIMARY_MODEL; on any exception fall back to _FALLBACK_MODEL.
    3. Attempt to parse the JSON response.
    4. On parse failure send a repair request -- up to _MAX_RETRIES times.
    5. Raise ValueError if retries are exhausted.

    Args:
        image_bytes: Raw bytes of the payment screenshot (JPEG or PNG).

    Returns:
        PaymentExtract with all monetary fields as Decimal.

    Raises:
        RuntimeError: If all models fail to respond.
        ValueError: If JSON cannot be parsed after retries.
    """
    image_b64 = base64.b64encode(image_bytes).decode()
    client = _build_client()

    raw: Optional[str] = None
    active_model = _PRIMARY_MODEL

    for model in (_PRIMARY_MODEL, _FALLBACK_MODEL):
        try:
            raw = await _call_model(client, model, image_b64)
            active_model = model
            break
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Vision model %s failed: %s%s",
                model,
                exc,
                " -- trying fallback" if model == _PRIMARY_MODEL else "",
            )

    if raw is None:
        raise RuntimeError(
            f"All vision models failed ({_PRIMARY_MODEL}, {_FALLBACK_MODEL})"
        )

    last_raw = raw
    for attempt in range(_MAX_RETRIES + 1):
        try:
            return _parse_extract(last_raw)
        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            if attempt == _MAX_RETRIES:
                raise ValueError(
                    f"Could not parse extraction response after {_MAX_RETRIES} "
                    f"repair retries: {exc}\nLast response: {last_raw!r}"
                ) from exc
            logger.warning(
                "JSON parse failed (attempt %d/%d): %s -- requesting repair",
                attempt + 1,
                _MAX_RETRIES,
                exc,
            )
            last_raw = await _repair_json(client, active_model, last_raw)

    # Unreachable -- loop always returns or raises inside
    raise RuntimeError("Unexpected exit from retry loop")
