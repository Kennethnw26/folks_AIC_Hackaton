from __future__ import annotations

import base64
import logging

from schemas.llm_responses import PaymentProof
from tools.chutes_client import MODEL_VISION, call_chutes_vision

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a payment proof extractor. Analyze the image and return STRICT JSON only.\n\n"
    "Required fields:\n"
    '- amount: float (no currency symbol, no thousand separators)\n'
    '- currency: ISO 4217 3-letter code. Aliases:\n'
    '    "RM" / "Ringgit" -> "MYR"\n'
    '    "S$" -> "SGD"\n'
    '    "US$" / "$" -> "USD"\n'
    '    "€" -> "EUR"\n'
    '    "\xa5" -> use context: Japanese -> "JPY", Chinese -> "CNY"\n'
    '    "\xa3" -> "GBP"\n'
    '- date: ISO YYYY-MM-DD\n'
    '- beneficiary_name: string, the recipient name\n'
    '- bank_ref: string, transaction reference if visible, else ""\n'
    '- confidence: 0.0-1.0, your overall extraction confidence\n\n'
    "Output JSON only. No prose, no markdown code fences, no commentary."
)


async def extract_proof(image_bytes: bytes) -> PaymentProof | None:
    image_b64 = base64.b64encode(image_bytes).decode()
    result = await call_chutes_vision(
        model=MODEL_VISION,
        image_b64=image_b64,
        system_prompt=_SYSTEM_PROMPT,
        user_text="Extract all payment fields from this receipt image and return JSON only.",
        response_model=PaymentProof,
    )
    return result
