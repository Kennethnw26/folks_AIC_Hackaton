"""Tests for ocr.extract — happy path and failure path.

No real HTTP calls are made; the vision model client is patched with
unittest.mock.AsyncMock so tests run without a Chutes API key.
"""
from __future__ import annotations

import asyncio
import json
from decimal import Decimal
from unittest.mock import AsyncMock, patch

import pytest

from ocr.extract import PaymentExtract, extract_payment_proof

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
_VALID_PAYLOAD = {
    "amount": "1250.00",
    "currency": "MYR",
    "payment_date": "2024-03-20",
    "sender_name": "Lim Wei Jie",
    "reference": "TXN-20240320-001",
    "destination_account": "5141 2345 6789",
    "raw_text": (
        "CIMB Clicks Transfer Successful "
        "Amount MYR 1,250.00 To Lim Wei Jie "
        "Acc 5141 2345 6789 Ref TXN-20240320-001 Date 20 Mar 2024"
    ),
    "extraction_confidence": 0.93,
}

_FAKE_IMAGE = b"\xff\xd8\xff\xe0fake_jpeg_bytes"


# ---------------------------------------------------------------------------
# Test 1 — Happy path
# ---------------------------------------------------------------------------
def test_extract_payment_proof_happy_path() -> None:
    """Valid vision response is parsed into a PaymentExtract with Decimal amount."""
    valid_json_str = json.dumps(_VALID_PAYLOAD)

    with patch("ocr.extract._call_model", new=AsyncMock(return_value=valid_json_str)):
        result = asyncio.run(extract_payment_proof(_FAKE_IMAGE))

    assert isinstance(result, PaymentExtract)
    assert result.amount == Decimal("1250.00")
    assert isinstance(result.amount, Decimal), "amount must be Decimal, not float"
    assert result.currency == "MYR"
    assert result.sender_name == "Lim Wei Jie"
    assert result.reference == "TXN-20240320-001"
    assert result.destination_account == "5141 2345 6789"
    assert result.extraction_confidence == pytest.approx(0.93)
    assert result.payment_date.isoformat() == "2024-03-20"


# ---------------------------------------------------------------------------
# Test 2 — Failure path: both models raise, RuntimeError is propagated
# ---------------------------------------------------------------------------
def test_extract_payment_proof_all_models_fail() -> None:
    """RuntimeError is raised when both primary and fallback models fail."""

    async def _always_fail(client, model, image_b64):  # noqa: ARG001
        raise ConnectionError(f"Model {model} unreachable")

    with patch("ocr.extract._call_model", side_effect=_always_fail):
        with pytest.raises(RuntimeError, match="All vision models failed"):
            asyncio.run(extract_payment_proof(_FAKE_IMAGE))
