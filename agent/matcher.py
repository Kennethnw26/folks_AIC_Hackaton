"""Core matching engine for the Global Treasury Agent.

Deterministically scores invoice-payment pairs, then routes top candidates
through Hermes 4 (via Chutes) for narrative reasoning and final confidence.

Weights are module-level constants so they can be tuned without touching logic.
All money arithmetic uses Decimal; amounts are quantized to 4 d.p. for
computation and 2 d.p. for display / storage.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from decimal import ROUND_HALF_UP, Decimal
from typing import Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel
from rapidfuzz import fuzz

from db.models import Counterparty, Invoice, Payment
from fx.rates import convert, get_rate

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunable weight constants
# ---------------------------------------------------------------------------
WEIGHT_AMOUNT: Decimal = Decimal("0.40")
WEIGHT_DATE: Decimal = Decimal("0.20")
WEIGHT_NAME: Decimal = Decimal("0.20")
WEIGHT_REFERENCE: Decimal = Decimal("0.20")

# Amount tolerance band
AMOUNT_EXACT_THRESHOLD: Decimal = Decimal("0.005")   # 0.5% → fit = 1.0
AMOUNT_ZERO_THRESHOLD: Decimal = Decimal("0.10")     # 10%  → fit = 0.0

# Date window (days from issued_date)
DATE_PERFECT_DAYS: int = 30    # 0–30 days → fit = 1.0
DATE_ZERO_DAYS: int = 60       # >60 days  → fit = 0.0

# Bank-fee detection band (proportion of converted invoice amount)
FEE_MIN_PCT: Decimal = Decimal("0.002")   # 0.2%
FEE_MAX_PCT: Decimal = Decimal("0.020")   # 2.0%

# Reference fuzzy-match threshold for partial score
_REF_PARTIAL_THRESHOLD: int = 80

# Hermes: max candidates forwarded for LLM verification
_TOP_K: int = 5

_MYR = "MYR"
_DP4 = Decimal("0.0001")
_DP2 = Decimal("0.01")


# ---------------------------------------------------------------------------
# Pydantic output models
# ---------------------------------------------------------------------------
class MatchScore(BaseModel):
    """Deterministic sub-scores for one invoice-payment pair."""

    amount_fit: Decimal
    date_fit: Decimal
    name_fit: Decimal
    reference_fit: Decimal
    fee_inferred: Decimal          # MYR; Decimal("0") when no fee detected
    composite_confidence: Decimal  # weighted blend, 0–1


class VerifiedMatch(BaseModel):
    """LLM-enriched match result returned to callers."""

    invoice_id: int
    payment_id: int
    score: MatchScore
    final_confidence: int          # 0–100, set by Hermes
    hermes_narrative: str
    caveats: list[str]
    fx_rate_used: Decimal


# ---------------------------------------------------------------------------
# JSON-schema contracts for Hermes responses
# ---------------------------------------------------------------------------
_VERIFY_SCHEMA: dict = {
    "type": "json_schema",
    "json_schema": {
        "name": "match_verification",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "final_confidence": {
                    "type": "integer",
                    "description": "Integer 0–100 confidence the payment matches the invoice",
                },
                "narrative": {
                    "type": "string",
                    "description": "One-paragraph English reasoning for the match decision",
                },
                "caveats": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of concerns or uncertainties about this match",
                },
            },
            "required": ["final_confidence", "narrative", "caveats"],
            "additionalProperties": False,
        },
    },
}

_RANK_EXPLAIN_SCHEMA: dict = {
    "type": "json_schema",
    "json_schema": {
        "name": "ranking_explanation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "explanation": {
                    "type": "string",
                    "description": (
                        "One paragraph explaining specifically why this candidate "
                        "ranks below the winner"
                    ),
                },
            },
            "required": ["explanation"],
            "additionalProperties": False,
        },
    },
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _async_client() -> AsyncOpenAI:
    api_key = os.getenv("CHUTES_API_KEY", "")
    base_url = os.getenv("CHUTES_BASE_URL", "https://llm.chutes.ai/v1")
    return AsyncOpenAI(api_key=api_key, base_url=base_url)


def _model() -> str:
    return os.getenv("CHUTES_MODEL", "NousResearch/Hermes-4-405B")


def _q4(d: Decimal) -> Decimal:
    return d.quantize(_DP4, rounding=ROUND_HALF_UP)


def _q2(d: Decimal) -> Decimal:
    return d.quantize(_DP2, rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# 1. score_pair — fully deterministic, no I/O except FX cache / API
# ---------------------------------------------------------------------------
def score_pair(
    invoice: Invoice,
    payment: Payment,
    counterparty: Optional[Counterparty] = None,
) -> MatchScore:
    """Score one invoice-payment candidate pair deterministically.

    Args:
        invoice:      The candidate invoice.
        payment:      The incoming payment to reconcile.
        counterparty: Counterparty record for the invoice customer (optional).
                      When absent, name_fit is 0.

    Returns:
        MatchScore with all sub-scores and composite_confidence.
    """
    inv_amount = Decimal(str(invoice.amount))
    pay_amount = Decimal(str(payment.raw_amount))

    # ------------------------------------------------------------------
    # FX: convert invoice amount into payment currency on payment date
    # ------------------------------------------------------------------
    fx_rate = get_rate(invoice.currency, payment.currency, payment.payment_date)
    converted = _q4(inv_amount * fx_rate)   # invoice value in payment.currency

    # ------------------------------------------------------------------
    # amount_fit
    # ------------------------------------------------------------------
    if converted == Decimal("0"):
        amount_fit = Decimal("0")
    else:
        gap_pct = abs(converted - pay_amount) / converted
        if gap_pct <= AMOUNT_EXACT_THRESHOLD:
            amount_fit = Decimal("1")
        elif gap_pct >= AMOUNT_ZERO_THRESHOLD:
            amount_fit = Decimal("0")
        else:
            span = AMOUNT_ZERO_THRESHOLD - AMOUNT_EXACT_THRESHOLD
            amount_fit = Decimal("1") - (gap_pct - AMOUNT_EXACT_THRESHOLD) / span
    amount_fit = _q4(amount_fit)

    # ------------------------------------------------------------------
    # date_fit
    # ------------------------------------------------------------------
    delta_days = (payment.payment_date - invoice.issued_date).days
    if 0 <= delta_days <= DATE_PERFECT_DAYS:
        date_fit = Decimal("1")
    elif delta_days > DATE_ZERO_DAYS or delta_days < 0:
        date_fit = Decimal("0")
    else:
        # Linear decay from day 30 to day 60
        overshoot = delta_days - DATE_PERFECT_DAYS
        span = DATE_ZERO_DAYS - DATE_PERFECT_DAYS
        date_fit = Decimal("1") - Decimal(overshoot) / Decimal(span)
    date_fit = _q4(date_fit)

    # ------------------------------------------------------------------
    # name_fit
    # ------------------------------------------------------------------
    if counterparty is not None and payment.sender_name:
        raw = fuzz.token_set_ratio(
            payment.sender_name,
            counterparty.normalized_name,
        )
        name_fit = _q4(Decimal(str(raw)) / Decimal("100"))
    else:
        name_fit = Decimal("0")

    # ------------------------------------------------------------------
    # reference_fit
    # ------------------------------------------------------------------
    ref = (payment.reference or "").strip()
    inv_no = (invoice.invoice_no or "").strip()
    if ref and inv_no:
        if inv_no in ref:
            reference_fit = Decimal("1")
        else:
            ratio = fuzz.partial_ratio(inv_no, ref)
            reference_fit = Decimal("0.6") if ratio >= _REF_PARTIAL_THRESHOLD else Decimal("0")
    else:
        reference_fit = Decimal("0")

    # ------------------------------------------------------------------
    # fee_inferred (MYR)
    # Payment is *short* → gap = converted − pay_amount > 0.
    # Classify as bank fee if 0.2 % ≤ gap/converted ≤ 2 %.
    # ------------------------------------------------------------------
    fee_inferred = Decimal("0")
    if converted > Decimal("0"):
        gap_signed = converted - pay_amount   # positive = short
        gap_pct_signed = gap_signed / converted
        if FEE_MIN_PCT <= gap_pct_signed <= FEE_MAX_PCT:
            fee_myr = convert(gap_signed, payment.currency, _MYR, payment.payment_date)
            fee_inferred = _q2(fee_myr)

    # ------------------------------------------------------------------
    # composite_confidence
    # ------------------------------------------------------------------
    composite = _q4(
        WEIGHT_AMOUNT * amount_fit
        + WEIGHT_DATE * date_fit
        + WEIGHT_NAME * name_fit
        + WEIGHT_REFERENCE * reference_fit
    )

    return MatchScore(
        amount_fit=amount_fit,
        date_fit=date_fit,
        name_fit=name_fit,
        reference_fit=reference_fit,
        fee_inferred=fee_inferred,
        composite_confidence=composite,
    )


# ---------------------------------------------------------------------------
# 2. verify_with_hermes — LLM overlay on a scored pair
# ---------------------------------------------------------------------------
async def verify_with_hermes(
    invoice: Invoice,
    payment: Payment,
    score: MatchScore,
) -> VerifiedMatch:
    """Ask Hermes 4 for final_confidence, a narrative, and caveats.

    Uses strict JSON schema enforcement via response_format.  All monetary
    evidence is passed in plaintext so the model can reason about it.

    Args:
        invoice: Candidate invoice.
        payment: Incoming payment.
        score:   Pre-computed MatchScore from score_pair().

    Returns:
        VerifiedMatch with hermes_narrative and final_confidence populated.
    """
    fx_rate = get_rate(invoice.currency, payment.currency, payment.payment_date)

    prompt = (
        "You are a senior treasury reconciliation analyst. "
        "Determine whether the payment below matches the invoice, "
        "using the provided match scores as evidence.\n\n"
        "=== INVOICE ===\n"
        f"  ID:           {invoice.id}\n"
        f"  Invoice No:   {invoice.invoice_no}\n"
        f"  Amount:       {invoice.amount} {invoice.currency}\n"
        f"  Issued:       {invoice.issued_date}\n"
        f"  Due:          {invoice.due_date}\n"
        f"  Status:       {invoice.status}\n\n"
        "=== PAYMENT ===\n"
        f"  ID:           {payment.id}\n"
        f"  Amount:       {payment.raw_amount} {payment.currency}\n"
        f"  Date:         {payment.payment_date}\n"
        f"  Sender:       {payment.sender_name}\n"
        f"  Reference:    {payment.reference}\n\n"
        "=== MATCH SCORES (all 0–1) ===\n"
        f"  amount_fit:    {score.amount_fit}\n"
        f"  date_fit:      {score.date_fit}\n"
        f"  name_fit:      {score.name_fit}\n"
        f"  reference_fit: {score.reference_fit}\n"
        f"  fee_inferred:  {score.fee_inferred} MYR\n"
        f"  composite:     {score.composite_confidence}\n"
        f"  fx_rate_used:  {fx_rate} "
        f"({invoice.currency} → {payment.currency})\n\n"
        "Respond with JSON containing:\n"
        "  final_confidence (integer 0–100)\n"
        "  narrative (one-paragraph English reasoning)\n"
        "  caveats (list of concerns)\n"
    )

    client = _async_client()
    resp = await client.chat.completions.create(
        model=_model(),
        messages=[{"role": "user", "content": prompt}],
        response_format=_VERIFY_SCHEMA,  # type: ignore[arg-type]
        temperature=0,
    )

    data = json.loads(resp.choices[0].message.content)

    return VerifiedMatch(
        invoice_id=invoice.id,           # type: ignore[arg-type]
        payment_id=payment.id,           # type: ignore[arg-type]
        score=score,
        final_confidence=int(data["final_confidence"]),
        hermes_narrative=data["narrative"],
        caveats=data["caveats"],
        fx_rate_used=fx_rate,
    )


async def _explain_below_winner(
    candidate: VerifiedMatch,
    winner: VerifiedMatch,
) -> str:
    """Ask Hermes why candidate ranked below winner; returns one paragraph."""
    prompt = (
        "Two invoices were scored against the same payment.\n\n"
        f"WINNER  (invoice {winner.invoice_id}, confidence {winner.final_confidence}):\n"
        f"  {winner.hermes_narrative[:600]}\n\n"
        f"CANDIDATE  (invoice {candidate.invoice_id}, confidence {candidate.final_confidence}):\n"
        f"  {candidate.hermes_narrative[:600]}\n\n"
        f"In one paragraph, explain specifically why invoice {candidate.invoice_id} "
        f"ranks below invoice {winner.invoice_id}."
    )

    client = _async_client()
    resp = await client.chat.completions.create(
        model=_model(),
        messages=[{"role": "user", "content": prompt}],
        response_format=_RANK_EXPLAIN_SCHEMA,  # type: ignore[arg-type]
        temperature=0,
    )
    data = json.loads(resp.choices[0].message.content)
    return data["explanation"]


# ---------------------------------------------------------------------------
# 3. rank_candidates — end-to-end pipeline for one payment
# ---------------------------------------------------------------------------
async def rank_candidates(
    payment: Payment,
    candidates: list[Invoice],
    counterparties: Optional[dict[int, Counterparty]] = None,
) -> list[VerifiedMatch]:
    """Score all candidates, verify top-5 with Hermes, return sorted list.

    Steps:
    1. Deterministically score every candidate via score_pair().
    2. Take the top _TOP_K by composite_confidence.
    3. Call verify_with_hermes on all top candidates in parallel.
    4. Sort descending by final_confidence.
    5. For candidates at positions 2 and 3 (0-indexed 1 and 2),
       append a Hermes explanation of why they rank below the winner.

    Args:
        payment:        The payment to reconcile.
        candidates:     All candidate invoices for this payment.
        counterparties: Optional map of customer_id → Counterparty, used for
                        name matching.  Pass None to skip name scoring.

    Returns:
        list[VerifiedMatch] sorted descending by final_confidence.
    """
    if not candidates:
        return []

    cp_map: dict[int, Counterparty] = counterparties or {}

    # --- Deterministic scoring -------------------------------------------
    scored: list[tuple[Decimal, Invoice, MatchScore]] = []
    for inv in candidates:
        cp = cp_map.get(inv.customer_id) if inv.customer_id is not None else None
        ms = score_pair(inv, payment, cp)
        scored.append((ms.composite_confidence, inv, ms))

    # Top _TOP_K by composite
    top_k = sorted(scored, key=lambda t: t[0], reverse=True)[:_TOP_K]

    # --- Parallel LLM verification ----------------------------------------
    verify_tasks = [verify_with_hermes(inv, payment, ms) for _, inv, ms in top_k]
    results: list[VerifiedMatch] = list(await asyncio.gather(*verify_tasks))

    # --- Sort by final_confidence -----------------------------------------
    results.sort(key=lambda vm: vm.final_confidence, reverse=True)

    # --- Explain positions 2 and 3 ----------------------------------------
    if len(results) >= 2:
        winner = results[0]
        subordinates = results[1:3]
        explanations: list[str] = list(
            await asyncio.gather(
                *[_explain_below_winner(vm, winner) for vm in subordinates]
            )
        )
        for vm, extra in zip(subordinates, explanations):
            vm.hermes_narrative = (
                vm.hermes_narrative + "\n\n[Why ranked below winner] " + extra
            )

    return results
