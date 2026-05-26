from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass

from sqlmodel import Session, select

from db.models import Invoice, Payment, Tenant
from schemas.llm_responses import ArbitrationResponse
from tools.chutes_client import MODEL_MATCH, call_chutes
from tools.fx import get_rate
from tools.matcher import CandidateScore, compute_det_score

logger = logging.getLogger(__name__)

_DET_THRESHOLD = 0.85
_TOP_K = 5


@dataclass
class MatchResult:
    invoice_id: int
    invoice_number: str
    amount: float
    currency: str
    score: float
    source: str  # "deterministic" | "llm"


async def run(proof_dict: dict, tenant_id: int, session: Session, llm_budget: list[int]) -> MatchResult | None:
    invoices = session.exec(
        select(Invoice).where(Invoice.tenant_id == tenant_id, Invoice.status == "open")
    ).all()
    if not invoices:
        return None

    tenant = session.get(Tenant, tenant_id)
    home_currency = tenant.home_currency if tenant else "MYR"
    p_date = str(proof_dict["date"])

    p_norm_rate = await get_rate(proof_dict["currency"], home_currency, p_date)
    p_norm = proof_dict["amount"] * (p_norm_rate or 1.0)

    # Normalize invoice amounts in parallel
    rates = await asyncio.gather(*[
        get_rate(inv.currency, home_currency, p_date)
        for inv in invoices
    ])

    candidates: list[CandidateScore] = []
    for inv, rate in zip(invoices, rates):
        i_norm = inv.amount * (rate or 1.0)
        # Build a temporary Payment-like object for scoring
        class _P:
            currency = proof_dict["currency"]
            date = proof_dict["date"]
            beneficiary_name = proof_dict["beneficiary_name"]
        det = compute_det_score(_P(), inv, p_norm, i_norm)
        candidates.append(CandidateScore(
            invoice_id=inv.id,
            invoice_number=inv.invoice_number,
            amount=inv.amount,
            currency=inv.currency,
            beneficiary_name=inv.beneficiary_name,
            det_score=det,
        ))

    candidates.sort(key=lambda c: c.det_score, reverse=True)
    top = candidates[0]

    if top.det_score >= _DET_THRESHOLD:
        return MatchResult(
            invoice_id=top.invoice_id,
            invoice_number=top.invoice_number,
            amount=top.amount,
            currency=top.currency,
            score=top.det_score,
            source="deterministic",
        )

    # LLM arbitration
    if llm_budget[0] >= 3:
        return MatchResult(
            invoice_id=top.invoice_id,
            invoice_number=top.invoice_number,
            amount=top.amount,
            currency=top.currency,
            score=top.det_score,
            source="deterministic_budget_exhausted",
        )

    llm_budget[0] += 1
    top5 = candidates[:_TOP_K]
    payment_json = json.dumps(proof_dict, default=str)
    candidates_json = json.dumps([
        {"invoice_id": c.invoice_id, "invoice_number": c.invoice_number,
         "amount": c.amount, "currency": c.currency,
         "beneficiary_name": c.beneficiary_name, "det_score": c.det_score}
        for c in top5
    ])

    prompt = (
        "You are a payment-invoice matching arbiter. Score each candidate invoice 0.0-1.0 "
        "for likelihood of matching this payment. Consider amount proximity (after FX normalization), "
        "beneficiary name similarity, reference patterns, date alignment.\n\n"
        f"Payment:\n{payment_json}\n\n"
        f"Candidates (top 5 by deterministic score):\n{candidates_json}\n\n"
        'Return STRICT JSON only:\n'
        '{"scores": [{"invoice_id": <int>, "score": <float 0..1>}, ...], "reasoning": "<one sentence>"}'
    )

    arb: ArbitrationResponse | None = await call_chutes(
        model=MODEL_MATCH,
        messages=[{"role": "user", "content": prompt}],
        response_model=ArbitrationResponse,
    )

    if arb is None:
        return MatchResult(
            invoice_id=top.invoice_id,
            invoice_number=top.invoice_number,
            amount=top.amount,
            currency=top.currency,
            score=top.det_score,
            source="deterministic_llm_failed",
        )

    score_map = {s.invoice_id: s.score for s in arb.scores}
    best_candidate = top
    best_final = 0.0
    for c in top5:
        llm_s = score_map.get(c.invoice_id, 0.0)
        final = 0.4 * c.det_score + 0.6 * llm_s
        if final > best_final:
            best_final = final
            best_candidate = c

    return MatchResult(
        invoice_id=best_candidate.invoice_id,
        invoice_number=best_candidate.invoice_number,
        amount=best_candidate.amount,
        currency=best_candidate.currency,
        score=best_final,
        source="llm",
    )
