from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from math import prod

import networkx as nx
import tldextract
from rapidfuzz.distance import Levenshtein
from sqlmodel import Session, select

from db.models import BeneficiaryHistory, Payment, VendorDomain
from schemas.llm_responses import NarrativeResponse, UrgencyResponse
from tools.chutes_client import MODEL_FRAUD, call_chutes

logger = logging.getLogger(__name__)


@dataclass
class FraudAssessment:
    score: float
    narrative: str | None
    signals: dict[str, float]


# ---------------------------------------------------------------------------
# Signal 1: new beneficiary
# ---------------------------------------------------------------------------
def _signal_new_beneficiary(account_number: str, tenant_id: int, session: Session) -> float:
    existing = session.exec(
        select(BeneficiaryHistory).where(
            BeneficiaryHistory.tenant_id == tenant_id,
            BeneficiaryHistory.account_number == account_number,
        )
    ).first()
    return 0.0 if existing else 1.0


# ---------------------------------------------------------------------------
# Signal 2: domain spoof
# ---------------------------------------------------------------------------
def _signal_domain_spoof(raw_text: str, tenant_id: int, session: Session) -> float:
    import re
    email_match = re.search(r"[\w.+-]+@([\w.-]+\.[a-zA-Z]{2,})", raw_text)
    if not email_match:
        return 0.0

    observed = tldextract.extract(email_match.group(1)).registered_domain
    if not observed:
        return 0.0

    vendor_domains = session.exec(
        select(VendorDomain).where(VendorDomain.tenant_id == tenant_id)
    ).all()

    min_dist = None
    for vd in vendor_domains:
        vd_registered = tldextract.extract(vd.domain).registered_domain or vd.domain
        dist = Levenshtein.distance(observed, vd_registered)
        if min_dist is None or dist < min_dist:
            min_dist = dist

    if min_dist is None:
        return 0.3
    if min_dist == 0:
        return 0.0
    if 1 <= min_dist <= 2:
        return 0.9
    return 0.3


# ---------------------------------------------------------------------------
# Signal 3: trust graph
# ---------------------------------------------------------------------------
def _signal_trust_graph(beneficiary_name: str, tenant_id: int, session: Session) -> float:
    confirmed = session.exec(
        select(Payment).where(
            Payment.tenant_id == tenant_id,
            Payment.status == "confirmed",
        )
    ).all()

    G = nx.DiGraph()
    for p in confirmed:
        G.add_edge(f"tenant_{tenant_id}", p.beneficiary_name,
                   weight=G[f"tenant_{tenant_id}"].get(p.beneficiary_name, {}).get("weight", 0) + 1
                   if G.has_edge(f"tenant_{tenant_id}", p.beneficiary_name) else 1)

    if not G.has_edge(f"tenant_{tenant_id}", beneficiary_name):
        return 0.8
    weight = G[f"tenant_{tenant_id}"][beneficiary_name]["weight"]
    if weight <= 2:
        return 0.4
    return 0.1


# ---------------------------------------------------------------------------
# Signal 4: urgency anomaly (LLM)
# ---------------------------------------------------------------------------
async def _signal_urgency_anomaly(raw_text: str, llm_budget: list[int]) -> float:
    if llm_budget[0] >= 3:
        return 0.0

    llm_budget[0] += 1
    prompt = (
        "Analyze this message for Business Email Compromise (BEC) urgency manipulation: "
        "deadline pressure, secrecy demands, executive impersonation (\"CEO needs this now\"), "
        "unusual payment route changes, threats, weekend/holiday urgency.\n\n"
        f"Message:\n{raw_text}\n\n"
        'Return STRICT JSON only:\n'
        '{"urgency_score": <float 0..1>, "indicators": ["<short phrase>", ...]}'
    )

    result: UrgencyResponse | None = await call_chutes(
        model=MODEL_FRAUD,
        messages=[{"role": "user", "content": prompt}],
        response_model=UrgencyResponse,
    )
    return result.urgency_score if result else 0.0


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------
def _noisy_or(scores: list[float]) -> float:
    return 1.0 - prod(1.0 - s for s in scores)


async def run(
    proof_dict: dict,
    raw_text: str,
    tenant_id: int,
    session: Session,
    llm_budget: list[int],
) -> FraudAssessment:
    account_number = proof_dict.get("bank_ref", "") or proof_dict.get("beneficiary_name", "")
    beneficiary_name = proof_dict.get("beneficiary_name", "")

    s1, s2, s3, s4 = await asyncio.gather(
        asyncio.to_thread(_signal_new_beneficiary, account_number, tenant_id, session),
        asyncio.to_thread(_signal_domain_spoof, raw_text, tenant_id, session),
        asyncio.to_thread(_signal_trust_graph, beneficiary_name, tenant_id, session),
        _signal_urgency_anomaly(raw_text, llm_budget),
    )

    signals = {
        "new_beneficiary": s1,
        "domain_spoof": s2,
        "trust_graph": s3,
        "urgency_anomaly": s4,
    }
    aggregate = _noisy_or(list(signals.values()))

    narrative: str | None = None
    if aggregate > 0.3 and llm_budget[0] < 3:
        llm_budget[0] += 1
        signals_json = json.dumps(signals)
        prompt = (
            "Generate ONE sentence (max 20 words) explaining why this payment was flagged. "
            "Reference the strongest fired signals concretely.\n\n"
            f"Signal breakdown:\n{signals_json}\n\n"
            'Return STRICT JSON only:\n{"narrative": "<one sentence>"}'
        )
        result: NarrativeResponse | None = await call_chutes(
            model=MODEL_FRAUD,
            messages=[{"role": "user", "content": prompt}],
            response_model=NarrativeResponse,
        )
        if result:
            narrative = result.narrative

    return FraudAssessment(score=aggregate, narrative=narrative, signals=signals)
