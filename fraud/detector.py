"""BEC fraud detection layer for the Global Treasury Agent.

Implements four orthogonal fraud signals and a noisy-OR aggregator that
returns a FraudAssessment for downstream agent display. The detector
NEVER auto-blocks — it surfaces a recommendation; the user takes the
final action.

Signals
-------
1. signal_new_beneficiary  — payment routed to a destination account
   never seen in this counterparty's recent payment history.
2. signal_domain_spoof     — instruction email's sender domain differs
   from the counterparty's verified domain (typosquat, TLD swap,
   hyphenation, or no similarity).
3. signal_trust_graph      — destination account is present in the
   FlaggedAccount table (cross-organization fraud reports).
4. signal_urgency_anomaly  — LLM-judged urgency anomaly vs the
   counterparty's normal communication style.

Aggregator: assess_fraud uses noisy-OR (1 - prod(1 - score)) and a
Hermes-generated narrative.

Thresholds
----------
fraud_confidence >= 75  → "block"
fraud_confidence 40-74  → "verify"
fraud_confidence  < 40  → "proceed"
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from datetime import datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from email.utils import parseaddr
from typing import Any, Literal, Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from rapidfuzz.distance import Levenshtein
from sqlmodel import Session, select

from db.models import Counterparty, FlaggedAccount, Match, Payment

load_dotenv()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------
_HISTORY_DEPTH_LIMIT: int = 10
_HISTORY_DEPTH_STRONG: int = 3

_LEVENSHTEIN_TYPOSQUAT_MAX: int = 3
_TRUST_GRAPH_BASE: Decimal = Decimal("0.4")
_TRUST_GRAPH_PER_REPORTER: Decimal = Decimal("0.2")
_TRUST_GRAPH_CAP: Decimal = Decimal("0.95")
_TRUST_GRAPH_RECENCY_DAYS: int = 30
_TRUST_GRAPH_RECENCY_BOOST: Decimal = Decimal("0.1")

_THRESHOLD_BLOCK: int = 75
_THRESHOLD_VERIFY: int = 40

_RecommendedAction = Literal["proceed", "verify", "block"]

_DP4 = Decimal("0.0001")


# ---------------------------------------------------------------------------
# Pydantic output models
# ---------------------------------------------------------------------------
class SignalEvidence(BaseModel):
    """One fraud signal's score plus its supporting evidence."""

    signal_name: str
    score: Decimal = Field(ge=Decimal("0"), le=Decimal("1"))
    evidence: dict[str, Any]
    explanation: str


class FraudAssessment(BaseModel):
    """Aggregate fraud assessment for one payment."""

    payment_id: int
    fraud_confidence: int = Field(ge=0, le=100)
    signals: list[SignalEvidence]
    narrative: str
    recommended_action: _RecommendedAction


# ---------------------------------------------------------------------------
# JSON-schema contracts for Hermes
# ---------------------------------------------------------------------------
_URGENCY_SCHEMA: dict = {
    "type": "json_schema",
    "json_schema": {
        "name": "urgency_anomaly",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "score": {
                    "type": "number",
                    "description": (
                        "Urgency-anomaly score 0-1. 0 = perfectly consistent "
                        "with this counterparty's normal communication style. "
                        "1 = extreme deviation (e.g. unusual urgency, threats, "
                        "secrecy demands, off-hours pressure)."
                    ),
                },
                "explanation": {
                    "type": "string",
                    "description": "One-sentence justification.",
                },
            },
            "required": ["score", "explanation"],
            "additionalProperties": False,
        },
    },
}

_NARRATIVE_SCHEMA: dict = {
    "type": "json_schema",
    "json_schema": {
        "name": "fraud_narrative",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "narrative": {
                    "type": "string",
                    "description": (
                        "One paragraph English explanation of the decision, "
                        "naming each signal that fired and what it implies. "
                        "Plain, non-technical, suitable for a small-business "
                        "owner on WhatsApp."
                    ),
                },
            },
            "required": ["narrative"],
            "additionalProperties": False,
        },
    },
}


# ---------------------------------------------------------------------------
# Hermes client helpers
# ---------------------------------------------------------------------------
def _async_client() -> AsyncOpenAI:
    api_key = os.getenv("CHUTES_API_KEY", "")
    base_url = os.getenv("CHUTES_BASE_URL", "https://llm.chutes.ai/v1")
    return AsyncOpenAI(api_key=api_key, base_url=base_url)


def _model() -> str:
    return os.getenv("CHUTES_MODEL", "NousResearch/Hermes-4-405B")


def _q4(d: Decimal) -> Decimal:
    return d.quantize(_DP4, rounding=ROUND_HALF_UP)


def _clamp01(d: Decimal) -> Decimal:
    if d < Decimal("0"):
        return Decimal("0")
    if d > Decimal("1"):
        return Decimal("1")
    return d


# ---------------------------------------------------------------------------
# Signal 1: new beneficiary
# ---------------------------------------------------------------------------
def signal_new_beneficiary(
    payment: Payment,
    counterparty: Counterparty,
    db: Session,
) -> SignalEvidence:
    """Flag payments routed to an unseen destination account.

    Looks at the last _HISTORY_DEPTH_LIMIT confirmed payments to this
    counterparty (via the Match table) and checks whether the current
    destination_account appears in the historical set.
    """
    # Find confirmed matches whose invoice belongs to this counterparty.
    # Then pull the linked payments. We sort by confirmed_at desc and
    # cap at the history limit.
    stmt = (
        select(Payment)
        .join(Match, Match.payment_id == Payment.id)
        .where(Match.status == "confirmed")
        .where(Payment.sender_name == counterparty.display_name)
        .order_by(Match.confirmed_at.desc() if Match.confirmed_at is not None else Match.id.desc())
        .limit(_HISTORY_DEPTH_LIMIT)
    )
    historical_payments: list[Payment] = list(db.exec(stmt).all())

    distinct_accounts = sorted({
        p.destination_account
        for p in historical_payments
        if p.destination_account and p.id != payment.id
    })
    history_depth = len(distinct_accounts)
    current_account = payment.destination_account
    is_new = current_account not in distinct_accounts

    if history_depth == 0:
        score = Decimal("0.2")
        explanation = (
            "No prior payment history for this counterparty — score is "
            "low to avoid over-firing on first-time payments."
        )
    elif history_depth < _HISTORY_DEPTH_STRONG:
        if is_new:
            score = Decimal("0.5")
            explanation = (
                f"Destination account is new, but only {history_depth} "
                "historical accounts on file — moderate suspicion."
            )
        else:
            score = Decimal("0.0")
            explanation = (
                "Destination account matches a previously seen account."
            )
    else:
        if is_new:
            score = Decimal("0.95")
            explanation = (
                f"Destination account never seen in the last "
                f"{history_depth} payments to this counterparty — strong "
                "indicator of beneficiary substitution."
            )
        else:
            score = Decimal("0.0")
            explanation = (
                "Destination account matches a previously seen account."
            )

    evidence = {
        "old_accounts": distinct_accounts,
        "new_account": current_account,
        "history_depth": history_depth,
    }
    return SignalEvidence(
        signal_name="new_beneficiary",
        score=score,
        evidence=evidence,
        explanation=explanation,
    )


# Signal 2: domain spoof

def _parse_sender_domain(headers: dict[str, Any] | None) -> Optional[str]:
    """Extract the lower-cased sender domain from email headers."""
    if not headers:
        return None
    from_value = headers.get("From") or headers.get("from") or ""
    if not from_value:
        return None
    _, addr = parseaddr(from_value)
    if not addr or "@" not in addr:
        # Fall back to bare-domain detection.
        m = re.search(r"@([A-Za-z0-9.\-]+)", from_value)
        return m.group(1).lower() if m else None
    return addr.split("@", 1)[1].lower().strip()


def _strip_hyphens(s: str) -> str:
    return s.replace("-", "")


def _domain_parts(domain: str) -> tuple[str, str]:
    """Split domain into (label, tld) on first dot."""
    if "." not in domain:
        return domain, ""
    label, tld = domain.split(".", 1)
    return label, tld


def signal_domain_spoof(
    instruction_email_headers: dict[str, Any] | None,
    counterparty: Counterparty,
) -> SignalEvidence:
    """Compare instruction email sender domain to counterparty.verified_domain."""
    real_domain = (counterparty.verified_domain or "").lower().strip()
    observed_domain = _parse_sender_domain(instruction_email_headers)

    if not observed_domain or not real_domain:
        return SignalEvidence(
            signal_name="domain_spoof",
            score=Decimal("0.0"),
            evidence={
                "real_domain": real_domain or None,
                "observed_domain": observed_domain,
                "distance": None,
                "pattern": "no_headers_or_unverified",
            },
            explanation=(
                "No email headers available or counterparty has no "
                "verified domain on file — signal not evaluated."
            ),
        )

    # a. exact match
    if observed_domain == real_domain:
        return SignalEvidence(
            signal_name="domain_spoof",
            score=Decimal("0.0"),
            evidence={
                "real_domain": real_domain,
                "observed_domain": observed_domain,
                "distance": 0,
                "pattern": "exact_match",
            },
            explanation="Sender domain matches verified domain exactly.",
        )

    distance = Levenshtein.distance(observed_domain, real_domain)

    real_label, real_tld = _domain_parts(real_domain)
    obs_label, obs_tld = _domain_parts(observed_domain)

    real_label_nohyphen = _strip_hyphens(real_label)
    obs_label_nohyphen = _strip_hyphens(obs_label)

    # d. hyphenation differences (compare after stripping hyphens; TLD
    # may also differ — common combined-attack pattern)
    if (
        real_label_nohyphen == obs_label_nohyphen
        and real_label != obs_label
    ):
        return SignalEvidence(
            signal_name="domain_spoof",
            score=Decimal("0.9"),
            evidence={
                "real_domain": real_domain,
                "observed_domain": observed_domain,
                "distance": distance,
                "pattern": "hyphenation_swap",
            },
            explanation=(
                "Sender domain differs in hyphenation from the verified "
                "domain — a common BEC tactic."
            ),
        )

    # c. TLD swap (same label, different TLD)
    if real_label == obs_label and real_tld != obs_tld:
        return SignalEvidence(
            signal_name="domain_spoof",
            score=Decimal("0.85"),
            evidence={
                "real_domain": real_domain,
                "observed_domain": observed_domain,
                "distance": distance,
                "pattern": "tld_swap",
            },
            explanation=(
                f"Same brand label but different TLD "
                f"({real_tld!r} vs {obs_tld!r}) — likely lookalike domain."
            ),
        )

    # b. Levenshtein <= 3 → typosquat
    if distance <= _LEVENSHTEIN_TYPOSQUAT_MAX:
        return SignalEvidence(
            signal_name="domain_spoof",
            score=Decimal("0.9"),
            evidence={
                "real_domain": real_domain,
                "observed_domain": observed_domain,
                "distance": distance,
                "pattern": "typosquat",
            },
            explanation=(
                f"Sender domain is within {distance} edits of the "
                "verified domain — likely typosquat."
            ),
        )

    # e. no similarity
    return SignalEvidence(
        signal_name="domain_spoof",
        score=Decimal("0.3"),
        evidence={
            "real_domain": real_domain,
            "observed_domain": observed_domain,
            "distance": distance,
            "pattern": "no_similarity",
        },
        explanation=(
            "Sender domain bears no resemblance to the verified domain "
            "— suspicious but not a clear spoof."
        ),
    )


# ---------------------------------------------------------------------------
# Signal 3: trust graph
# ---------------------------------------------------------------------------
def signal_trust_graph(payment: Payment, db: Session) -> SignalEvidence:
    """Check whether the destination account is in the FlaggedAccount table."""
    stmt = select(FlaggedAccount).where(
        FlaggedAccount.account_number == payment.destination_account
    )
    flagged: Optional[FlaggedAccount] = db.exec(stmt).first()

    if flagged is None:
        return SignalEvidence(
            signal_name="trust_graph",
            score=Decimal("0.0"),
            evidence={
                "reporter_count": 0,
                "total_loss_estimate": 0.0,
                "first_flagged_at": None,
                "most_recent_report": None,
            },
            explanation=(
                "Destination account not present in the cross-organization "
                "fraud-report graph."
            ),
        )

    base = _TRUST_GRAPH_BASE + (
        _TRUST_GRAPH_PER_REPORTER * Decimal(flagged.reporter_count)
    )
    score = _clamp01(min(base, _TRUST_GRAPH_CAP))

    # Recency boost (we treat first_flagged_at as the most recent report
    # marker when no per-report table exists).
    now = datetime.utcnow()
    most_recent = flagged.first_flagged_at
    if most_recent is not None and (now - most_recent) <= timedelta(
        days=_TRUST_GRAPH_RECENCY_DAYS
    ):
        score = _clamp01(min(score + _TRUST_GRAPH_RECENCY_BOOST, _TRUST_GRAPH_CAP))
        recency_boosted = True
    else:
        recency_boosted = False

    explanation = (
        f"Destination account is flagged by {flagged.reporter_count} other "
        f"organizations (severity={flagged.severity})"
        + (" with a recent report" if recency_boosted else "")
        + "."
    )

    evidence = {
        "reporter_count": flagged.reporter_count,
        "total_loss_estimate": float(flagged.total_loss_estimate),
        "first_flagged_at": (
            flagged.first_flagged_at.isoformat()
            if flagged.first_flagged_at is not None
            else None
        ),
        "most_recent_report": (
            flagged.first_flagged_at.isoformat()
            if flagged.first_flagged_at is not None
            else None
        ),
        "severity": flagged.severity,
        "bank_name": flagged.bank_name,
    }
    return SignalEvidence(
        signal_name="trust_graph",
        score=score,
        evidence=evidence,
        explanation=explanation,
    )


# ---------------------------------------------------------------------------
# Signal 4: urgency anomaly (LLM)
# ---------------------------------------------------------------------------
async def signal_urgency_anomaly(
    instruction_text: str,
    counterparty: Counterparty,
) -> SignalEvidence:
    """Ask Hermes whether the instruction's urgency matches counterparty norms."""
    if not instruction_text:
        return SignalEvidence(
            signal_name="urgency_anomaly",
            score=Decimal("0.0"),
            evidence={
                "instruction_excerpt": "",
                "style_summary": None,
            },
            explanation="No instruction text provided — signal not evaluated.",
        )

    try:
        style_summary = (
            json.loads(counterparty.payment_history_summary_json)
            if counterparty.payment_history_summary_json
            else {}
        )
    except (TypeError, ValueError):
        style_summary = {}

    style_blob = json.dumps(style_summary, indent=2, default=str)
    excerpt = instruction_text.strip()
    if len(excerpt) > 2000:
        excerpt = excerpt[:2000] + "...[truncated]"

    prompt = (
        "You are a fraud analyst at a Malaysian SME bank. Compare the "
        "tone and urgency of the payment instruction below against the "
        "counterparty's normal communication style. Return a score "
        "between 0 and 1 where 0 means 'totally consistent with their "
        "normal style' and 1 means 'extreme deviation that resembles a "
        "BEC scam (urgency, secrecy, threats, off-hours pressure, "
        "instructions to change banking details quickly)'.\n\n"
        "=== COUNTERPARTY ===\n"
        f"Name: {counterparty.display_name}\n"
        f"Communication style summary (JSON):\n{style_blob}\n\n"
        "=== INSTRUCTION TEXT ===\n"
        f"{excerpt}\n\n"
        "Respond with a JSON object: {\"score\": <0-1>, "
        "\"explanation\": \"<one sentence>\"}."
    )

    client = _async_client()
    try:
        resp = await client.chat.completions.create(
            model=_model(),
            messages=[{"role": "user", "content": prompt}],
            response_format=_URGENCY_SCHEMA,  # type: ignore[arg-type]
            temperature=0,
        )
        data = json.loads(resp.choices[0].message.content)
        raw_score = Decimal(str(data["score"]))
        score = _clamp01(_q4(raw_score))
        explanation = str(data["explanation"])
    except Exception as exc:  # noqa: BLE001
        logger.warning("Urgency-anomaly LLM call failed: %s", exc)
        return SignalEvidence(
            signal_name="urgency_anomaly",
            score=Decimal("0.0"),
            evidence={
                "instruction_excerpt": excerpt[:200],
                "style_summary": style_summary,
                "llm_error": str(exc),
            },
            explanation=(
                "Urgency-anomaly LLM call failed; signal not evaluated."
            ),
        )

    return SignalEvidence(
        signal_name="urgency_anomaly",
        score=score,
        evidence={
            "instruction_excerpt": excerpt[:500],
            "style_summary": style_summary,
        },
        explanation=explanation,
    )


# ---------------------------------------------------------------------------
# Narrative generator
# ---------------------------------------------------------------------------
async def _generate_narrative(
    payment: Payment,
    counterparty: Counterparty,
    signals: list[SignalEvidence],
    fraud_confidence: int,
    recommended_action: str,
) -> str:
    """Ask Hermes to produce a plain-English narrative covering all signals."""
    signal_blocks = []
    for s in signals:
        signal_blocks.append(
            f"- {s.signal_name} (score {s.score}): {s.explanation}\n"
            f"  evidence: {json.dumps(s.evidence, default=str)}"
        )
    signal_text = "\n".join(signal_blocks)

    prompt = (
        "You are a fraud analyst writing for a Malaysian SME owner on "
        "WhatsApp. Summarize the assessment in ONE paragraph (3-5 short "
        "sentences). Reference each signal that fired (score > 0). State "
        "the recommended action. Do not include lists or bullet points.\n\n"
        "=== PAYMENT ===\n"
        f"  id: {payment.id}\n"
        f"  amount: {payment.raw_amount} {payment.currency}\n"
        f"  date: {payment.payment_date}\n"
        f"  sender: {payment.sender_name}\n"
        f"  destination_account: {payment.destination_account}\n\n"
        f"=== COUNTERPARTY ===\n"
        f"  display_name: {counterparty.display_name}\n"
        f"  verified_domain: {counterparty.verified_domain}\n\n"
        "=== SIGNALS ===\n"
        f"{signal_text}\n\n"
        f"=== AGGREGATE ===\n"
        f"  fraud_confidence: {fraud_confidence}/100\n"
        f"  recommended_action: {recommended_action}\n\n"
        "Respond with JSON: {\"narrative\": \"...\"}."
    )

    client = _async_client()
    try:
        resp = await client.chat.completions.create(
            model=_model(),
            messages=[{"role": "user", "content": prompt}],
            response_format=_NARRATIVE_SCHEMA,  # type: ignore[arg-type]
            temperature=0,
        )
        data = json.loads(resp.choices[0].message.content)
        return str(data["narrative"])
    except Exception as exc:  # noqa: BLE001
        logger.warning("Narrative LLM call failed: %s", exc)
        # Deterministic fallback so the agent can still respond.
        fired = [s for s in signals if s.score > Decimal("0")]
        reasons = "; ".join(f"{s.signal_name}={s.score}" for s in fired)
        return (
            f"Fraud confidence is {fraud_confidence}/100 — recommended "
            f"action: {recommended_action}. Signals that fired: "
            f"{reasons or 'none'}."
        )


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------
def _noisy_or(scores: list[Decimal]) -> Decimal:
    """Combine independent probabilities via noisy-OR: 1 - prod(1 - s)."""
    product = Decimal("1")
    for s in scores:
        product *= (Decimal("1") - _clamp01(s))
    return _clamp01(Decimal("1") - product)


def _recommend(fraud_confidence: int) -> _RecommendedAction:
    if fraud_confidence >= _THRESHOLD_BLOCK:
        return "block"
    if fraud_confidence >= _THRESHOLD_VERIFY:
        return "verify"
    return "proceed"


async def assess_fraud(
    payment: Payment,
    counterparty: Counterparty,
    instruction_text: str,
    email_headers: dict[str, Any] | None,
    db: Session,
) -> FraudAssessment:
    """Run all four signals, combine via noisy-OR, build a recommendation.

    The three synchronous signals are wrapped in asyncio.to_thread so
    they run concurrently with the LLM urgency-anomaly call.
    """
    sig_new = asyncio.to_thread(signal_new_beneficiary, payment, counterparty, db)
    sig_dom = asyncio.to_thread(signal_domain_spoof, email_headers, counterparty)
    sig_trust = asyncio.to_thread(signal_trust_graph, payment, db)
    sig_urg = signal_urgency_anomaly(instruction_text, counterparty)

    results = await asyncio.gather(sig_new, sig_dom, sig_trust, sig_urg)
    signals: list[SignalEvidence] = list(results)

    aggregate = _noisy_or([s.score for s in signals])
    fraud_confidence = int(_q4(aggregate * Decimal("100")).to_integral_value(
        rounding=ROUND_HALF_UP
    ))
    # Defensive clamp in case of FP drift
    fraud_confidence = max(0, min(100, fraud_confidence))

    recommended_action = _recommend(fraud_confidence)

    narrative = await _generate_narrative(
        payment=payment,
        counterparty=counterparty,
        signals=signals,
        fraud_confidence=fraud_confidence,
        recommended_action=recommended_action,
    )

    # NEVER auto-block: this assessment is for the agent to relay to the
    # user with action buttons; no DB mutation or block is performed here.
    return FraudAssessment(
        payment_id=payment.id if payment.id is not None else 0,
        fraud_confidence=fraud_confidence,
        signals=signals,
        narrative=narrative,
        recommended_action=recommended_action,
    )
