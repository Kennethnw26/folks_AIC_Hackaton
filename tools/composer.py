from __future__ import annotations

from agents.matching_agent import MatchResult
from agents.fraud_agent import FraudAssessment


def build_reply(match: MatchResult, fraud: FraudAssessment) -> dict:
    if fraud.score < 0.3:
        risk_emoji, risk_label = "\U0001f7e2", "LOW"
    elif fraud.score < 0.6:
        risk_emoji, risk_label = "\U0001f7e1", "MEDIUM"
    else:
        risk_emoji, risk_label = "\U0001f534", "HIGH"

    body = (
        f"\U0001f4b3 Match: Invoice #{match.invoice_number}\n"
        f"Amount: {match.amount} {match.currency}\n"
        f"Confidence: {int(match.score * 100)}%\n\n"
        f"{risk_emoji} Risk: {risk_label}"
    )
    if fraud.narrative:
        body += f"\n⚠️ {fraud.narrative}"

    return {
        "type": "interactive",
        "interactive": {
            "type": "button",
            "body": {"text": body},
            "action": {
                "buttons": [
                    {"type": "reply", "reply": {"id": f"confirm_{match.invoice_id}", "title": "Confirm Match"}},
                    {"type": "reply", "reply": {"id": f"flag_{match.invoice_id}", "title": "Flag Fraud"}},
                    {"type": "reply", "reply": {"id": "skip", "title": "Skip"}},
                ]
            },
        },
    }
