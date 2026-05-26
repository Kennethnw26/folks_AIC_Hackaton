from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from sqlmodel import Session, select

from db.models import (
    BeneficiaryHistory,
    Invoice,
    MessageIdempotency,
    OcrRetryState,
    Payment,
    Tenant,
)
from db.session import engine

logger = logging.getLogger(__name__)


def _get_demo_tenant_id(session: Session) -> int:
    tenant = session.exec(select(Tenant)).first()
    return tenant.id if tenant else 1


async def handle_message(envelope: dict) -> None:
    message_id = envelope.get("message_id", "")
    sender = envelope.get("from", "")

    with Session(engine) as session:
        # Idempotency check
        if message_id and session.get(MessageIdempotency, message_id):
            return
        if message_id:
            session.add(MessageIdempotency(message_id=message_id, processed_at=datetime.utcnow()))
            session.commit()

        msg_type = envelope.get("type", "")

        if msg_type == "image":
            await _run_ocr_pipeline(envelope, sender, session)
        elif msg_type == "interactive":
            await _handle_button_response(envelope, sender, session)
        elif msg_type == "text" and envelope.get("text", "").startswith("/report"):
            await _run_fraud_only(envelope, sender, session)
        elif msg_type == "text":
            await _handle_text_followup(envelope, sender, session)


async def _run_ocr_pipeline(envelope: dict, sender: str, session: Session) -> None:
    from tools.ocr import extract_proof
    from tools.whatsapp_client import send_text, send_interactive
    from tools.composer import build_reply
    import agents.matching_agent as matching_agent
    import agents.fraud_agent as fraud_agent

    media_id = envelope.get("image", {}).get("id", "")
    if not media_id:
        return

    from tools.whatsapp_client import download_media
    try:
        image_bytes = await download_media(media_id)
    except Exception as e:
        logger.error("Media download failed: %s", e)
        await send_text(sender, "Could not download your image. Please try again.")
        return

    proof = await extract_proof(image_bytes)

    if proof is None or proof.confidence < 0.7:
        tenant_id = _get_demo_tenant_id(session)
        session.add(OcrRetryState(
            tenant_id=tenant_id,
            whatsapp_user=sender,
            expected_at=datetime.utcnow(),
            raw_proof_partial="{}",
        ))
        session.commit()
        await send_text(sender, "Image unclear, please resend a clearer screenshot of the payment proof")
        return

    proof_dict = {
        "amount": proof.amount,
        "currency": proof.currency,
        "date": proof.date,
        "beneficiary_name": proof.beneficiary_name,
        "bank_ref": proof.bank_ref,
    }

    tenant_id = _get_demo_tenant_id(session)
    llm_budget = [1]  # OCR already used 1

    match_result, fraud_assessment = await asyncio.gather(
        matching_agent.run(proof_dict, tenant_id, session, llm_budget),
        fraud_agent.run(proof_dict, "", tenant_id, session, llm_budget),
    )

    if match_result is None:
        await send_text(sender, "No open invoices found to match against.")
        return

    # Persist payment
    payment = Payment(
        tenant_id=tenant_id,
        amount=proof.amount,
        currency=proof.currency,
        date=proof.date,
        beneficiary_name=proof.beneficiary_name,
        bank_ref=proof.bank_ref,
        fraud_score=fraud_assessment.score,
        status="pending",
    )
    session.add(payment)
    session.commit()
    session.refresh(payment)

    reply = build_reply(match_result, fraud_assessment)
    await send_interactive(sender, reply)


async def _handle_button_response(envelope: dict, sender: str, session: Session) -> None:
    from tools.whatsapp_client import send_text

    interactive = envelope.get("interactive", {})
    button_id: str = interactive.get("button_reply", {}).get("id", "")

    if button_id.startswith("confirm_"):
        invoice_id = int(button_id.split("_", 1)[1])
        invoice = session.get(Invoice, invoice_id)
        if invoice:
            invoice.status = "closed"
            # Find latest pending payment for sender (simple heuristic)
            payments = session.exec(
                select(Payment).where(Payment.status == "pending")
                .order_by(Payment.id.desc())
            ).all()
            for p in payments:
                if p.matched_invoice_id is None:
                    p.matched_invoice_id = invoice_id
                    p.status = "confirmed"
                    # Add to beneficiary history if new
                    existing = session.exec(
                        select(BeneficiaryHistory).where(
                            BeneficiaryHistory.tenant_id == p.tenant_id,
                            BeneficiaryHistory.account_number == p.bank_ref,
                        )
                    ).first()
                    if not existing:
                        session.add(BeneficiaryHistory(
                            tenant_id=p.tenant_id,
                            account_number=p.bank_ref,
                            beneficiary_name=p.beneficiary_name,
                            first_seen=datetime.utcnow(),
                        ))
                    break
            session.commit()
        await send_text(sender, f"✅ Invoice #{invoice_id} confirmed and closed.")

    elif button_id.startswith("flag_"):
        invoice_id = int(button_id.split("_", 1)[1])
        payments = session.exec(
            select(Payment).where(Payment.status == "pending").order_by(Payment.id.desc())
        ).all()
        for p in payments:
            if p.matched_invoice_id is None:
                p.status = "flagged"
                session.commit()
                break
        await send_text(sender, f"🚩 Payment flagged for manual review.")

    elif button_id == "skip":
        payments = session.exec(
            select(Payment).where(Payment.status == "pending").order_by(Payment.id.desc())
        ).all()
        for p in payments:
            if p.matched_invoice_id is None:
                p.status = "skipped"
                session.commit()
                break
        await send_text(sender, "Payment skipped.")


async def _run_fraud_only(envelope: dict, sender: str, session: Session) -> None:
    from tools.whatsapp_client import send_text
    import agents.fraud_agent as fraud_agent

    text = envelope.get("text", "")
    tenant_id = _get_demo_tenant_id(session)
    llm_budget = [0]

    proof_dict = {"amount": 0, "currency": "MYR", "date": None, "beneficiary_name": "", "bank_ref": ""}
    assessment = await fraud_agent.run(proof_dict, text, tenant_id, session, llm_budget)

    risk = "LOW" if assessment.score < 0.3 else "MEDIUM" if assessment.score < 0.6 else "HIGH"
    msg = f"Fraud assessment: {risk} ({assessment.score:.0%})"
    if assessment.narrative:
        msg += f"\n{assessment.narrative}"
    await send_text(sender, msg)


async def _handle_text_followup(envelope: dict, sender: str, session: Session) -> None:
    from tools.whatsapp_client import send_text
    retry = session.exec(
        select(OcrRetryState).where(OcrRetryState.whatsapp_user == sender)
        .order_by(OcrRetryState.id.desc())
    ).first()
    if retry:
        await send_text(sender, "Please send a payment screenshot image to proceed.")
    else:
        await send_text(sender, "Send a payment screenshot to begin reconciliation, or /report <text> to run fraud-only analysis.")
