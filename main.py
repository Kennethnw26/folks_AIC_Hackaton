from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from sqlmodel import Session, select

from config import settings
from db.session import create_db_and_tables, engine
from db.models import Invoice, Payment, Tenant, BeneficiaryHistory, VendorDomain
from seed import run_seed
from tools.fx import load_supported
from webhooks.whatsapp import router as whatsapp_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db_and_tables()
    run_seed()
    await load_supported()
    yield


app = FastAPI(
    title="Global Treasury Agent",
    version="1.0.0",
    description="WhatsApp AI agent for cross-border payment reconciliation and BEC fraud detection.",
    lifespan=lifespan,
)

app.include_router(whatsapp_router)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "version": app.version}


# ---------------------------------------------------------------------------
# Dev-only endpoints
# ---------------------------------------------------------------------------
if settings.env == "dev":
    from pydantic import BaseModel as _BaseModel

    class _SimulateImageRequest(_BaseModel):
        image_b64: str
        sender: str = "demo_user"

    @app.post("/dev/simulate_image")
    async def dev_simulate_image(req: _SimulateImageRequest) -> dict:
        import base64
        from tools.ocr import extract_proof
        import agents.matching_agent as matching_agent
        import agents.fraud_agent as fraud_agent
        from tools.composer import build_reply

        image_b64 = req.image_b64
        image_bytes = base64.b64decode(image_b64)
        proof = await extract_proof(image_bytes)
        if proof is None:
            return {"error": "OCR failed"}

        proof_dict = {
            "amount": proof.amount,
            "currency": proof.currency,
            "date": str(proof.date),
            "beneficiary_name": proof.beneficiary_name,
            "bank_ref": proof.bank_ref,
        }

        with Session(engine) as session:
            tenant = session.exec(select(Tenant)).first()
            tenant_id = tenant.id if tenant else 1

        llm_budget = [1]

        # Use separate sessions per agent to avoid SQLite threading conflicts
        with Session(engine) as match_session:
            match_result = await matching_agent.run(proof_dict, tenant_id, match_session, llm_budget)

        with Session(engine) as fraud_session:
            fraud_assessment = await fraud_agent.run(proof_dict, "", tenant_id, fraud_session, llm_budget)

        if match_result is None:
            return {"error": "No matching invoice found", "proof": proof.model_dump(mode="json")}

        reply = build_reply(match_result, fraud_assessment)
        return {
            "proof": proof.model_dump(mode="json"),
            "match": {
                "invoice_id": match_result.invoice_id,
                "invoice_number": match_result.invoice_number,
                "score": round(match_result.score, 4),
                "source": match_result.source,
            },
            "fraud": {
                "score": round(fraud_assessment.score, 4),
                "narrative": fraud_assessment.narrative,
                "signals": {k: round(v, 4) for k, v in fraud_assessment.signals.items()},
            },
            "reply_payload": reply,
        }

    @app.get("/dev/state")
    def dev_state() -> dict:
        with Session(engine) as session:
            return {
                "tenants": [t.model_dump() for t in session.exec(select(Tenant)).all()],
                "invoices": [i.model_dump() for i in session.exec(select(Invoice)).all()],
                "payments": [p.model_dump() for p in session.exec(select(Payment)).all()],
                "beneficiary_history": [b.model_dump() for b in session.exec(select(BeneficiaryHistory)).all()],
                "vendor_domains": [v.model_dump() for v in session.exec(select(VendorDomain)).all()],
            }
