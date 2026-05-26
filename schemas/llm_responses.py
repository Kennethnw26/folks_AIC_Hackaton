from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


class PaymentProof(BaseModel):
    amount: float
    currency: str = Field(min_length=3, max_length=3)
    date: date
    beneficiary_name: str
    bank_ref: str = ""
    confidence: float = Field(ge=0.0, le=1.0)


class ArbitrationScore(BaseModel):
    invoice_id: int
    score: float = Field(ge=0.0, le=1.0)


class ArbitrationResponse(BaseModel):
    scores: list[ArbitrationScore]
    reasoning: str


class UrgencyResponse(BaseModel):
    urgency_score: float = Field(ge=0.0, le=1.0)
    indicators: list[str]


class NarrativeResponse(BaseModel):
    narrative: str
