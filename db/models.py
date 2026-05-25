"""SQLModel data models for the Global Treasury Agent.

Every column that participates in lookups, joins, or filters is indexed.
JSON-typed payloads are stored as TEXT (SQLite) and serialized by the
calling code with json.dumps / json.loads.
"""
from __future__ import annotations

from datetime import datetime, date
from typing import Optional

from sqlmodel import Field, SQLModel


# ---------------------------------------------------------------------------
# Invoice
# ---------------------------------------------------------------------------
class Invoice(SQLModel, table=True):
    __tablename__ = "invoice"

    id: Optional[int] = Field(default=None, primary_key=True)
    invoice_no: str = Field(index=True, unique=True)
    customer_id: int = Field(index=True, foreign_key="counterparty.id")
    amount: float = Field(index=True)
    currency: str = Field(index=True, max_length=3)
    issued_date: date = Field(index=True)
    due_date: date = Field(index=True)
    status: str = Field(
        index=True,
        default="open",
        description="open | partially_paid | paid | written_off",
    )
    reference_hint: Optional[str] = Field(
        default=None,
        index=True,
        description="Free-text reference customers are expected to quote",
    )


# ---------------------------------------------------------------------------
# Payment
# ---------------------------------------------------------------------------
class Payment(SQLModel, table=True):
    __tablename__ = "payment"

    id: Optional[int] = Field(default=None, primary_key=True)
    raw_amount: float = Field(index=True)
    currency: str = Field(index=True, max_length=3)
    payment_date: date = Field(index=True)
    sender_name: str = Field(index=True)
    reference: Optional[str] = Field(default=None, index=True)
    destination_account: str = Field(index=True)
    source_image_path: Optional[str] = Field(default=None, index=True)
    status: str = Field(
        index=True,
        default="unreconciled",
        description="unreconciled | matched | flagged | reconciled",
    )


# ---------------------------------------------------------------------------
# Counterparty
# ---------------------------------------------------------------------------
class Counterparty(SQLModel, table=True):
    __tablename__ = "counterparty"

    id: Optional[int] = Field(default=None, primary_key=True)
    display_name: str = Field(index=True)
    verified_domain: Optional[str] = Field(default=None, index=True)
    normalized_name: str = Field(index=True)
    embedding_vec_json: Optional[str] = Field(
        default=None,
        description="JSON-encoded list[float] from sentence-transformers",
    )
    payment_history_summary_json: Optional[str] = Field(
        default=None,
        description="JSON blob: avg/median/recent payment behaviour",
    )


# ---------------------------------------------------------------------------
# Match
# ---------------------------------------------------------------------------
class Match(SQLModel, table=True):
    __tablename__ = "match"

    id: Optional[int] = Field(default=None, primary_key=True)
    invoice_id: int = Field(index=True, foreign_key="invoice.id")
    payment_id: int = Field(index=True, foreign_key="payment.id")
    confidence: float = Field(index=True)
    sub_scores_json: str = Field(
        description="JSON: amount / name / reference / date / history sub-scores"
    )
    hermes_narrative: Optional[str] = Field(default=None)
    fx_rate_used: Optional[float] = Field(default=None, index=True)
    fee_inferred: Optional[float] = Field(default=None, index=True)
    status: str = Field(
        index=True,
        default="proposed",
        description="proposed | confirmed | rejected | superseded",
    )
    confirmed_by: Optional[str] = Field(default=None, index=True)
    confirmed_at: Optional[datetime] = Field(default=None, index=True)


# ---------------------------------------------------------------------------
# FlaggedAccount
# ---------------------------------------------------------------------------
class FlaggedAccount(SQLModel, table=True):
    __tablename__ = "flagged_account"

    id: Optional[int] = Field(default=None, primary_key=True)
    account_number: str = Field(index=True, unique=True)
    bank_name: str = Field(index=True)
    first_flagged_at: datetime = Field(index=True, default_factory=datetime.utcnow)
    reporter_count: int = Field(index=True, default=1)
    total_loss_estimate: float = Field(index=True, default=0.0)
    severity: str = Field(
        index=True,
        default="medium",
        description="low | medium | high | critical",
    )


# ---------------------------------------------------------------------------
# FraudAlert
# ---------------------------------------------------------------------------
class FraudAlert(SQLModel, table=True):
    __tablename__ = "fraud_alert"

    id: Optional[int] = Field(default=None, primary_key=True)
    payment_id: int = Field(index=True, foreign_key="payment.id")
    fraud_confidence: float = Field(index=True)
    signals_json: str = Field(
        description="JSON list of triggered fraud signals with weights"
    )
    narrative: Optional[str] = Field(default=None)
    action_taken: str = Field(
        index=True,
        default="logged",
        description="logged | held | escalated | dismissed",
    )


# ---------------------------------------------------------------------------
# Correction
# ---------------------------------------------------------------------------
class Correction(SQLModel, table=True):
    __tablename__ = "correction"

    id: Optional[int] = Field(default=None, primary_key=True)
    original_match_id: int = Field(index=True, foreign_key="match.id")
    corrected_invoice_id: int = Field(index=True, foreign_key="invoice.id")
    reason: str = Field(index=True)
    timestamp: datetime = Field(index=True, default_factory=datetime.utcnow)
