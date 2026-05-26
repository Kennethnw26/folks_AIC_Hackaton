from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from sqlmodel import Field, SQLModel


class Tenant(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    home_currency: str = "MYR"
    whatsapp_phone_id: str


class Invoice(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    tenant_id: int = Field(foreign_key="tenant.id", index=True)
    invoice_number: str
    amount: float
    currency: str
    due_date: date
    beneficiary_name: str
    status: str = "open"


class Payment(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    tenant_id: int = Field(foreign_key="tenant.id", index=True)
    amount: float
    currency: str
    date: date
    beneficiary_name: str
    bank_ref: str
    matched_invoice_id: Optional[int] = Field(default=None, foreign_key="invoice.id")
    fraud_score: float = 0.0
    status: str = "pending"


class BeneficiaryHistory(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    tenant_id: int = Field(foreign_key="tenant.id", index=True)
    account_number: str
    beneficiary_name: str
    first_seen: datetime


class VendorDomain(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    tenant_id: int = Field(foreign_key="tenant.id", index=True)
    domain: str


class MessageIdempotency(SQLModel, table=True):
    message_id: str = Field(primary_key=True)
    processed_at: datetime


class OcrRetryState(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    tenant_id: int = Field(foreign_key="tenant.id")
    whatsapp_user: str
    expected_at: datetime
    raw_proof_partial: str
