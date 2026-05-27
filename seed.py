from __future__ import annotations

from datetime import date, datetime

from sqlmodel import Session, select

from config import settings
from db.models import BeneficiaryHistory, Invoice, Payment, Tenant, VendorDomain
from db.session import engine


def run_seed() -> None:
    with Session(engine) as session:
        existing = session.exec(select(Tenant).where(Tenant.name == "Demo Sdn Bhd")).first()
        if existing:
            return

        tenant = Tenant(
            name="Demo Sdn Bhd",
            home_currency="MYR",
            whatsapp_phone_id=settings.meta_phone_id or "demo_phone_id",
        )
        session.add(tenant)
        session.flush()

        invoices = [
            Invoice(tenant_id=tenant.id, invoice_number="INV-001", amount=5000.00, currency="USD",
                    due_date=date(2026, 5, 30), beneficiary_name="Acme Corporation"),
            Invoice(tenant_id=tenant.id, invoice_number="INV-002", amount=12500.00, currency="SGD",
                    due_date=date(2026, 6, 5), beneficiary_name="Global Logistics Pte"),
            Invoice(tenant_id=tenant.id, invoice_number="INV-003", amount=8900.00, currency="EUR",
                    due_date=date(2026, 6, 10), beneficiary_name="Partners GmbH"),
            Invoice(tenant_id=tenant.id, invoice_number="INV-004", amount=45000.00, currency="MYR",
                    due_date=date(2026, 5, 28), beneficiary_name="Local Supplier Sdn Bhd"),
            Invoice(tenant_id=tenant.id, invoice_number="INV-005", amount=35000.00, currency="CNY",
                    due_date=date(2026, 6, 15), beneficiary_name="Shenzhen Components Ltd"),
        ]
        for inv in invoices:
            session.add(inv)

        domains = [
            VendorDomain(tenant_id=tenant.id, domain="acme.com"),
            VendorDomain(tenant_id=tenant.id, domain="globallogistics.com.sg"),
            VendorDomain(tenant_id=tenant.id, domain="partners.de"),
        ]
        for d in domains:
            session.add(d)

        history = [
            BeneficiaryHistory(tenant_id=tenant.id, account_number="ACME-0012345",
                               beneficiary_name="Acme Corporation", first_seen=datetime.utcnow()),
            BeneficiaryHistory(tenant_id=tenant.id, account_number="GL-SG-00492",
                               beneficiary_name="Global Logistics Pte", first_seen=datetime.utcnow()),
            BeneficiaryHistory(tenant_id=tenant.id, account_number="PART-DE-7731",
                               beneficiary_name="Partners GmbH", first_seen=datetime.utcnow()),
            BeneficiaryHistory(tenant_id=tenant.id, account_number="LOCAL-99887",
                               beneficiary_name="Local Supplier Sdn Bhd", first_seen=datetime.utcnow()),
            BeneficiaryHistory(tenant_id=tenant.id, account_number="SZ-CN-35001",
                               beneficiary_name="Shenzhen Components Ltd", first_seen=datetime.utcnow()),
        ]
        for h in history:
            session.add(h)

        # Pre-confirmed payments so trust graph is warmed for all known beneficiaries
        confirmed_payments = [
            Payment(tenant_id=tenant.id, amount=5000.00, currency="USD", date=date(2026, 1, 15),
                    beneficiary_name="Acme Corporation", bank_ref="ACME-0012345",
                    fraud_score=0.0, status="confirmed", matched_invoice_id=None),
            Payment(tenant_id=tenant.id, amount=5000.00, currency="USD", date=date(2026, 3, 10),
                    beneficiary_name="Acme Corporation", bank_ref="ACME-0012345",
                    fraud_score=0.0, status="confirmed", matched_invoice_id=None),
            Payment(tenant_id=tenant.id, amount=12500.00, currency="SGD", date=date(2026, 2, 5),
                    beneficiary_name="Global Logistics Pte", bank_ref="GL-SG-00492",
                    fraud_score=0.0, status="confirmed", matched_invoice_id=None),
            Payment(tenant_id=tenant.id, amount=8900.00, currency="EUR", date=date(2026, 1, 10),
                    beneficiary_name="Partners GmbH", bank_ref="PART-DE-7731",
                    fraud_score=0.0, status="confirmed", matched_invoice_id=None),
            Payment(tenant_id=tenant.id, amount=45000.00, currency="MYR", date=date(2026, 2, 20),
                    beneficiary_name="Local Supplier Sdn Bhd", bank_ref="LOCAL-99887",
                    fraud_score=0.0, status="confirmed", matched_invoice_id=None),
            Payment(tenant_id=tenant.id, amount=35000.00, currency="CNY", date=date(2026, 1, 20),
                    beneficiary_name="Shenzhen Components Ltd", bank_ref="SZ-CN-35001",
                    fraud_score=0.0, status="confirmed", matched_invoice_id=None),
        ]
        for p in confirmed_payments:
            session.add(p)

        session.commit()
