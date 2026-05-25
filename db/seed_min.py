"""Minimal seed: create tables and insert 1 counterparty + 3 invoices.

Run:
    python -m db.seed_min
"""
from __future__ import annotations

import json
from datetime import date

from sqlmodel import Session, select

from db.models import Counterparty, Invoice
from db.session import engine, init_db


def seed() -> None:
    init_db()

    with Session(engine) as session:
        existing = session.exec(
            select(Counterparty).where(Counterparty.normalized_name == "acme pte ltd")
        ).first()
        if existing:
            print(f"Counterparty already seeded (id={existing.id}); skipping.")
            return

        cp = Counterparty(
            display_name="Acme Pte Ltd",
            verified_domain="acme.sg",
            normalized_name="acme pte ltd",
            embedding_vec_json=None,
            payment_history_summary_json=json.dumps(
                {
                    "avg_days_to_pay": 14,
                    "median_amount_sgd": 5200.0,
                    "preferred_currency": "SGD",
                    "recent_payment_count": 8,
                }
            ),
        )
        session.add(cp)
        session.commit()
        session.refresh(cp)
        assert cp.id is not None

        invoices = [
            Invoice(
                invoice_no="INV-2026-0001",
                customer_id=cp.id,
                amount=4980.00,
                currency="SGD",
                issued_date=date(2026, 5, 1),
                due_date=date(2026, 5, 31),
                status="open",
                reference_hint="ACME-0001",
            ),
            Invoice(
                invoice_no="INV-2026-0002",
                customer_id=cp.id,
                amount=12350.50,
                currency="USD",
                issued_date=date(2026, 5, 10),
                due_date=date(2026, 6, 9),
                status="open",
                reference_hint="ACME-0002",
            ),
            Invoice(
                invoice_no="INV-2026-0003",
                customer_id=cp.id,
                amount=2100.00,
                currency="EUR",
                issued_date=date(2026, 5, 18),
                due_date=date(2026, 6, 17),
                status="open",
                reference_hint="ACME-0003",
            ),
        ]
        session.add_all(invoices)
        session.commit()

        print(f"Seeded counterparty id={cp.id} and {len(invoices)} invoices.")


if __name__ == "__main__":
    seed()
