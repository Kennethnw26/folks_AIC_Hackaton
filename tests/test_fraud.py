"""Tests for fraud/detector.py.

The headline test mirrors the demo case: RM 84,200 from
"Tan Ah Kow Trading" to a new Maybank account, spoofed instruction
domain, three other SMEs flagging the same account, and an urgency
anomaly. All four signals must fire and produce a 90+ confidence
"block" recommendation.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlmodel import Session, SQLModel, create_engine

from db.models import Counterparty, FlaggedAccount, Invoice, Match, Payment
from fraud.detector import (
    FraudAssessment,
    SignalEvidence,
    _noisy_or,
    _recommend,
    assess_fraud,
    signal_domain_spoof,
    signal_new_beneficiary,
    signal_trust_graph,
    signal_urgency_anomaly,
)


# ---------------------------------------------------------------------------
# In-memory DB fixture
# ---------------------------------------------------------------------------
@pytest.fixture()
def db_session() -> Session:
    """Fresh in-memory SQLite session with all tables created."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


# ---------------------------------------------------------------------------
# Demo-case fixtures
# ---------------------------------------------------------------------------
_REAL_DOMAIN = "tanahkow.com.my"
_SPOOF_DOMAIN = "tan-ahkow.com"          # hyphenation + TLD swap
_REAL_ACCOUNT = "MBB-5142-0099-3318"     # historical Maybank account
_OLD_ACCOUNT_2 = "MBB-5142-0099-3319"
_OLD_ACCOUNT_3 = "MBB-5142-0099-3320"
_NEW_ACCOUNT = "MBB-5550-9100-7711"      # never-before-seen Maybank account


def _build_demo_counterparty(db: Session) -> Counterparty:
    cp = Counterparty(
        display_name="Tan Ah Kow Trading",
        normalized_name="TAN AH KOW TRADING",
        verified_domain=_REAL_DOMAIN,
        payment_history_summary_json=json.dumps({
            "typical_tone": "calm, formal, bilingual (EN/BM)",
            "avg_payment_myr": 38500,
            "median_payment_myr": 35000,
            "usual_reference_pattern": "INV-XXXX",
            "off_hours_instructions": False,
        }),
    )
    db.add(cp)
    db.commit()
    db.refresh(cp)
    return cp


def _seed_historical_payments(db: Session, counterparty: Counterparty) -> None:
    """Create three historical confirmed payments to two distinct Maybank
    accounts so history_depth >= 3."""
    accounts = [_REAL_ACCOUNT, _OLD_ACCOUNT_2, _REAL_ACCOUNT, _OLD_ACCOUNT_3]
    for i, acct in enumerate(accounts, start=1):
        inv = Invoice(
            invoice_no=f"INV-OLD-{i:03d}",
            customer_id=counterparty.id,
            amount=30_000.0 + 1000 * i,
            currency="MYR",
            issued_date=date(2026, 1, i),
            due_date=date(2026, 2, i),
            status="paid",
        )
        db.add(inv)
        db.commit()
        db.refresh(inv)

        pay = Payment(
            raw_amount=inv.amount,
            currency="MYR",
            payment_date=date(2026, 1, i + 5),
            sender_name=counterparty.display_name,
            reference=inv.invoice_no,
            destination_account=acct,
            status="reconciled",
        )
        db.add(pay)
        db.commit()
        db.refresh(pay)

        match = Match(
            invoice_id=inv.id,
            payment_id=pay.id,
            confidence=0.99,
            sub_scores_json="{}",
            status="confirmed",
            confirmed_at=datetime(2026, 1, i + 5, 12, 0, 0),
        )
        db.add(match)
    db.commit()


def _seed_flagged_account(db: Session) -> None:
    """Mirror the demo: three other SMEs reported this account recently."""
    flagged = FlaggedAccount(
        account_number=_NEW_ACCOUNT,
        bank_name="Maybank",
        first_flagged_at=datetime.utcnow() - timedelta(days=5),
        reporter_count=3,
        total_loss_estimate=192_400.0,
        severity="critical",
    )
    db.add(flagged)
    db.commit()


def _build_demo_payment(counterparty: Counterparty) -> Payment:
    return Payment(
        id=999,
        raw_amount=84_200.00,
        currency="MYR",
        payment_date=date(2026, 5, 24),
        sender_name=counterparty.display_name,
        reference="URGENT INV-9921",
        destination_account=_NEW_ACCOUNT,
        status="unreconciled",
    )


_SPOOF_HEADERS = {
    "From": "Tan Ah Kow <ahkow@tan-ahkow.com>",
    "Subject": "URGENT — please pay today, supplier waiting",
    "Date": "Mon, 24 May 2026 18:42:11 +0800",
}

_URGENT_INSTRUCTION = (
    "Boss, urgent please! Our usual Maybank account is frozen by bank "
    "for audit. PLEASE transfer the RM84,200 to the new account "
    f"{_NEW_ACCOUNT} BEFORE 5pm today or supplier will cancel order. "
    "Do not call my office — I am in meetings all afternoon. Just go "
    "ahead and pay, I will explain later. Thanks."
)


# ---------------------------------------------------------------------------
# Mocked Hermes responses
# ---------------------------------------------------------------------------
def _mock_hermes_response(payload: dict) -> MagicMock:
    msg = MagicMock()
    msg.content = json.dumps(payload)
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ---------------------------------------------------------------------------
# Headline test: synthetic demo case fires all four signals
# ---------------------------------------------------------------------------
class TestDemoCaseAllSignalsFire:
    """RM 84,200 → 'Tan Ah Kow Trading' demo case must hit 90+ confidence."""

    def test_signal_new_beneficiary_fires_strong(self, db_session: Session):
        cp = _build_demo_counterparty(db_session)
        _seed_historical_payments(db_session, cp)
        pay = _build_demo_payment(cp)

        sig = signal_new_beneficiary(pay, cp, db_session)
        assert sig.signal_name == "new_beneficiary"
        # history_depth >= 3 and account is new → 0.95
        assert sig.score == Decimal("0.95")
        assert sig.evidence["history_depth"] >= 3
        assert sig.evidence["new_account"] == _NEW_ACCOUNT
        assert _NEW_ACCOUNT not in sig.evidence["old_accounts"]
        assert _REAL_ACCOUNT in sig.evidence["old_accounts"]

    def test_signal_domain_spoof_fires(self, db_session: Session):
        cp = _build_demo_counterparty(db_session)
        sig = signal_domain_spoof(_SPOOF_HEADERS, cp)
        assert sig.signal_name == "domain_spoof"
        # tan-ahkow.com vs tanahkow.com.my — hyphenation_swap or tld_swap or typosquat
        assert sig.score >= Decimal("0.85"), (
            f"expected spoof score >= 0.85, got {sig.score} "
            f"(pattern={sig.evidence['pattern']})"
        )
        assert sig.evidence["pattern"] in {"hyphenation_swap", "tld_swap", "typosquat"}
        assert sig.evidence["observed_domain"] == _SPOOF_DOMAIN

    def test_signal_trust_graph_fires(self, db_session: Session):
        cp = _build_demo_counterparty(db_session)
        _seed_flagged_account(db_session)
        pay = _build_demo_payment(cp)

        sig = signal_trust_graph(pay, db_session)
        assert sig.signal_name == "trust_graph"
        # 0.4 + 0.2 * 3 = 1.0, capped at 0.95, +0.1 recency clamped at 0.95
        assert sig.score == Decimal("0.95")
        assert sig.evidence["reporter_count"] == 3
        assert sig.evidence["total_loss_estimate"] == pytest.approx(192_400.0)

    @pytest.mark.asyncio
    async def test_signal_urgency_anomaly_fires(self, db_session: Session):
        cp = _build_demo_counterparty(db_session)

        fake = _mock_hermes_response({
            "score": 0.92,
            "explanation": (
                "Tone is highly urgent and demands secrecy, inconsistent "
                "with this counterparty's calm formal style."
            ),
        })
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=fake)

        with patch("fraud.detector._async_client", return_value=mock_client):
            sig = await signal_urgency_anomaly(_URGENT_INSTRUCTION, cp)

        assert sig.signal_name == "urgency_anomaly"
        assert sig.score == Decimal("0.9200")
        assert "urgent" in sig.explanation.lower()

    @pytest.mark.asyncio
    async def test_assess_fraud_demo_case_90_plus(self, db_session: Session):
        cp = _build_demo_counterparty(db_session)
        _seed_historical_payments(db_session, cp)
        _seed_flagged_account(db_session)
        pay = _build_demo_payment(cp)

        # Two distinct Hermes calls: urgency-anomaly + narrative.
        urgency_resp = _mock_hermes_response({
            "score": 0.9,
            "explanation": "Highly urgent and demands secrecy.",
        })
        narrative_resp = _mock_hermes_response({
            "narrative": (
                "All four fraud signals fired: this destination account "
                "is new to this counterparty, the email domain is a "
                "lookalike, three other SMEs flagged the account, and "
                "the tone is unusually urgent. Recommend BLOCK and "
                "verbal verification via a known phone number."
            ),
        })

        # Round-robin the two responses by call order.
        call_iter = iter([urgency_resp, narrative_resp])
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=lambda *a, **kw: next(call_iter)
        )

        with patch("fraud.detector._async_client", return_value=mock_client):
            assessment = await assess_fraud(
                payment=pay,
                counterparty=cp,
                instruction_text=_URGENT_INSTRUCTION,
                email_headers=_SPOOF_HEADERS,
                db=db_session,
            )

        assert isinstance(assessment, FraudAssessment)
        assert assessment.payment_id == pay.id
        assert len(assessment.signals) == 4
        names = {s.signal_name for s in assessment.signals}
        assert names == {
            "new_beneficiary",
            "domain_spoof",
            "trust_graph",
            "urgency_anomaly",
        }
        # All four signals fired
        for s in assessment.signals:
            assert s.score > Decimal("0"), f"signal {s.signal_name} did not fire"

        # 90+ confidence headline assertion
        assert assessment.fraud_confidence >= 90, (
            f"fraud_confidence={assessment.fraud_confidence}, "
            f"signal scores={[(s.signal_name, s.score) for s in assessment.signals]}"
        )
        assert assessment.recommended_action == "block"
        assert assessment.narrative  # populated


# ---------------------------------------------------------------------------
# Signal unit tests
# ---------------------------------------------------------------------------
class TestSignalNewBeneficiary:
    def test_zero_history_returns_low_score(self, db_session: Session):
        cp = _build_demo_counterparty(db_session)
        pay = _build_demo_payment(cp)
        sig = signal_new_beneficiary(pay, cp, db_session)
        assert sig.score == Decimal("0.2")
        assert sig.evidence["history_depth"] == 0

    def test_known_account_returns_zero(self, db_session: Session):
        cp = _build_demo_counterparty(db_session)
        _seed_historical_payments(db_session, cp)
        pay = _build_demo_payment(cp)
        pay.destination_account = _REAL_ACCOUNT  # known
        sig = signal_new_beneficiary(pay, cp, db_session)
        assert sig.score == Decimal("0.0")

    def test_partial_history_new_account_moderate(self, db_session: Session):
        """history_depth 1-2 with new account → 0.5."""
        cp = _build_demo_counterparty(db_session)
        # Seed exactly 2 confirmed payments to 2 distinct accounts.
        for i, acct in enumerate([_REAL_ACCOUNT, _OLD_ACCOUNT_2], start=1):
            inv = Invoice(
                invoice_no=f"INV-X-{i}",
                customer_id=cp.id,
                amount=1000.0,
                currency="MYR",
                issued_date=date(2026, 1, i),
                due_date=date(2026, 2, i),
                status="paid",
            )
            db_session.add(inv)
            db_session.commit()
            db_session.refresh(inv)
            p = Payment(
                raw_amount=1000.0, currency="MYR",
                payment_date=date(2026, 1, i + 1),
                sender_name=cp.display_name,
                destination_account=acct,
                status="reconciled",
            )
            db_session.add(p)
            db_session.commit()
            db_session.refresh(p)
            db_session.add(Match(
                invoice_id=inv.id, payment_id=p.id, confidence=0.99,
                sub_scores_json="{}", status="confirmed",
                confirmed_at=datetime(2026, 1, i + 1, 12, 0, 0),
            ))
        db_session.commit()

        pay = _build_demo_payment(cp)
        sig = signal_new_beneficiary(pay, cp, db_session)
        assert sig.score == Decimal("0.5")
        assert sig.evidence["history_depth"] == 2


class TestSignalDomainSpoof:
    def _cp(self, domain: str = "acme.com") -> Counterparty:
        return Counterparty(
            id=1,
            display_name="Acme",
            normalized_name="ACME",
            verified_domain=domain,
        )

    def test_exact_match_scores_zero(self):
        cp = self._cp("acme.com")
        sig = signal_domain_spoof({"From": "boss@acme.com"}, cp)
        assert sig.score == Decimal("0.0")
        assert sig.evidence["pattern"] == "exact_match"

    def test_typosquat_levenshtein(self):
        cp = self._cp("acme.com")
        sig = signal_domain_spoof({"From": "boss@acrne.com"}, cp)
        assert sig.score == Decimal("0.9")
        assert sig.evidence["pattern"] == "typosquat"

    def test_tld_swap(self):
        cp = self._cp("acme.com.my")
        sig = signal_domain_spoof({"From": "boss@acme.com"}, cp)
        assert sig.evidence["pattern"] == "tld_swap"
        assert sig.score == Decimal("0.85")

    def test_hyphenation_swap(self):
        cp = self._cp("tanahkow.com")
        sig = signal_domain_spoof({"From": "boss@tan-ahkow.com"}, cp)
        assert sig.evidence["pattern"] == "hyphenation_swap"
        assert sig.score == Decimal("0.9")

    def test_no_similarity(self):
        cp = self._cp("acme.com")
        sig = signal_domain_spoof({"From": "ceo@totallyunrelated.org"}, cp)
        assert sig.score == Decimal("0.3")
        assert sig.evidence["pattern"] == "no_similarity"

    def test_no_headers_scores_zero(self):
        cp = self._cp("acme.com")
        sig = signal_domain_spoof(None, cp)
        assert sig.score == Decimal("0.0")
        assert sig.evidence["pattern"] == "no_headers_or_unverified"


class TestSignalTrustGraph:
    def test_not_flagged_scores_zero(self, db_session: Session):
        cp = _build_demo_counterparty(db_session)
        pay = _build_demo_payment(cp)
        sig = signal_trust_graph(pay, db_session)
        assert sig.score == Decimal("0.0")
        assert sig.evidence["reporter_count"] == 0

    def test_single_reporter_no_recency(self, db_session: Session):
        cp = _build_demo_counterparty(db_session)
        pay = _build_demo_payment(cp)
        old = FlaggedAccount(
            account_number=_NEW_ACCOUNT,
            bank_name="Maybank",
            first_flagged_at=datetime.utcnow() - timedelta(days=120),
            reporter_count=1,
            total_loss_estimate=10_000.0,
            severity="medium",
        )
        db_session.add(old)
        db_session.commit()
        sig = signal_trust_graph(pay, db_session)
        # 0.4 + 0.2 * 1 = 0.6, no recency boost
        assert sig.score == Decimal("0.6")

    def test_cap_at_0_95(self, db_session: Session):
        cp = _build_demo_counterparty(db_session)
        pay = _build_demo_payment(cp)
        db_session.add(FlaggedAccount(
            account_number=_NEW_ACCOUNT, bank_name="Maybank",
            first_flagged_at=datetime.utcnow() - timedelta(days=5),
            reporter_count=10, total_loss_estimate=1.0, severity="critical",
        ))
        db_session.commit()
        sig = signal_trust_graph(pay, db_session)
        assert sig.score == Decimal("0.95")


# ---------------------------------------------------------------------------
# Aggregator math tests
# ---------------------------------------------------------------------------
class TestAggregatorMath:
    def test_noisy_or_empty_is_zero(self):
        assert _noisy_or([]) == Decimal("0")

    def test_noisy_or_single_value_returned(self):
        assert _noisy_or([Decimal("0.5")]) == Decimal("0.5")

    def test_noisy_or_two_values(self):
        # 1 - (1-0.5)(1-0.5) = 0.75
        result = _noisy_or([Decimal("0.5"), Decimal("0.5")])
        assert result == Decimal("0.75")

    def test_noisy_or_strong_signals(self):
        # 1 - (1-0.95)(1-0.9)(1-0.95)(1-0.9) = 1 - 0.05*0.1*0.05*0.1
        # = 1 - 0.0000250 ≈ 0.999975
        result = _noisy_or([
            Decimal("0.95"),
            Decimal("0.9"),
            Decimal("0.95"),
            Decimal("0.9"),
        ])
        assert result > Decimal("0.99")

    def test_recommend_block(self):
        assert _recommend(75) == "block"
        assert _recommend(99) == "block"

    def test_recommend_verify(self):
        assert _recommend(40) == "verify"
        assert _recommend(74) == "verify"

    def test_recommend_proceed(self):
        assert _recommend(0) == "proceed"
        assert _recommend(39) == "proceed"


# ---------------------------------------------------------------------------
# Critical rule: never auto-block
# ---------------------------------------------------------------------------
class TestNeverAutoBlock:
    @pytest.mark.asyncio
    async def test_assess_fraud_does_not_mutate_payment(self, db_session: Session):
        """Even at max confidence, payment.status must remain unchanged."""
        cp = _build_demo_counterparty(db_session)
        _seed_historical_payments(db_session, cp)
        _seed_flagged_account(db_session)
        pay = _build_demo_payment(cp)
        original_status = pay.status

        urgency_resp = _mock_hermes_response({
            "score": 0.95, "explanation": "extreme urgency",
        })
        narrative_resp = _mock_hermes_response({
            "narrative": "All signals fired; recommend block.",
        })
        call_iter = iter([urgency_resp, narrative_resp])
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=lambda *a, **kw: next(call_iter)
        )

        with patch("fraud.detector._async_client", return_value=mock_client):
            assessment = await assess_fraud(
                payment=pay, counterparty=cp,
                instruction_text=_URGENT_INSTRUCTION,
                email_headers=_SPOOF_HEADERS, db=db_session,
            )

        # The detector returns a recommendation, but never mutates state.
        assert assessment.recommended_action == "block"
        assert pay.status == original_status, (
            "Detector must NEVER auto-block — payment status must be untouched"
        )
