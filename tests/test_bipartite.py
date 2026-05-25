"""Tests for agent/bipartite.py — lump-sum, partial, and bipartite matching."""
from __future__ import annotations

import sys
from datetime import date
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import pytest

_PACKAGE_ROOT = Path(__file__).parent.parent
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from agent.bipartite import (
    Assignment,
    AssignmentResult,
    detect_lump_sum,
    detect_partial,
    solve_assignment,
)
from db.models import Invoice, Payment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_TODAY = date(2024, 3, 15)


def _payment(
    id: int,
    amount: float,
    currency: str = "USD",
    reference: str = "",
    sender: str = "Acme Corp",
) -> Payment:
    return Payment(
        id=id,
        raw_amount=amount,
        currency=currency,
        payment_date=_TODAY,
        sender_name=sender,
        reference=reference,
        destination_account="ACC-001",
    )


def _invoice(
    id: int,
    amount: float,
    currency: str = "USD",
    invoice_no: str = "",
    days_old: int = 10,
) -> Invoice:
    issued = date(2024, 3, _TODAY.day - days_old) if days_old < _TODAY.day else date(2024, 2, 14)
    return Invoice(
        id=id,
        invoice_no=invoice_no or f"INV-{id:04d}",
        customer_id=1,
        amount=amount,
        currency=currency,
        issued_date=issued,
        due_date=date(2024, 4, 15),
        status="open",
    )


# ---------------------------------------------------------------------------
# 1. Lump-sum detection — primary test
#    USD 30 payment covering three USD 10 invoices.
# ---------------------------------------------------------------------------
class TestDetectLumpSum:

    def test_lump_sum_usd_three_invoices(self):
        """USD 30 should fire lump-sum detection against three USD 10 invoices."""
        payment = _payment(id=1, amount=30.0, currency="USD")
        invoices = [
            _invoice(id=10, amount=10.0, currency="USD", days_old=20),
            _invoice(id=11, amount=10.0, currency="USD", days_old=15),
            _invoice(id=12, amount=10.0, currency="USD", days_old=5),
        ]

        # FX USD→USD = 1.0, no network call needed
        result = detect_lump_sum(payment, invoices)

        assert result is not None, "Lump-sum detection should fire"
        assert result.kind == "lump_sum"
        assert set(result.invoice_ids) == {10, 11, 12}
        # Each invoice should be allocated its USD 10 in full
        for inv_id in [10, 11, 12]:
            assert result.allocations[inv_id] == Decimal("10.00"), (
                f"Invoice {inv_id} should be allocated USD 10.00"
            )
        # Residual should be near zero (payment == sum)
        assert abs(result.residual_amount) <= Decimal("0.02"), (
            f"Residual should be ≈ 0, got {result.residual_amount}"
        )
        # Confidence should be high (gap is 0 %)
        assert result.confidence >= Decimal("0.98")

    def test_lump_sum_returns_none_when_no_match(self):
        """Payment amount far from any prefix sum → None."""
        payment = _payment(id=2, amount=99.0, currency="USD")
        invoices = [
            _invoice(id=20, amount=10.0, currency="USD"),
            _invoice(id=21, amount=10.0, currency="USD"),
        ]
        result = detect_lump_sum(payment, invoices)
        assert result is None

    def test_lump_sum_empty_invoices(self):
        """Empty invoice list → None (no crash)."""
        payment = _payment(id=3, amount=50.0)
        assert detect_lump_sum(payment, []) is None

    def test_lump_sum_reference_hits_prioritised(self):
        """Invoices whose numbers appear in payment reference should be selected first."""
        payment = _payment(
            id=4,
            amount=20.0,
            currency="USD",
            reference="INV-0101 INV-0102 payment",
        )
        # Two invoices explicitly referenced + one extra
        inv_referenced_a = _invoice(id=101, amount=10.0, invoice_no="INV-0101", days_old=30)
        inv_referenced_b = _invoice(id=102, amount=10.0, invoice_no="INV-0102", days_old=25)
        inv_extra = _invoice(id=103, amount=10.0, invoice_no="INV-0103", days_old=1)

        result = detect_lump_sum(payment, [inv_extra, inv_referenced_a, inv_referenced_b])

        assert result is not None
        assert set(result.invoice_ids) == {101, 102}
        assert 103 not in result.invoice_ids

    def test_lump_sum_within_1_5_pct_tolerance(self):
        """Payment 1 % above sum should still be detected (within 1.5 % band)."""
        # Sum of invoices = 100.00, payment = 101.00 → 1 % above
        payment = _payment(id=5, amount=101.0, currency="USD")
        invoices = [
            _invoice(id=30, amount=50.0, currency="USD", days_old=20),
            _invoice(id=31, amount=50.0, currency="USD", days_old=10),
        ]
        result = detect_lump_sum(payment, invoices)
        assert result is not None
        assert result.kind == "lump_sum"
        # Residual should be ≈ −1.00 (payment exceeded invoice sum)
        assert result.residual_amount <= Decimal("0.00")


# ---------------------------------------------------------------------------
# 2. Partial-payment detection
# ---------------------------------------------------------------------------
class TestDetectPartial:

    def _mock_score(self, name_fit: float, reference_fit: float):
        """Return a mock MatchScore with given name and reference fits."""
        from agent.matcher import MatchScore
        return MatchScore(
            amount_fit=Decimal("0.6"),
            date_fit=Decimal("1.0"),
            name_fit=Decimal(str(name_fit)),
            reference_fit=Decimal(str(reference_fit)),
            fee_inferred=Decimal("0"),
            composite_confidence=Decimal("0.7"),
        )

    def test_partial_detected_at_50_pct(self):
        """50 % payment with strong name and reference fits → partial."""
        payment = _payment(id=10, amount=50.0, currency="USD")
        invoice = _invoice(id=200, amount=100.0, currency="USD")

        with patch("agent.bipartite.score_pair") as mock_sp:
            mock_sp.return_value = self._mock_score(
                name_fit=0.85, reference_fit=0.75
            )
            result = detect_partial(payment, invoice)

        assert result is not None
        assert result.kind == "partial"
        assert result.residual_amount == Decimal("50.00")   # 100 − 50
        assert result.allocations[200] == Decimal("50.00")

    def test_partial_not_detected_below_30_pct(self):
        """Payment at 20 % of invoice → not a partial (too small)."""
        payment = _payment(id=11, amount=20.0, currency="USD")
        invoice = _invoice(id=201, amount=100.0, currency="USD")

        with patch("agent.bipartite.score_pair") as mock_sp:
            mock_sp.return_value = self._mock_score(0.9, 0.9)
            result = detect_partial(payment, invoice)

        assert result is None

    def test_partial_not_detected_above_95_pct(self):
        """Payment at 98 % of invoice → bipartite territory, not partial."""
        payment = _payment(id=12, amount=98.0, currency="USD")
        invoice = _invoice(id=202, amount=100.0, currency="USD")

        with patch("agent.bipartite.score_pair") as mock_sp:
            mock_sp.return_value = self._mock_score(0.9, 0.9)
            result = detect_partial(payment, invoice)

        assert result is None

    def test_partial_not_detected_weak_name_fit(self):
        """Low name_fit (≤ 0.70) should block partial detection."""
        payment = _payment(id=13, amount=60.0, currency="USD")
        invoice = _invoice(id=203, amount=100.0, currency="USD")

        with patch("agent.bipartite.score_pair") as mock_sp:
            mock_sp.return_value = self._mock_score(name_fit=0.60, reference_fit=0.80)
            result = detect_partial(payment, invoice)

        assert result is None

    def test_partial_not_detected_weak_reference_fit(self):
        """Low reference_fit (≤ 0.50) should block partial detection."""
        payment = _payment(id=14, amount=60.0, currency="USD")
        invoice = _invoice(id=204, amount=100.0, currency="USD")

        with patch("agent.bipartite.score_pair") as mock_sp:
            mock_sp.return_value = self._mock_score(name_fit=0.90, reference_fit=0.40)
            result = detect_partial(payment, invoice)

        assert result is None


# ---------------------------------------------------------------------------
# 3. solve_assignment
# ---------------------------------------------------------------------------
class TestSolveAssignment:

    def test_one_to_one_exact_match(self):
        """Single payment ≈ single invoice → one_to_one assignment."""
        payment = _payment(id=20, amount=500.0, currency="USD")
        invoice = _invoice(id=300, amount=500.0, currency="USD", invoice_no="INV-0300")
        # Patch score_pair to return high confidence
        from agent.matcher import MatchScore
        high_score = MatchScore(
            amount_fit=Decimal("1"),
            date_fit=Decimal("1"),
            name_fit=Decimal("0"),
            reference_fit=Decimal("0"),
            fee_inferred=Decimal("0"),
            composite_confidence=Decimal("0.6"),
        )
        with patch("agent.bipartite.score_pair", return_value=high_score):
            result = solve_assignment([payment], [invoice])

        assert len(result.assignments) == 1
        asgn = result.assignments[0]
        assert asgn.payment_id == 20
        assert asgn.invoice_ids == [300]
        assert asgn.kind == "one_to_one"

    def test_empty_inputs(self):
        """Empty payments or invoices → no assignments, all unassigned."""
        result = solve_assignment([], [_invoice(id=400, amount=100.0)])
        assert result.assignments == []
        assert 400 in result.unassigned_invoices

    def test_low_confidence_becomes_unassigned(self):
        """Pairs below 0.50 composite confidence should be unassigned."""
        payment = _payment(id=30, amount=999.0, currency="USD")
        invoice = _invoice(id=500, amount=1.0, currency="USD")
        from agent.matcher import MatchScore
        low_score = MatchScore(
            amount_fit=Decimal("0"),
            date_fit=Decimal("0"),
            name_fit=Decimal("0"),
            reference_fit=Decimal("0"),
            fee_inferred=Decimal("0"),
            composite_confidence=Decimal("0.1"),
        )
        with patch("agent.bipartite.score_pair", return_value=low_score):
            result = solve_assignment([payment], [invoice])

        assert result.assignments == []
        assert 30 in result.unassigned_payments
        assert 500 in result.unassigned_invoices
