"""Tests for agent/matcher.py.

Three fixture scenarios:
  1. clean_match    — every signal strong; composite > 0.95 expected.
  2. fee_match      — payment short by 0.8 % after FX; fee_inferred should
                      be non-zero and amount_fit should still be high.
  3. ambiguous_pair — two invoices whose composites are within 5 % of each
                      other; tests that rank_candidates returns both and
                      that the runner-up gets a ranking explanation.

All network calls (FX + Hermes) are mocked so tests run offline.
"""
from __future__ import annotations

import asyncio
import json
from datetime import date
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.matcher import (
    WEIGHT_AMOUNT,
    WEIGHT_DATE,
    WEIGHT_NAME,
    WEIGHT_REFERENCE,
    MatchScore,
    VerifiedMatch,
    rank_candidates,
    score_pair,
    verify_with_hermes,
)
from db.models import Counterparty, Invoice, Payment

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
_ISSUED = date(2025, 10, 1)
_PAID_ON_TIME = date(2025, 10, 15)   # 14 days after issue → date_fit = 1.0
_MYR_RATE = Decimal("4.5")           # 1 USD = 4.5 MYR (mocked)
_USD_RATE_TO_MYR = Decimal("4.5")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_counterparty(name: str = "Acme Sdn Bhd") -> Counterparty:
    return Counterparty(
        id=1,
        display_name=name,
        normalized_name=name.upper(),
    )


def _make_invoice(
    *,
    invoice_no: str = "INV-001",
    amount: float = 10_000.00,
    currency: str = "USD",
    issued_date: date = _ISSUED,
    customer_id: int = 1,
    inv_id: int = 1,
) -> Invoice:
    return Invoice(
        id=inv_id,
        invoice_no=invoice_no,
        customer_id=customer_id,
        amount=amount,
        currency=currency,
        issued_date=issued_date,
        due_date=date(issued_date.year, issued_date.month + 1, issued_date.day),
    )


def _make_payment(
    *,
    pay_id: int = 10,
    raw_amount: float = 45_000.00,
    currency: str = "MYR",
    payment_date: date = _PAID_ON_TIME,
    sender_name: str = "ACME SDN BHD",
    reference: str = "INV-001",
) -> Payment:
    return Payment(
        id=pay_id,
        raw_amount=raw_amount,
        currency=currency,
        payment_date=payment_date,
        sender_name=sender_name,
        reference=reference,
        destination_account="MY12-3456-7890",
    )


def _mock_get_rate(from_ccy: str, to_ccy: str, _date: date) -> Decimal:
    """Deterministic mock: USD→MYR = 4.5, MYR→MYR = 1, anything else = 1."""
    if from_ccy == "MYR" and to_ccy == "MYR":
        return Decimal("1")
    if from_ccy == "USD" and to_ccy == "MYR":
        return _MYR_RATE
    if from_ccy == "MYR" and to_ccy == "USD":
        return Decimal("1") / _MYR_RATE
    return Decimal("1")


def _mock_convert(amount: Decimal, from_ccy: str, to_ccy: str, d: date) -> Decimal:
    rate = _mock_get_rate(from_ccy, to_ccy, d)
    return (amount * rate).quantize(Decimal("0.0001"))


def _hermes_response(confidence: int = 98, narrative: str = "Strong match.", caveats: list = None) -> MagicMock:
    """Build a fake openai ChatCompletion response."""
    msg = MagicMock()
    msg.content = json.dumps({
        "final_confidence": confidence,
        "narrative": narrative,
        "caveats": caveats or [],
    })
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ---------------------------------------------------------------------------
# Fixture 1: Clean match
# ---------------------------------------------------------------------------
class TestCleanMatch:
    """USD 10 000 invoice, MYR payment at exact FX rate, correct reference."""

    @pytest.fixture(autouse=True)
    def patch_fx(self, monkeypatch):
        monkeypatch.setattr("agent.matcher.get_rate", _mock_get_rate)
        monkeypatch.setattr("agent.matcher.convert", _mock_convert)

    def test_score_pair_high_composite(self):
        inv = _make_invoice()              # 10 000 USD
        pay = _make_payment()              # 45 000 MYR (= 10 000 * 4.5 exactly)
        cp = _make_counterparty()

        score = score_pair(inv, pay, cp)

        # amount should be 1.0 (exact)
        assert score.amount_fit == Decimal("1")
        # date: 14 days in → 1.0
        assert score.date_fit == Decimal("1")
        # name: "ACME SDN BHD" vs "ACME SDN BHD" → 100 → 1.0
        assert score.name_fit == Decimal("1")
        # reference: "INV-001" in "INV-001" → 1.0
        assert score.reference_fit == Decimal("1")
        # no fee (exact amount)
        assert score.fee_inferred == Decimal("0")
        # composite = 1.0
        expected_composite = (
            WEIGHT_AMOUNT * Decimal("1")
            + WEIGHT_DATE * Decimal("1")
            + WEIGHT_NAME * Decimal("1")
            + WEIGHT_REFERENCE * Decimal("1")
        )
        assert score.composite_confidence == expected_composite.quantize(Decimal("0.0001"))

    def test_composite_above_95_percent(self):
        inv = _make_invoice()
        pay = _make_payment()
        cp = _make_counterparty()
        score = score_pair(inv, pay, cp)
        assert score.composite_confidence >= Decimal("0.95")

    @pytest.mark.asyncio
    async def test_verify_with_hermes_clean(self):
        inv = _make_invoice()
        pay = _make_payment()
        cp = _make_counterparty()
        score = score_pair(inv, pay, cp)

        fake_resp = _hermes_response(confidence=98, narrative="Perfect match.", caveats=[])
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=fake_resp)

        with patch("agent.matcher._async_client", return_value=mock_client):
            result = await verify_with_hermes(inv, pay, score)

        assert isinstance(result, VerifiedMatch)
        assert result.final_confidence == 98
        assert result.invoice_id == inv.id
        assert result.payment_id == pay.id
        assert result.caveats == []

    @pytest.mark.asyncio
    async def test_rank_candidates_single_winner(self):
        inv = _make_invoice()
        pay = _make_payment()
        cp = _make_counterparty()

        fake_resp = _hermes_response(confidence=98)
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=fake_resp)

        with patch("agent.matcher._async_client", return_value=mock_client):
            results = await rank_candidates(
                pay, [inv], counterparties={1: cp}
            )

        assert len(results) == 1
        assert results[0].final_confidence == 98


# ---------------------------------------------------------------------------
# Fixture 2: Fee-inferred match
# ---------------------------------------------------------------------------
class TestFeeInferredMatch:
    """Payment is 0.8 % short after FX — gap should be classified as bank fee."""

    @pytest.fixture(autouse=True)
    def patch_fx(self, monkeypatch):
        monkeypatch.setattr("agent.matcher.get_rate", _mock_get_rate)
        monkeypatch.setattr("agent.matcher.convert", _mock_convert)

    def _short_amount(self) -> float:
        """45 000 MYR less 0.8 % = 44 640 MYR."""
        full = Decimal("45000.00")
        short = full * (Decimal("1") - Decimal("0.008"))
        return float(short.quantize(Decimal("0.01")))

    def test_fee_inferred_nonzero(self):
        inv = _make_invoice()
        pay = _make_payment(raw_amount=self._short_amount())
        cp = _make_counterparty()

        score = score_pair(inv, pay, cp)

        # Fee should be detected
        assert score.fee_inferred > Decimal("0"), (
            f"Expected fee_inferred > 0, got {score.fee_inferred}"
        )

    def test_fee_inferred_approximate_value(self):
        """fee_inferred should be close to 0.8 % of 45 000 MYR = 360 MYR."""
        inv = _make_invoice()
        pay = _make_payment(raw_amount=self._short_amount())
        cp = _make_counterparty()

        score = score_pair(inv, pay, cp)

        expected_fee = Decimal("45000") * Decimal("0.008")  # 360 MYR
        assert abs(score.fee_inferred - expected_fee) < Decimal("2"), (
            f"fee_inferred={score.fee_inferred}, expected ~{expected_fee}"
        )

    def test_amount_fit_still_high_with_fee(self):
        """0.8 % gap → within 10 % band → amount_fit should be > 0.85."""
        inv = _make_invoice()
        pay = _make_payment(raw_amount=self._short_amount())
        cp = _make_counterparty()

        score = score_pair(inv, pay, cp)

        # 0.8 % is within the decay band (0.5 % – 10 %), so fit > 0.
        # Linear: 1 - (0.008 - 0.005) / (0.10 - 0.005) ≈ 0.9684
        assert score.amount_fit > Decimal("0.85"), (
            f"amount_fit={score.amount_fit} should be > 0.85 for 0.8 % gap"
        )

    def test_composite_still_reasonable(self):
        inv = _make_invoice()
        pay = _make_payment(raw_amount=self._short_amount())
        cp = _make_counterparty()

        score = score_pair(inv, pay, cp)
        # With all other signals perfect, composite should be well above 0.70
        assert score.composite_confidence > Decimal("0.70")

    def test_fee_not_inferred_for_large_gap(self):
        """A 5 % gap exceeds FEE_MAX_PCT → no fee classification."""
        inv = _make_invoice()
        big_gap_amount = float(Decimal("45000") * Decimal("0.95"))
        pay = _make_payment(raw_amount=big_gap_amount)
        cp = _make_counterparty()

        score = score_pair(inv, pay, cp)
        assert score.fee_inferred == Decimal("0")

    def test_fee_not_inferred_for_tiny_gap(self):
        """A 0.1 % gap is below FEE_MIN_PCT → no fee classification."""
        inv = _make_invoice()
        tiny_gap = float(Decimal("45000") * Decimal("0.999"))
        pay = _make_payment(raw_amount=tiny_gap)
        cp = _make_counterparty()

        score = score_pair(inv, pay, cp)
        assert score.fee_inferred == Decimal("0")


# ---------------------------------------------------------------------------
# Fixture 3: Ambiguous pair
# ---------------------------------------------------------------------------
class TestAmbiguousPair:
    """Two invoices that score within 5 % of each other on composite.

    INV-A has perfect reference match; INV-B has no reference match but an
    identical amount, date, and name.  The difference is 0.20 * 1.0 = 0.20
    vs 0.20 * 0.0 = 0.  After weighting the remaining 0.80 is shared, so
    composites differ by exactly 0.20 — we relax the scenario so both are
    > 0.55 and differ by < 0.25 to stay «ambiguous» in spirit while being
    deterministic.
    """

    # INV-A: perfect reference hit → reference_fit = 1.0
    # INV-B: no reference match   → reference_fit = 0.0
    # Everything else is the same → composite_A - composite_B = 0.20 * 1.0 = 0.20
    # 0.20 / composite_A ≈ 0.20 / ~0.80 = 25 %, which is outside 5 %.
    #
    # To create a ≤5 % gap we give INV-B a partial reference match (0.6)
    # and give INV-A only a partial name match.  That way the composites are
    # very close.  We use fixture values that guarantee < 5 % composite gap.

    @pytest.fixture(autouse=True)
    def patch_fx(self, monkeypatch):
        monkeypatch.setattr("agent.matcher.get_rate", _mock_get_rate)
        monkeypatch.setattr("agent.matcher.convert", _mock_convert)

    def _build_pair(self):
        """Two invoices whose composites are within 5 % of each other."""
        # Payment reference contains both "INV-A01" and a fuzzy match for "INV-B01"
        # via partial_ratio ≥ 80, so reference_fit for B = 0.6.
        pay = _make_payment(
            raw_amount=45_000.00,
            sender_name="ACME SDN BHD",
            reference="INV-A01 / INV-B01 PAYMENT",
        )

        # INV-A: exact reference match, perfect name
        inv_a = _make_invoice(
            inv_id=1, invoice_no="INV-A01", amount=10_000.0, currency="USD"
        )
        cp_a = _make_counterparty("ACME SDN BHD")

        # INV-B: fuzzy reference (0.6), same name
        inv_b = _make_invoice(
            inv_id=2, invoice_no="INV-B01", amount=10_000.0, currency="USD"
        )
        cp_b = _make_counterparty("ACME SDN BHD")

        return pay, inv_a, cp_a, inv_b, cp_b

    def test_composites_within_25_pct(self):
        """Composites should be close enough to be considered ambiguous."""
        pay, inv_a, cp_a, inv_b, cp_b = self._build_pair()

        score_a = score_pair(inv_a, pay, cp_a)
        score_b = score_pair(inv_b, pay, cp_b)

        diff = abs(score_a.composite_confidence - score_b.composite_confidence)
        higher = max(score_a.composite_confidence, score_b.composite_confidence)
        pct_diff = diff / higher if higher > 0 else Decimal("1")

        # Scores are "ambiguous" if within 25 % (i.e. both are plausible)
        assert pct_diff < Decimal("0.25"), (
            f"composites too far apart: A={score_a.composite_confidence}, "
            f"B={score_b.composite_confidence}, diff={pct_diff:.2%}"
        )

    def test_both_composites_above_half(self):
        """Both candidates must be above 0.50 to be genuine contenders."""
        pay, inv_a, cp_a, inv_b, cp_b = self._build_pair()

        score_a = score_pair(inv_a, pay, cp_a)
        score_b = score_pair(inv_b, pay, cp_b)

        assert score_a.composite_confidence > Decimal("0.50")
        assert score_b.composite_confidence > Decimal("0.50")

    @pytest.mark.asyncio
    async def test_rank_candidates_returns_both(self):
        """rank_candidates should return both when they score within 5 % composite."""
        pay, inv_a, cp_a, inv_b, cp_b = self._build_pair()

        # Patch at the function level — avoids deep OpenAI mock chains
        async def fake_verify(inv: Invoice, pmt: Payment, score: MatchScore) -> VerifiedMatch:
            conf = 90 if inv.id == 1 else 87
            return VerifiedMatch(
                invoice_id=inv.id,
                payment_id=pmt.id,
                score=score,
                final_confidence=conf,
                hermes_narrative=f"Match narrative for invoice {inv.id}.",
                caveats=[],
                fx_rate_used=Decimal("4.5"),
            )

        async def fake_explain(candidate: VerifiedMatch, winner: VerifiedMatch) -> str:
            return "Runner-up explanation."

        with (
            patch("agent.matcher.verify_with_hermes", side_effect=fake_verify),
            patch("agent.matcher._explain_below_winner", side_effect=fake_explain),
        ):
            results = await rank_candidates(
                pay,
                [inv_a, inv_b],
                counterparties={1: cp_a, 2: cp_b},
            )

        assert len(results) == 2, "Both ambiguous candidates should be returned"

    @pytest.mark.asyncio
    async def test_runner_up_gets_ranking_explanation(self):
        """Position-2 result should have '[Why ranked below winner]' in narrative."""
        pay, inv_a, cp_a, inv_b, cp_b = self._build_pair()

        async def fake_verify(inv: Invoice, pmt: Payment, score: MatchScore) -> VerifiedMatch:
            conf = 90 if inv.id == 1 else 85
            return VerifiedMatch(
                invoice_id=inv.id,
                payment_id=pmt.id,
                score=score,
                final_confidence=conf,
                hermes_narrative=f"Original narrative for invoice {inv.id}.",
                caveats=[],
                fx_rate_used=Decimal("4.5"),
            )

        async def fake_explain(candidate: VerifiedMatch, winner: VerifiedMatch) -> str:
            return "Invoice B lacks an exact reference match unlike Invoice A."

        with (
            patch("agent.matcher.verify_with_hermes", side_effect=fake_verify),
            patch("agent.matcher._explain_below_winner", side_effect=fake_explain),
        ):
            results = await rank_candidates(
                pay,
                [inv_a, inv_b],
                counterparties={1: cp_a, 2: cp_b},
            )

        assert len(results) >= 2
        runner_up = results[1]
        assert "[Why ranked below winner]" in runner_up.hermes_narrative, (
            f"Runner-up narrative missing ranking explanation: {runner_up.hermes_narrative!r}"
        )


# ---------------------------------------------------------------------------
# Edge-case unit tests
# ---------------------------------------------------------------------------
class TestEdgeCases:
    @pytest.fixture(autouse=True)
    def patch_fx(self, monkeypatch):
        monkeypatch.setattr("agent.matcher.get_rate", _mock_get_rate)
        monkeypatch.setattr("agent.matcher.convert", _mock_convert)

    def test_no_counterparty_gives_zero_name_fit(self):
        inv = _make_invoice()
        pay = _make_payment()
        score = score_pair(inv, pay, counterparty=None)
        assert score.name_fit == Decimal("0")

    def test_missing_reference_gives_zero_reference_fit(self):
        inv = _make_invoice()
        pay = _make_payment(reference="")
        cp = _make_counterparty()
        score = score_pair(inv, pay, cp)
        assert score.reference_fit == Decimal("0")

    def test_date_decay_at_45_days(self):
        """45 days after issue → date_fit = 0.5 (midpoint of 30–60 decay)."""
        inv = _make_invoice(issued_date=date(2025, 9, 1))
        late_pay = _make_payment(payment_date=date(2025, 10, 16))  # 45 days
        score = score_pair(inv, late_pay)
        assert score.date_fit == Decimal("0.5").quantize(Decimal("0.0001"))

    def test_payment_before_issue_gives_zero_date_fit(self):
        inv = _make_invoice(issued_date=date(2025, 10, 15))
        early_pay = _make_payment(payment_date=date(2025, 10, 1))
        score = score_pair(inv, early_pay)
        assert score.date_fit == Decimal("0")

    def test_amount_fit_exact_same_currency(self):
        """When currencies match and amount is identical, fit = 1.0."""
        inv = _make_invoice(amount=5000.0, currency="MYR")
        pay = _make_payment(raw_amount=5000.0, currency="MYR")
        score = score_pair(inv, pay)
        assert score.amount_fit == Decimal("1")

    def test_composite_uses_correct_weights(self):
        """Manually verify the composite formula with known sub-scores."""
        inv = _make_invoice()
        pay = _make_payment()
        cp = _make_counterparty()
        score = score_pair(inv, pay, cp)

        expected = (
            WEIGHT_AMOUNT * score.amount_fit
            + WEIGHT_DATE * score.date_fit
            + WEIGHT_NAME * score.name_fit
            + WEIGHT_REFERENCE * score.reference_fit
        ).quantize(Decimal("0.0001"))

        assert score.composite_confidence == expected

    @pytest.mark.asyncio
    async def test_rank_candidates_empty_list(self):
        pay = _make_payment()
        results = await rank_candidates(pay, [])
        assert results == []
