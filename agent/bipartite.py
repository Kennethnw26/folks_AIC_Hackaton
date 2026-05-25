"""Bipartite matching with lump-sum and partial-payment detection.

Builds on agent/matcher.py (score_pair, _async_client, _model).

Processing order in assign():
  1. detect_lump_sum  — payment covers several invoices in bulk
  2. detect_partial   — payment covers a fraction of one invoice
  3. solve_assignment — bipartite optimum for the remainder
  4. escalate_ambiguous — Hermes review when bipartite confidence < 60 %

All monetary arithmetic uses Decimal; amounts in the payment's currency.
"""
from __future__ import annotations

import json
import logging
from decimal import Decimal, ROUND_HALF_UP
from typing import Literal, Optional

import numpy as np
from pydantic import BaseModel
from scipy.optimize import linear_sum_assignment

from agent.matcher import _async_client, _model, score_pair
from db.models import Invoice, Payment
from fx.rates import get_rate

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tolerances / thresholds
# ---------------------------------------------------------------------------
_LUMP_SUM_TOLERANCE: Decimal = Decimal("0.015")   # 1.5 % of payment amount
_PARTIAL_MIN_PCT: Decimal = Decimal("0.30")        # 30 %
_PARTIAL_MAX_PCT: Decimal = Decimal("0.95")        # 95 %
_PARTIAL_NAME_MIN: Decimal = Decimal("0.7")
_PARTIAL_REF_MIN: Decimal = Decimal("0.5")
_BIPARTITE_MIN_CONF: Decimal = Decimal("0.50")
_ESCALATION_THRESHOLD: Decimal = Decimal("0.60")

_DP4 = Decimal("0.0001")
_DP2 = Decimal("0.01")


def _q4(d: Decimal) -> Decimal:
    return d.quantize(_DP4, rounding=ROUND_HALF_UP)


def _q2(d: Decimal) -> Decimal:
    return d.quantize(_DP2, rounding=ROUND_HALF_UP)


def _converted(invoice: Invoice, payment: Payment) -> Decimal:
    """Invoice amount converted into payment.currency on payment.payment_date."""
    rate = get_rate(invoice.currency, payment.currency, payment.payment_date)
    return _q4(Decimal(str(invoice.amount)) * rate)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class Assignment(BaseModel):
    """One resolved payment-to-invoice allocation."""

    payment_id: int
    invoice_ids: list[int]
    # Allocated amounts are in payment.currency
    allocations: dict[int, Decimal]
    confidence: Decimal
    kind: Literal["one_to_one", "lump_sum", "partial", "many_to_many"]
    # Positive residual  → invoices not fully settled by this payment.
    # Negative residual  → payment exceeds total invoice value (overpayment).
    residual_amount: Decimal
    narrative: str


class AssignmentResult(BaseModel):
    """Complete reconciliation result for a batch of payments and invoices."""

    assignments: list[Assignment]
    unassigned_payments: list[int]   # payment IDs
    unassigned_invoices: list[int]   # invoice IDs
    overall_narrative: str


# ---------------------------------------------------------------------------
# Hermes JSON-schema contracts
# ---------------------------------------------------------------------------
_ESCALATION_SCHEMA: dict = {
    "type": "json_schema",
    "json_schema": {
        "name": "escalation_result",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "assignments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "payment_id": {"type": "integer"},
                            "invoice_ids": {
                                "type": "array",
                                "items": {"type": "integer"},
                            },
                            "confidence": {
                                "type": "number",
                                "description": "0–1 float",
                            },
                            "kind": {
                                "type": "string",
                                "enum": [
                                    "one_to_one",
                                    "lump_sum",
                                    "partial",
                                    "many_to_many",
                                ],
                            },
                            "narrative": {"type": "string"},
                        },
                        "required": [
                            "payment_id",
                            "invoice_ids",
                            "confidence",
                            "kind",
                            "narrative",
                        ],
                        "additionalProperties": False,
                    },
                },
                "unassigned_payment_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                },
                "overall_narrative": {"type": "string"},
            },
            "required": [
                "assignments",
                "unassigned_payment_ids",
                "overall_narrative",
            ],
            "additionalProperties": False,
        },
    },
}


# ---------------------------------------------------------------------------
# 1. solve_assignment
# ---------------------------------------------------------------------------
def solve_assignment(
    payments: list[Payment],
    invoices: list[Invoice],
) -> AssignmentResult:
    """Bipartite optimum matching via scipy linear_sum_assignment.

    Args:
        payments: Unmatched payments to assign.
        invoices: Open invoices to receive assignments.

    Returns:
        AssignmentResult; items with composite_confidence < 0.50 become
        unassigned.
    """
    if not payments or not invoices:
        return AssignmentResult(
            assignments=[],
            unassigned_payments=[p.id for p in payments],  # type: ignore[misc]
            unassigned_invoices=[i.id for i in invoices],  # type: ignore[misc]
            overall_narrative="No payments or invoices to match.",
        )

    n_pay = len(payments)
    n_inv = len(invoices)
    dim = max(n_pay, n_inv)

    # Build cost matrix (dummies default to cost 1.0 → confidence 0.0)
    cost = np.ones((dim, dim), dtype=float)
    scores: dict[tuple[int, int], Decimal] = {}   # (pay_idx, inv_idx) -> confidence

    for pi, payment in enumerate(payments):
        for ii, invoice in enumerate(invoices):
            ms = score_pair(invoice, payment)
            conf = ms.composite_confidence
            scores[(pi, ii)] = conf
            cost[pi, ii] = float(Decimal("1") - conf)

    row_ind, col_ind = linear_sum_assignment(cost)

    assignments: list[Assignment] = []
    unassigned_pay: list[int] = []
    unassigned_inv: list[int] = list(range(n_inv))   # start full, remove matched

    matched_inv_indices: set[int] = set()

    for pi, ii in zip(row_ind, col_ind):
        # Skip dummy rows/cols
        if pi >= n_pay or ii >= n_inv:
            continue

        conf = scores[(pi, ii)]
        if conf < _BIPARTITE_MIN_CONF:
            unassigned_pay.append(payments[pi].id)  # type: ignore[arg-type]
            continue

        payment = payments[pi]
        invoice = invoices[ii]
        conv = _converted(invoice, payment)
        pay_amount = Decimal(str(payment.raw_amount))
        residual = _q2(conv - pay_amount)

        assignments.append(
            Assignment(
                payment_id=payment.id,  # type: ignore[arg-type]
                invoice_ids=[invoice.id],  # type: ignore[arg-type]
                allocations={invoice.id: pay_amount},  # type: ignore[arg-type]
                confidence=_q4(conf),
                kind="one_to_one",
                residual_amount=residual,
                narrative=(
                    f"Bipartite optimum: payment {payment.id} → invoice "
                    f"{invoice.id} (confidence {conf:.2%})."
                ),
            )
        )
        matched_inv_indices.add(ii)

    unassigned_inv_ids = [
        invoices[ii].id  # type: ignore[misc]
        for ii in range(n_inv)
        if ii not in matched_inv_indices
    ]

    return AssignmentResult(
        assignments=assignments,
        unassigned_payments=unassigned_pay,
        unassigned_invoices=unassigned_inv_ids,
        overall_narrative=(
            f"Bipartite solver: {len(assignments)} assignment(s) from "
            f"{n_pay} payment(s) and {n_inv} invoice(s)."
        ),
    )


# ---------------------------------------------------------------------------
# 2. detect_lump_sum
# ---------------------------------------------------------------------------
def detect_lump_sum(
    payment: Payment,
    invoices: list[Invoice],
) -> Optional[Assignment]:
    """Detect whether payment covers several invoices as a bulk settlement.

    Algorithm:
    1. Identify invoices whose invoice_no appears literally in payment.reference
       (reference hits); place them first.
    2. Sort remaining invoices oldest-first (by issued_date).
    3. Accumulate converted amounts from front to back; when the running sum
       is within 1.5 % of the payment amount, return a lump-sum Assignment.
    4. If no prefix sum matches, return None.

    Args:
        payment:  Incoming payment.
        invoices: Candidate open invoices (any length).

    Returns:
        Assignment with kind="lump_sum", or None if no match found.
    """
    if not invoices:
        return None

    pay_amount = Decimal(str(payment.raw_amount))
    ref = (payment.reference or "").strip()

    # --- Priority ordering ---------------------------------------------------
    ref_hit_ids: set[int] = set()
    if ref:
        for inv in invoices:
            inv_no = (inv.invoice_no or "").strip()
            if inv_no and inv_no in ref:
                ref_hit_ids.add(inv.id)  # type: ignore[arg-type]

    ref_hits = [i for i in invoices if i.id in ref_hit_ids]
    rest = sorted(
        [i for i in invoices if i.id not in ref_hit_ids],
        key=lambda i: i.issued_date,
    )
    ordered = ref_hits + rest

    # --- Cumulative sum search -----------------------------------------------
    running = Decimal("0")
    selected: list[Invoice] = []

    for inv in ordered:
        conv = _converted(inv, payment)
        running = _q4(running + conv)
        selected.append(inv)

        if pay_amount == Decimal("0"):
            break

        gap_pct = abs(running - pay_amount) / pay_amount
        if gap_pct <= _LUMP_SUM_TOLERANCE:
            # Build allocations: each invoice gets its full converted amount.
            # Residual = payment_amount − sum(converted) (small, due to tolerance).
            allocations: dict[int, Decimal] = {
                inv.id: _q2(_converted(inv, payment))  # type: ignore[index]
                for inv in selected
            }
            residual = _q2(running - pay_amount)
            inv_ids = [inv.id for inv in selected]  # type: ignore[misc]

            return Assignment(
                payment_id=payment.id,  # type: ignore[arg-type]
                invoice_ids=inv_ids,
                allocations=allocations,
                confidence=_q4(Decimal("1") - gap_pct),
                kind="lump_sum",
                residual_amount=residual,
                narrative=(
                    f"Lump-sum detected: payment {payment.id} "
                    f"({payment.raw_amount} {payment.currency}) covers "
                    f"{len(selected)} invoice(s) "
                    f"({', '.join(str(i) for i in inv_ids)}); "
                    f"gap {gap_pct:.3%} within 1.5 % tolerance."
                ),
            )

        # Already over-shot by more than tolerance — stop accumulating
        if running > pay_amount * (Decimal("1") + _LUMP_SUM_TOLERANCE):
            break

    return None


# ---------------------------------------------------------------------------
# 3. detect_partial
# ---------------------------------------------------------------------------
def detect_partial(
    payment: Payment,
    invoice: Invoice,
) -> Optional[Assignment]:
    """Detect whether payment is a partial settlement of a single invoice.

    Conditions (all must hold):
    - payment amount is 30 %–95 % of invoice converted amount.
    - score_pair name_fit > 0.70.
    - score_pair reference_fit > 0.50.

    Args:
        payment: Incoming payment.
        invoice: Candidate invoice.

    Returns:
        Assignment with kind="partial", or None.
    """
    conv = _converted(invoice, payment)
    if conv == Decimal("0"):
        return None

    pay_amount = Decimal(str(payment.raw_amount))
    ratio = _q4(pay_amount / conv)

    if not (_PARTIAL_MIN_PCT <= ratio <= _PARTIAL_MAX_PCT):
        return None

    ms = score_pair(invoice, payment)
    if ms.name_fit <= _PARTIAL_NAME_MIN or ms.reference_fit <= _PARTIAL_REF_MIN:
        return None

    residual = _q2(conv - pay_amount)   # positive: remaining outstanding

    # Confidence blends ratio coverage with name and reference fits
    confidence = _q4(
        (ms.name_fit + ms.reference_fit) / Decimal("2") * ratio
    )

    return Assignment(
        payment_id=payment.id,  # type: ignore[arg-type]
        invoice_ids=[invoice.id],  # type: ignore[arg-type]
        allocations={invoice.id: _q2(pay_amount)},  # type: ignore[index]
        confidence=confidence,
        kind="partial",
        residual_amount=residual,
        narrative=(
            f"Partial payment detected: payment {payment.id} "
            f"({payment.raw_amount} {payment.currency}) covers "
            f"{ratio:.1%} of invoice {invoice.id} "
            f"({invoice.amount} {invoice.currency} → {conv} {payment.currency}); "
            f"outstanding residual {residual} {payment.currency}."
        ),
    )


# ---------------------------------------------------------------------------
# 4. escalate_ambiguous
# ---------------------------------------------------------------------------
async def escalate_ambiguous(
    payments: list[Payment],
    invoices: list[Invoice],
    low_confidence_pairs: list[tuple[Payment, Invoice, Decimal]],
) -> AssignmentResult:
    """Ask Hermes 4 to resolve low-confidence payment-invoice candidates.

    Args:
        payments:             Full payment list (for context).
        invoices:             Full invoice list (for context).
        low_confidence_pairs: List of (payment, invoice, confidence) tuples
                              where bipartite confidence fell below 60 %.

    Returns:
        AssignmentResult with Hermes-proposed assignments.
    """
    # Summarise each candidate pair for the prompt
    pair_lines: list[str] = []
    for pay, inv, conf in low_confidence_pairs:
        conv = _converted(inv, pay)
        pair_lines.append(
            f"  - Payment {pay.id} ({pay.raw_amount} {pay.currency}, "
            f"ref='{pay.reference}', sender='{pay.sender_name}') ↔ "
            f"Invoice {inv.id} ({inv.amount} {inv.currency} → "
            f"{conv} {pay.currency}, no={inv.invoice_no}), "
            f"bipartite_confidence={conf:.2%}"
        )

    prompt = (
        "You are a senior treasury reconciliation analyst. "
        "The bipartite solver produced the following low-confidence "
        "payment-invoice candidates (confidence < 60 %). "
        "Review each pair and propose final allocations.\n\n"
        "LOW-CONFIDENCE PAIRS:\n"
        + "\n".join(pair_lines)
        + "\n\n"
        "For each payment, decide which invoice(s) it should be matched to "
        "(or leave unassigned). Return a JSON object conforming to the schema.\n"
        "Possible kinds: one_to_one, lump_sum, partial, many_to_many.\n"
        "confidence: your 0–1 estimate of correctness.\n"
    )

    client = _async_client()
    resp = await client.chat.completions.create(
        model=_model(),
        messages=[{"role": "user", "content": prompt}],
        response_format=_ESCALATION_SCHEMA,  # type: ignore[arg-type]
        temperature=0,
    )
    data = json.loads(resp.choices[0].message.content)

    # Build Assignment objects from Hermes output
    inv_map: dict[int, Invoice] = {inv.id: inv for inv in invoices}  # type: ignore[index]
    pay_map: dict[int, Payment] = {pay.id: pay for pay in payments}  # type: ignore[index]

    assignments: list[Assignment] = []
    for item in data["assignments"]:
        pay_id: int = item["payment_id"]
        inv_ids: list[int] = item["invoice_ids"]
        hermes_conf = _q4(Decimal(str(item["confidence"])))
        kind = item["kind"]

        pay = pay_map.get(pay_id)
        if pay is None:
            logger.warning("Hermes returned unknown payment_id %d", pay_id)
            continue

        pay_amount = Decimal(str(pay.raw_amount))

        # Allocations: split payment amount proportionally across invoices
        inv_list = [inv_map[i] for i in inv_ids if i in inv_map]
        if not inv_list:
            continue

        if len(inv_list) == 1:
            allocations: dict[int, Decimal] = {inv_list[0].id: _q2(pay_amount)}  # type: ignore[index]
            conv = _converted(inv_list[0], pay)
            residual = _q2(conv - pay_amount)
        else:
            # Proportional split by converted invoice amounts
            convs = [_converted(inv, pay) for inv in inv_list]
            total_conv = sum(convs, Decimal("0"))
            if total_conv == Decimal("0"):
                allocations = {inv.id: Decimal("0") for inv in inv_list}  # type: ignore[index]
                residual = Decimal("0")
            else:
                allocations = {}
                for inv, conv in zip(inv_list, convs):
                    share = _q2(pay_amount * conv / total_conv)
                    allocations[inv.id] = share  # type: ignore[index]
                residual = _q2(total_conv - pay_amount)

        assignments.append(
            Assignment(
                payment_id=pay_id,
                invoice_ids=inv_ids,
                allocations=allocations,
                confidence=hermes_conf,
                kind=kind,
                residual_amount=residual,
                narrative=item["narrative"],
            )
        )

    hermes_assigned_pay_ids = {a.payment_id for a in assignments}
    original_pay_ids = {pay.id for pay in payments}  # type: ignore[misc]
    unassigned_pay_ids = list(
        (set(data.get("unassigned_payment_ids", [])) | (original_pay_ids - hermes_assigned_pay_ids))
        & original_pay_ids
    )

    original_inv_ids = {inv.id for inv in invoices}  # type: ignore[misc]
    assigned_inv_ids: set[int] = set()
    for a in assignments:
        assigned_inv_ids.update(a.invoice_ids)
    unassigned_inv_ids = list(original_inv_ids - assigned_inv_ids)

    return AssignmentResult(
        assignments=assignments,
        unassigned_payments=unassigned_pay_ids,
        unassigned_invoices=unassigned_inv_ids,
        overall_narrative=data.get("overall_narrative", "Hermes escalation complete."),
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
async def assign(
    payments: list[Payment],
    invoices: list[Invoice],
) -> AssignmentResult:
    """Orchestrate full reconciliation for a batch of payments and invoices.

    Processing order:
    1. detect_lump_sum  — each payment vs. all remaining open invoices.
    2. detect_partial   — each remaining payment vs. each remaining invoice.
    3. solve_assignment — bipartite for the remainder.
    4. escalate_ambiguous — Hermes review for bipartite pairs < 60 % conf.

    Args:
        payments: Unreconciled payments.
        invoices: Open invoices.

    Returns:
        AssignmentResult aggregating all phases.
    """
    all_assignments: list[Assignment] = []

    # Mutable pools
    remaining_pay: dict[int, Payment] = {p.id: p for p in payments}  # type: ignore[index]
    remaining_inv: dict[int, Invoice] = {i.id: i for i in invoices}  # type: ignore[index]

    # ------------------------------------------------------------------
    # Phase 1 — lump-sum detection
    # ------------------------------------------------------------------
    for pay_id, pay in list(remaining_pay.items()):
        inv_pool = list(remaining_inv.values())
        result = detect_lump_sum(pay, inv_pool)
        if result is not None:
            all_assignments.append(result)
            del remaining_pay[pay_id]
            for inv_id in result.invoice_ids:
                remaining_inv.pop(inv_id, None)
            logger.info(
                "Lump-sum: payment %d → invoices %s",
                pay_id,
                result.invoice_ids,
            )

    # ------------------------------------------------------------------
    # Phase 2 — partial-payment detection
    # ------------------------------------------------------------------
    for pay_id, pay in list(remaining_pay.items()):
        for inv_id, inv in list(remaining_inv.items()):
            result = detect_partial(pay, inv)
            if result is not None:
                all_assignments.append(result)
                del remaining_pay[pay_id]
                del remaining_inv[inv_id]
                logger.info(
                    "Partial: payment %d → invoice %d", pay_id, inv_id
                )
                break  # one partial match per payment

    # ------------------------------------------------------------------
    # Phase 3 — bipartite for the rest
    # ------------------------------------------------------------------
    rem_pay_list = list(remaining_pay.values())
    rem_inv_list = list(remaining_inv.values())

    bipartite_result = solve_assignment(rem_pay_list, rem_inv_list)

    # Separate high-confidence assignments from low-confidence pairs
    low_confidence_pairs: list[tuple[Payment, Invoice, Decimal]] = []
    pay_map: dict[int, Payment] = {p.id: p for p in rem_pay_list}  # type: ignore[index]
    inv_map: dict[int, Invoice] = {i.id: i for i in rem_inv_list}  # type: ignore[index]

    for asgn in bipartite_result.assignments:
        if asgn.confidence >= _ESCALATION_THRESHOLD:
            all_assignments.append(asgn)
            remaining_pay.pop(asgn.payment_id, None)
            for inv_id in asgn.invoice_ids:
                remaining_inv.pop(inv_id, None)
        else:
            # Queue for LLM escalation
            pay = pay_map.get(asgn.payment_id)
            if pay and asgn.invoice_ids:
                inv = inv_map.get(asgn.invoice_ids[0])
                if inv:
                    low_confidence_pairs.append((pay, inv, asgn.confidence))

    # ------------------------------------------------------------------
    # Phase 4 — escalate low-confidence bipartite pairs
    # ------------------------------------------------------------------
    if low_confidence_pairs:
        lc_payments = [t[0] for t in low_confidence_pairs]
        lc_invoices = [t[1] for t in low_confidence_pairs]

        escalated = await escalate_ambiguous(
            payments=lc_payments,
            invoices=lc_invoices,
            low_confidence_pairs=low_confidence_pairs,
        )
        all_assignments.extend(escalated.assignments)

        assigned_by_escalation: set[int] = set()
        for a in escalated.assignments:
            assigned_by_escalation.add(a.payment_id)
            for inv_id in a.invoice_ids:
                remaining_inv.pop(inv_id, None)
        for pay in lc_payments:
            if pay.id in assigned_by_escalation:
                remaining_pay.pop(pay.id, None)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Collect still-unassigned items
    # ------------------------------------------------------------------
    # Items that were in the bipartite unassigned lists and never escalated
    escalated_pay_ids = {t[0].id for t in low_confidence_pairs}
    final_unassigned_pay = list(
        set(remaining_pay.keys())
        | (set(bipartite_result.unassigned_payments) - escalated_pay_ids)
    )
    final_unassigned_inv = list(
        set(remaining_inv.keys())
        | (set(bipartite_result.unassigned_invoices) - {t[1].id for t in low_confidence_pairs})
    )

    phase_summary = (
        f"Phase 1 (lump-sum): {sum(1 for a in all_assignments if a.kind == 'lump_sum')} match(es). "
        f"Phase 2 (partial): {sum(1 for a in all_assignments if a.kind == 'partial')} match(es). "
        f"Phase 3 (bipartite): {sum(1 for a in all_assignments if a.kind == 'one_to_one')} match(es). "
        f"Phase 4 (escalated): {sum(1 for a in all_assignments if a.kind not in ('lump_sum', 'partial', 'one_to_one'))} match(es). "
        f"Unassigned payments: {len(final_unassigned_pay)}. "
        f"Unassigned invoices: {len(final_unassigned_inv)}."
    )

    return AssignmentResult(
        assignments=all_assignments,
        unassigned_payments=final_unassigned_pay,
        unassigned_invoices=final_unassigned_inv,
        overall_narrative=phase_summary,
    )
