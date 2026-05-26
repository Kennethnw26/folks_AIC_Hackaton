from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from rapidfuzz import fuzz

from db.models import Invoice, Payment


@dataclass
class CandidateScore:
    invoice_id: int
    invoice_number: str
    amount: float
    currency: str
    beneficiary_name: str
    det_score: float


def compute_det_score(
    payment: Payment,
    invoice: Invoice,
    p_norm: float,
    i_norm: float,
) -> float:
    if i_norm == 0:
        amount_score = 0.0
    else:
        amount_score = max(0.0, 1.0 - abs(p_norm - i_norm) / i_norm)

    currency_score = 1.0 if payment.currency == invoice.currency else 0.7

    p_date = date.fromisoformat(payment.date) if isinstance(payment.date, str) else payment.date
    days_delta = (p_date - invoice.due_date).days
    date_score = math.exp(-(days_delta ** 2) / (2 * 14 ** 2))

    ref_score = fuzz.token_sort_ratio(payment.beneficiary_name, invoice.beneficiary_name) / 100.0

    det = 0.4 * amount_score + 0.2 * currency_score + 0.2 * date_score + 0.2 * ref_score
    return round(det, 6)
