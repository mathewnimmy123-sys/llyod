"""Deterministic UK pension tax-relief maths (pure Python, no ADK).

Correction vs the original POC: it computed gross = net / (1 - rate) for *every* band.
That is only right for basic rate. Under relief-at-source the provider claims 20% on
the customer's behalf (gross = net / 0.8); higher/additional-rate taxpayers then claim
the *extra* (rate - 20%) via Self Assessment, which does NOT increase the amount in the
pension. The policy document (Section 1) says exactly this.

Money is handled with Decimal to avoid float drift; results are returned as floats/strings
so they serialise cleanly for the LLM.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

ANNUAL_ALLOWANCE = Decimal("60000")
BASIC_RATE = Decimal("0.20")
BAND_RATES = {"basic": Decimal("0.20"), "higher": Decimal("0.40"), "additional": Decimal("0.45")}
_MAX = Decimal("100000000")
_CENT = Decimal("0.01")


class CalculationError(ValueError):
    """Raised for invalid input; the tool wrapper turns this into an error payload."""


def _money(value: object, name: str) -> Decimal:
    if value is None or isinstance(value, bool):
        raise CalculationError(f"{name} must be a number")
    if isinstance(value, str):
        value = value.replace("£", "").replace(",", "").strip()
    try:
        d = Decimal(str(value))
    except InvalidOperation as exc:
        raise CalculationError(f"{name} must be a number") from exc
    if not d.is_finite():
        raise CalculationError(f"{name} must be a finite number")
    if d < 0:
        raise CalculationError(f"{name} cannot be negative")
    if d > _MAX:
        raise CalculationError(f"{name} is implausibly large")
    return d


def _q(d: Decimal) -> Decimal:
    return d.quantize(_CENT, rounding=ROUND_HALF_UP)


def calculate_relief(
    net_contribution_gbp: object,
    tax_band: str = "basic",
    employer_contribution_gbp: object = 0,
    annual_earnings_gbp: object = 0,
) -> dict:
    """net_contribution_gbp is what the customer pays from their own net pay.
    annual_earnings_gbp: 0 / omitted = unknown (earnings limit not checked)."""
    net = _money(net_contribution_gbp, "contribution_gbp")
    employer = _money(employer_contribution_gbp, "employer_contribution_gbp")
    earnings = _money(annual_earnings_gbp, "annual_earnings_gbp")
    band = str(tax_band).strip().lower()
    if band not in BAND_RATES:
        raise CalculationError(f"tax_band must be one of {sorted(BAND_RATES)}; got '{tax_band}'")
    rate = BAND_RATES[band]

    gross = _q(net / (1 - BASIC_RATE))
    basic_relief = _q(gross - net)
    extra_relief = _q(gross * (rate - BASIC_RATE))  # claimed via Self Assessment (0 for basic)
    counted = gross + employer
    remaining = max(Decimal(0), ANNUAL_ALLOWANCE - counted)
    excess = max(Decimal(0), counted - ANNUAL_ALLOWANCE)

    notes = [
        "Assumes a relief-at-source scheme and England/Wales/NI income tax bands "
        "(Scottish bands differ).",
        "Annual allowance counts personal gross + employer contributions. Tapered allowance, "
        "carry-forward and the money purchase annual allowance are NOT modelled.",
    ]
    result = {
        "status": "success",
        "net_user_contribution_gbp": float(_q(net)),
        "tax_band": band,
        "basic_rate_relief_added_by_provider_gbp": float(basic_relief),
        "additional_relief_claimable_via_self_assessment_gbp": float(extra_relief),
        "total_estimated_tax_relief_gbp": float(basic_relief + extra_relief),
        "effective_net_cost_after_all_relief_gbp": float(_q(net - extra_relief)),
        "total_gross_into_pension_gbp": float(gross),
        "employer_contribution_gbp": float(_q(employer)),
        "total_counted_against_annual_allowance_gbp": float(_q(counted)),
        "remaining_annual_allowance_gbp": float(_q(remaining)),
        "exceeds_annual_allowance": counted > ANNUAL_ALLOWANCE,
        "excess_over_allowance_gbp": float(_q(excess)),
    }
    if earnings > 0:
        result["exceeds_earnings_limit_for_tax_relief"] = gross > earnings
        notes.append("Personal contributions only attract tax relief up to 100% of relevant UK earnings.")
    result["assumptions"] = notes
    return result
