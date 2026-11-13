import math

import pytest

from invest_advisor.calculator import CalculationError, calculate_relief


def test_basic_rate_80_becomes_100():
    r = calculate_relief(8000, "basic")
    assert r["total_gross_into_pension_gbp"] == 10000.0
    assert r["basic_rate_relief_added_by_provider_gbp"] == 2000.0
    assert r["additional_relief_claimable_via_self_assessment_gbp"] == 0.0
    assert r["effective_net_cost_after_all_relief_gbp"] == 8000.0


def test_higher_rate_extra_relief_does_not_inflate_pension():
    # Original POC returned gross 13,333 for 8,000 @40%; correct gross is still 10,000.
    r = calculate_relief(8000, "higher")
    assert r["total_gross_into_pension_gbp"] == 10000.0
    assert r["additional_relief_claimable_via_self_assessment_gbp"] == 2000.0
    assert r["total_estimated_tax_relief_gbp"] == 4000.0
    assert r["effective_net_cost_after_all_relief_gbp"] == 6000.0


def test_additional_rate():
    r = calculate_relief(8000, "Additional")
    assert r["additional_relief_claimable_via_self_assessment_gbp"] == 2500.0


def test_allowance_includes_employer_and_flags_excess():
    r = calculate_relief(40000, "higher", employer_contribution_gbp=15000)  # gross 50,000 + 15,000
    assert r["total_counted_against_annual_allowance_gbp"] == 65000.0
    assert r["remaining_annual_allowance_gbp"] == 0.0
    assert r["exceeds_annual_allowance"] is True and r["excess_over_allowance_gbp"] == 5000.0


def test_remaining_allowance():
    r = calculate_relief(8000, "basic")
    assert r["remaining_annual_allowance_gbp"] == 50000.0 and r["exceeds_annual_allowance"] is False


def test_earnings_limit_flag_only_when_earnings_given():
    assert "exceeds_earnings_limit_for_tax_relief" not in calculate_relief(8000)
    assert calculate_relief(8000, annual_earnings_gbp=9000)["exceeds_earnings_limit_for_tax_relief"] is True
    assert calculate_relief(8000, annual_earnings_gbp=50000)["exceeds_earnings_limit_for_tax_relief"] is False


def test_rounding_is_two_dp():
    assert calculate_relief(100.01)["total_gross_into_pension_gbp"] == 125.01


@pytest.mark.parametrize("bad", [-1, "abc", None, True, math.nan, math.inf, 1e12])
def test_invalid_amounts_rejected(bad):
    with pytest.raises(CalculationError):
        calculate_relief(bad)


def test_unknown_band_rejected_not_silently_basic():
    with pytest.raises(CalculationError):
        calculate_relief(1000, "super")


def test_string_amounts_tolerated():
    assert calculate_relief("£8,000")["total_gross_into_pension_gbp"] == 10000.0
