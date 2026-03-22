from __future__ import annotations

from dataclasses import dataclass

from .phase2_config import FIXED_TENOR_YEARS, FLOAT_TENOR_YEARS, PARALLEL_SHOCK_BP, PHASE2_NOTIONAL
from .phase2_curves import curve_discount_factor, parallel_shift_curve, quarter_schedule


@dataclass
class SwapValuation:
    maturity_years: float
    fixed_coupon: float
    fixed_leg_pv: float
    floating_leg_pv: float
    receiver_fixed_pv: float
    payer_fixed_pv: float
    par_swap_rate: float
    annuity: float
    pv01: float
    dv01: float


def _projection_forward_rates(projection_curve, maturity_years: float) -> list[tuple[float, float]]:
    schedule = quarter_schedule(maturity_years, FLOAT_TENOR_YEARS)
    forwards: list[tuple[float, float]] = []
    prev_time = 0.0
    prev_discount = 1.0
    for pay_time in schedule:
        discount_factor = curve_discount_factor(projection_curve, pay_time)
        delta = pay_time - prev_time
        forward = (prev_discount / discount_factor - 1.0) / delta
        forwards.append((pay_time, forward))
        prev_time = pay_time
        prev_discount = discount_factor
    return forwards


def price_swap(discount_curve, projection_curve, maturity_years: float, fixed_coupon: float, notional: float = PHASE2_NOTIONAL) -> SwapValuation:
    fixed_schedule = quarter_schedule(maturity_years, FIXED_TENOR_YEARS)
    annuity = sum(FIXED_TENOR_YEARS * curve_discount_factor(discount_curve, pay_time) for pay_time in fixed_schedule)
    fixed_leg_pv = notional * fixed_coupon * annuity

    floating_leg_pv = 0.0
    for pay_time, forward in _projection_forward_rates(projection_curve, maturity_years):
        floating_leg_pv += notional * FLOAT_TENOR_YEARS * forward * curve_discount_factor(discount_curve, pay_time)

    par_swap_rate = floating_leg_pv / max(notional * annuity, 1e-12)
    receiver_fixed_pv = fixed_leg_pv - floating_leg_pv
    payer_fixed_pv = -receiver_fixed_pv
    pv01 = notional * annuity * 1e-4

    shocked_discount = parallel_shift_curve(discount_curve, PARALLEL_SHOCK_BP)
    shocked_projection = parallel_shift_curve(projection_curve, PARALLEL_SHOCK_BP)
    shocked_annuity = sum(FIXED_TENOR_YEARS * curve_discount_factor(shocked_discount, pay_time) for pay_time in fixed_schedule)
    shocked_fixed_leg_pv = notional * fixed_coupon * shocked_annuity
    shocked_floating_leg_pv = 0.0
    for pay_time, forward in _projection_forward_rates(shocked_projection, maturity_years):
        shocked_floating_leg_pv += notional * FLOAT_TENOR_YEARS * forward * curve_discount_factor(shocked_discount, pay_time)
    dv01 = (shocked_fixed_leg_pv - shocked_floating_leg_pv) - receiver_fixed_pv

    return SwapValuation(
        maturity_years=maturity_years,
        fixed_coupon=fixed_coupon,
        fixed_leg_pv=fixed_leg_pv,
        floating_leg_pv=floating_leg_pv,
        receiver_fixed_pv=receiver_fixed_pv,
        payer_fixed_pv=payer_fixed_pv,
        par_swap_rate=par_swap_rate,
        annuity=annuity,
        pv01=pv01,
        dv01=dv01,
    )
