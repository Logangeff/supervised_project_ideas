from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .phase2_config import FLOAT_TENOR_YEARS, PHASE2_GRID, PROJECTION_GRID, SLOPE_PIVOT_YEARS, SOFR_SHORT_END_MONTHS, SLOPE_SHOCK_MAX_BP


def curve_discount_factor(curve: pd.DataFrame, maturity_years: float) -> float:
    ordered = curve.sort_values("maturity_years").reset_index(drop=True)
    maturities = ordered["maturity_years"].to_numpy(dtype=float)
    log_discount = np.log(ordered["discount_factor"].to_numpy(dtype=float))
    return float(np.exp(np.interp(maturity_years, maturities, log_discount)))


def curve_zero_yield(curve: pd.DataFrame, maturity_years: float) -> float:
    maturity_years = max(float(maturity_years), 1e-8)
    discount_factor = curve_discount_factor(curve, maturity_years)
    return float(-math.log(discount_factor) / maturity_years)


def _interpolate_log_discount(anchor_map: dict[float, float], maturity_years: float) -> float:
    points = sorted(anchor_map)
    values = [math.log(anchor_map[point]) for point in points]
    return float(np.exp(np.interp(maturity_years, points, values)))


def compounded_sofr_window(sofr_daily: pd.DataFrame, end_date: pd.Timestamp, months: int) -> dict | None:
    end_date = pd.Timestamp(end_date).normalize()
    start_date = (end_date - pd.DateOffset(months=months)).normalize()
    accrual_end = end_date + pd.Timedelta(days=1)

    frame = sofr_daily.copy()
    frame = frame[pd.to_datetime(frame["date"]) <= end_date].copy()
    if frame.empty:
        return None

    before_start = frame[pd.to_datetime(frame["date"]) <= start_date].tail(1)
    after_start = frame[pd.to_datetime(frame["date"]) > start_date].copy()
    effective = pd.concat([before_start, after_start], ignore_index=True).drop_duplicates(subset=["date"]).sort_values("date")
    if effective.empty or pd.to_datetime(effective["date"]).min() > start_date:
        return None

    dates = pd.to_datetime(effective["date"]).tolist()
    rates = pd.to_numeric(effective["sofr_rate"], errors="coerce").tolist()
    factor = 1.0
    total_days = (accrual_end - start_date).days
    if total_days <= 0:
        return None

    for index, current_date in enumerate(dates):
        next_date = dates[index + 1] if index + 1 < len(dates) else accrual_end
        interval_start = max(current_date, start_date)
        interval_end = min(next_date, accrual_end)
        days = (interval_end - interval_start).days
        if days > 0:
            factor *= 1.0 + float(rates[index]) * days / 360.0

    annualized_rate = (factor - 1.0) * 360.0 / total_days
    return {
        "months": months,
        "start_date": str(start_date.date()),
        "end_date": str(end_date.date()),
        "factor": float(factor),
        "annualized_rate": float(annualized_rate),
        "discount_factor": float(1.0 / factor),
        "actual_days": int(total_days),
    }


def build_discount_proxy_curve(snapshot_date: pd.Timestamp, sofr_daily: pd.DataFrame, nss_curve: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    anchors: dict[float, float] = {}
    anchor_metadata: list[dict] = []
    month_to_tenor = {1: round(1.0 / 12.0, 6), 3: 0.25, 6: 0.5, 12: 1.0}

    for months in SOFR_SHORT_END_MONTHS:
        compounded = compounded_sofr_window(sofr_daily, snapshot_date, months)
        if compounded is None:
            raise ValueError(f"Insufficient SOFR history to build the {months}M node on {snapshot_date.date()}.")
        tenor = month_to_tenor[months]
        anchors[tenor] = compounded["discount_factor"]
        anchor_metadata.append({"tenor_years": tenor, **compounded})

    one_year_discount = anchors[1.0]
    one_year_zero = -math.log(one_year_discount)
    nss_one_year_zero = curve_zero_yield(nss_curve, 1.0)
    shift = one_year_zero - nss_one_year_zero

    records: list[dict] = []
    for maturity in PHASE2_GRID:
        if maturity in anchors:
            discount_factor = anchors[maturity]
            source_segment = "sofr_anchor"
        elif maturity < 1.0:
            discount_factor = _interpolate_log_discount(anchors, maturity)
            source_segment = "interpolated_short_end"
        else:
            base_zero = curve_zero_yield(nss_curve, maturity)
            shifted_zero = base_zero + shift
            discount_factor = math.exp(-shifted_zero * maturity)
            source_segment = "shifted_nss_long_end"
        records.append(
            {
                "date": snapshot_date,
                "curve_name": "public_discount_proxy",
                "maturity_years": float(maturity),
                "discount_factor": float(discount_factor),
                "source_segment": source_segment,
            }
        )

    curve = pd.DataFrame(records).sort_values("maturity_years").reset_index(drop=True)
    curve["discount_factor"] = np.minimum.accumulate(curve["discount_factor"].to_numpy(dtype=float))
    curve["discount_factor"] = np.clip(curve["discount_factor"], 1e-12, None)
    curve["zero_yield"] = curve.apply(lambda row: -math.log(row["discount_factor"]) / row["maturity_years"], axis=1)

    forward_rates = []
    prev_maturity = None
    prev_discount = None
    for maturity, discount_factor, zero_yield in zip(curve["maturity_years"], curve["discount_factor"], curve["zero_yield"], strict=True):
        if prev_maturity is None:
            forward_rates.append(float(zero_yield))
        else:
            delta = float(maturity - prev_maturity)
            forward_rates.append(float(-math.log(discount_factor / prev_discount) / delta))
        prev_maturity = float(maturity)
        prev_discount = float(discount_factor)
    curve["forward_rate"] = forward_rates

    metadata = {
        "snapshot_date": str(pd.Timestamp(snapshot_date).date()),
        "join_shift": float(shift),
        "one_year_sofr_zero": float(one_year_zero),
        "one_year_nss_zero_before_shift": float(nss_one_year_zero),
        "anchor_metadata": anchor_metadata,
        "discount_positive": bool((curve["discount_factor"] > 0.0).all()),
        "discount_monotonic": bool((curve["discount_factor"].diff().fillna(0.0) <= 1e-10).all()),
    }
    return curve, metadata


def build_projection_curve(snapshot_date: pd.Timestamp, nss_curve: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    prev_maturity = 0.0
    prev_discount = 1.0
    for maturity in PROJECTION_GRID:
        discount_factor = curve_discount_factor(nss_curve, maturity)
        zero_yield = -math.log(discount_factor) / maturity
        delta = maturity - prev_maturity
        forward_rate_3m = (prev_discount / discount_factor - 1.0) / delta
        records.append(
            {
                "date": snapshot_date,
                "curve_name": "projection_3m_treasury_proxy",
                "maturity_years": float(maturity),
                "discount_factor": float(discount_factor),
                "zero_yield": float(zero_yield),
                "forward_rate_3m": float(forward_rate_3m),
            }
        )
        prev_maturity = float(maturity)
        prev_discount = float(discount_factor)
    return pd.DataFrame(records)


def quarter_schedule(maturity_years: float, step: float) -> list[float]:
    periods = int(round(maturity_years / step))
    return [round(step * idx, 6) for idx in range(1, periods + 1)]


def parallel_shift_curve(curve: pd.DataFrame, shift_bp: float) -> pd.DataFrame:
    shifted = curve.copy().sort_values("maturity_years").reset_index(drop=True)
    shift = shift_bp / 10000.0
    shifted["zero_yield"] = shifted["zero_yield"] + shift
    shifted["discount_factor"] = np.exp(-shifted["zero_yield"] * shifted["maturity_years"])
    prev_discount = 1.0
    prev_maturity = 0.0
    forward_rates = []
    for maturity, discount_factor in zip(shifted["maturity_years"], shifted["discount_factor"], strict=True):
        delta = maturity - prev_maturity
        forward_rates.append((prev_discount / discount_factor - 1.0) / delta)
        prev_discount = discount_factor
        prev_maturity = maturity
    if "forward_rate_3m" in shifted.columns:
        shifted["forward_rate_3m"] = forward_rates
    else:
        shifted["forward_rate"] = forward_rates
    return shifted


def slope_shift_curve(curve: pd.DataFrame, mode: str, max_bp: float = SLOPE_SHOCK_MAX_BP, pivot_years: float = SLOPE_PIVOT_YEARS) -> pd.DataFrame:
    if mode not in {"steepener", "flattener"}:
        raise ValueError("mode must be 'steepener' or 'flattener'.")
    shifted = curve.copy().sort_values("maturity_years").reset_index(drop=True)
    max_maturity = float(shifted["maturity_years"].max())

    def shift_for_maturity(maturity: float) -> float:
        maturity = float(maturity)
        if maturity <= pivot_years:
            value = -max_bp * (pivot_years - maturity) / max(pivot_years - FLOAT_TENOR_YEARS, 1e-8)
        else:
            value = max_bp * (maturity - pivot_years) / max(max_maturity - pivot_years, 1e-8)
        return value if mode == "steepener" else -value

    shifts_bp = shifted["maturity_years"].map(shift_for_maturity)
    shifted["zero_yield"] = shifted["zero_yield"] + shifts_bp / 10000.0
    shifted["discount_factor"] = np.exp(-shifted["zero_yield"] * shifted["maturity_years"])
    prev_discount = 1.0
    prev_maturity = 0.0
    forward_rates = []
    for maturity, discount_factor in zip(shifted["maturity_years"], shifted["discount_factor"], strict=True):
        delta = maturity - prev_maturity
        forward_rates.append((prev_discount / discount_factor - 1.0) / delta)
        prev_discount = discount_factor
        prev_maturity = maturity
    shifted["forward_rate_3m"] = forward_rates
    shifted["scenario_curve_shift_bp"] = shifts_bp
    return shifted
