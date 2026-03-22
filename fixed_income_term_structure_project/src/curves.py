from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import BOOTSTRAP_GRID, BOND_PRICING_MATURITIES, MODEL_EVAL_GRID, PAR_SWAP_MATURITIES, TREASURY_SERIES


SERIES_BY_MATURITY = {item["maturity_years"]: item["series_id"] for item in TREASURY_SERIES}


def _interp_linear(x_points: list[float], y_points: list[float], target: float) -> float:
    return float(np.interp(target, x_points, y_points))


def _discount_factor_at_maturity(curve: pd.DataFrame, maturity_years: float) -> float:
    maturities = curve["maturity_years"].to_numpy(dtype=float)
    log_discount = np.log(curve["discount_factor"].to_numpy(dtype=float))
    return float(np.exp(np.interp(maturity_years, maturities, log_discount)))


def _compute_par_yield_map(row: pd.Series) -> dict[float, float]:
    observed_map = {
        maturity: float(row[series_id]) / 100.0
        for maturity, series_id in SERIES_BY_MATURITY.items()
        if pd.notna(row[series_id])
    }
    if 1.0 not in observed_map:
        raise ValueError("Missing 1Y Treasury series required to anchor the bootstrap.")

    known_long_maturities = sorted(maturity for maturity in observed_map if maturity >= 1.0)
    known_long_yields = [observed_map[maturity] for maturity in known_long_maturities]
    par_map = {}
    for maturity in BOOTSTRAP_GRID:
        if maturity <= 1.0 and maturity in observed_map:
            par_map[maturity] = observed_map[maturity]
        elif maturity < 1.0:
            short_known = sorted(maturity_key for maturity_key in observed_map if maturity_key <= 1.0)
            short_yields = [observed_map[maturity_key] for maturity_key in short_known]
            par_map[maturity] = _interp_linear(short_known, short_yields, maturity)
        else:
            par_map[maturity] = _interp_linear(known_long_maturities, known_long_yields, maturity)
    return par_map


def bootstrap_curve_from_row(row: pd.Series) -> pd.DataFrame:
    par_map = _compute_par_yield_map(row)
    discount_map: dict[float, float] = {}
    zero_map: dict[float, float] = {}

    for maturity in BOOTSTRAP_GRID:
        par_yield = par_map[maturity]
        if maturity <= 1.0:
            discount_factor = 1.0 / (1.0 + par_yield * maturity)
        else:
            previous_dates = [date for date in BOOTSTRAP_GRID if 0 < date < maturity]
            coupon_dates = [date for date in previous_dates if abs((date * 2) - round(date * 2)) < 1e-8]
            coupon_sum = sum(discount_map[date] for date in coupon_dates)
            discount_factor = (1.0 - (par_yield / 2.0) * coupon_sum) / (1.0 + par_yield / 2.0)
        discount_factor = max(discount_factor, 1e-12)
        zero_yield = -math.log(discount_factor) / maturity
        discount_map[maturity] = discount_factor
        zero_map[maturity] = zero_yield

    records = []
    previous_maturity = None
    previous_discount = None
    for maturity in BOOTSTRAP_GRID:
        discount_factor = discount_map[maturity]
        zero_yield = zero_map[maturity]
        if previous_maturity is None:
            forward_rate = zero_yield
        else:
            delta = maturity - previous_maturity
            forward_rate = -math.log(discount_factor / previous_discount) / delta
        records.append(
            {
                "maturity_years": maturity,
                "par_yield": par_map[maturity],
                "zero_yield": zero_yield,
                "discount_factor": discount_factor,
                "forward_rate": forward_rate,
            }
        )
        previous_maturity = maturity
        previous_discount = discount_factor
    return pd.DataFrame(records)


def par_swap_rate_from_discount_curve(curve: pd.DataFrame, maturity_years: float) -> float:
    payment_dates = [date for date in BOOTSTRAP_GRID if 0 < date <= maturity_years]
    payment_dates = [date for date in payment_dates if abs(date * 2 - round(date * 2)) < 1e-8]
    if not payment_dates:
        raise ValueError("No valid payment dates for par swap rate computation.")
    curve_map = dict(zip(curve["maturity_years"], curve["discount_factor"]))
    denom = sum(0.5 * curve_map[date] for date in payment_dates)
    return (1.0 - curve_map[maturity_years]) / denom


def quote_yield_from_discount_curve(curve: pd.DataFrame, maturity_years: float) -> float:
    if maturity_years >= 1.0 and maturity_years in set(curve["maturity_years"].tolist()):
        return par_swap_rate_from_discount_curve(curve, maturity_years)
    discount_factor = _discount_factor_at_maturity(curve, maturity_years)
    return (1.0 / discount_factor - 1.0) / maturity_years


def price_coupon_bond_from_curve(curve: pd.DataFrame, maturity_years: float, coupon_rate: float, face: float = 100.0) -> float:
    payment_dates = [date for date in BOOTSTRAP_GRID if 0 < date <= maturity_years]
    payment_dates = [date for date in payment_dates if abs(date * 2 - round(date * 2)) < 1e-8]
    curve_map = dict(zip(curve["maturity_years"], curve["discount_factor"]))
    coupon_cash = face * coupon_rate / 2.0
    price = 0.0
    for payment_date in payment_dates[:-1]:
        price += coupon_cash * curve_map[payment_date]
    price += (coupon_cash + face) * curve_map[maturity_years]
    return price


def compute_curve_quality(curve: pd.DataFrame) -> dict[str, float]:
    discount = curve["discount_factor"].to_numpy()
    forward = curve["forward_rate"].to_numpy()
    monotonic = bool(np.all(np.diff(discount) <= 1e-10))
    positive = bool(np.all(discount > 0.0))
    roughness = float(np.mean(np.square(np.diff(forward)))) if len(forward) > 1 else 0.0
    return {
        "discount_positive": positive,
        "discount_monotonic": monotonic,
        "forward_roughness": roughness,
    }


def select_curve_grid(curve: pd.DataFrame, grid: list[float] | None = None) -> pd.DataFrame:
    grid = MODEL_EVAL_GRID if grid is None else grid
    return curve[curve["maturity_years"].isin(grid)].copy().reset_index(drop=True)


def build_pricing_records(snapshot_date: pd.Timestamp, benchmark_curve: pd.DataFrame, model_curve: pd.DataFrame, model_name: str) -> list[dict]:
    records: list[dict] = []
    for maturity in BOND_PRICING_MATURITIES:
        benchmark_swap = par_swap_rate_from_discount_curve(benchmark_curve, maturity)
        benchmark_bond_price = price_coupon_bond_from_curve(benchmark_curve, maturity, benchmark_swap)
        model_bond_price = price_coupon_bond_from_curve(model_curve, maturity, benchmark_swap)
        records.append(
            {
                "date": snapshot_date,
                "model": model_name,
                "instrument_type": "coupon_bond",
                "maturity_years": maturity,
                "reference_coupon_rate": benchmark_swap,
                "benchmark_price": benchmark_bond_price,
                "model_price": model_bond_price,
                "pricing_error": model_bond_price - benchmark_bond_price,
            }
        )
    return records


def build_swap_rate_records(snapshot_date: pd.Timestamp, benchmark_curve: pd.DataFrame, model_curve: pd.DataFrame, model_name: str) -> list[dict]:
    records: list[dict] = []
    for maturity in PAR_SWAP_MATURITIES:
        benchmark_rate = par_swap_rate_from_discount_curve(benchmark_curve, maturity)
        model_rate = par_swap_rate_from_discount_curve(model_curve, maturity)
        records.append(
            {
                "date": snapshot_date,
                "model": model_name,
                "maturity_years": maturity,
                "benchmark_par_swap_rate": benchmark_rate,
                "model_par_swap_rate": model_rate,
                "swap_rate_error": model_rate - benchmark_rate,
            }
        )
    return records
