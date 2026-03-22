from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .config import BOOTSTRAP_GRID, MODEL_EVAL_GRID


def nss_zero_yield(maturity: float, beta0: float, beta1: float, beta2: float, beta3: float, tau1: float, tau2: float) -> float:
    maturity = max(float(maturity), 1e-8)
    x1 = maturity / tau1
    x2 = maturity / tau2
    load1 = (1.0 - math.exp(-x1)) / x1
    load2 = load1 - math.exp(-x1)
    load3 = ((1.0 - math.exp(-x2)) / x2) - math.exp(-x2)
    return beta0 + beta1 * load1 + beta2 * load2 + beta3 * load3


def nss_zero_yield_vector(maturities: np.ndarray, params: np.ndarray) -> np.ndarray:
    return np.array([nss_zero_yield(float(maturity), *params) for maturity in maturities], dtype=float)


def curve_frame_from_zero_yields(maturities: list[float], zero_yields: np.ndarray) -> pd.DataFrame:
    records: list[dict] = []
    previous_maturity = None
    previous_discount = None
    for maturity, zero_yield in zip(maturities, zero_yields, strict=True):
        discount_factor = float(math.exp(-zero_yield * maturity))
        if previous_maturity is None:
            forward_rate = float(zero_yield)
        else:
            delta = maturity - previous_maturity
            forward_rate = -math.log(discount_factor / previous_discount) / delta
        records.append(
            {
                "maturity_years": float(maturity),
                "zero_yield": float(zero_yield),
                "discount_factor": discount_factor,
                "forward_rate": float(forward_rate),
            }
        )
        previous_maturity = maturity
        previous_discount = discount_factor
    return pd.DataFrame(records)


def fit_nss_parameters(curve: pd.DataFrame) -> dict:
    eval_curve = curve[curve["maturity_years"].isin(MODEL_EVAL_GRID)].copy()
    maturities = eval_curve["maturity_years"].to_numpy(dtype=float)
    zero_yields = eval_curve["zero_yield"].to_numpy(dtype=float)
    if len(maturities) < 4:
        raise ValueError("Need at least four maturity points to fit the NSS curve.")

    initial = np.array(
        [
            float(zero_yields[-1]),
            float(zero_yields[0] - zero_yields[-1]),
            -0.01,
            0.01,
            1.5,
            6.0,
        ],
        dtype=float,
    )
    lower = np.array([-0.05, -0.20, -0.20, -0.20, 0.05, 0.10], dtype=float)
    upper = np.array([0.20, 0.20, 0.20, 0.20, 10.0, 20.0], dtype=float)

    def residuals(params: np.ndarray) -> np.ndarray:
        fitted = nss_zero_yield_vector(maturities, params)
        return fitted - zero_yields

    result = least_squares(residuals, x0=initial, bounds=(lower, upper), max_nfev=10000)
    fitted = nss_zero_yield_vector(maturities, result.x)
    errors = fitted - zero_yields
    return {
        "beta0": float(result.x[0]),
        "beta1": float(result.x[1]),
        "beta2": float(result.x[2]),
        "beta3": float(result.x[3]),
        "tau1": float(result.x[4]),
        "tau2": float(result.x[5]),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "mae": float(np.mean(np.abs(errors))),
        "success": bool(result.success),
        "nfev": int(result.nfev),
        "cost": float(result.cost),
    }


def build_nss_curve(params: dict, maturities: list[float] | None = None) -> pd.DataFrame:
    maturities = BOOTSTRAP_GRID if maturities is None else maturities
    ordered = [float(maturity) for maturity in maturities]
    param_vector = np.array(
        [params["beta0"], params["beta1"], params["beta2"], params["beta3"], params["tau1"], params["tau2"]],
        dtype=float,
    )
    zero_yields = nss_zero_yield_vector(np.array(ordered, dtype=float), param_vector)
    return curve_frame_from_zero_yields(ordered, zero_yields)
