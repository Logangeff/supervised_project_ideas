from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .config import BOOTSTRAP_GRID, HULL_WHITE_DEFAULT_A, HULL_WHITE_DEFAULT_SIGMA, MODEL_EVAL_GRID
from .nss import curve_frame_from_zero_yields


def _eval_target_curve(curve: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    eval_curve = curve[curve["maturity_years"].isin(MODEL_EVAL_GRID)].copy()
    maturities = eval_curve["maturity_years"].to_numpy(dtype=float)
    zero_yields = eval_curve["zero_yield"].to_numpy(dtype=float)
    if len(maturities) < 4:
        raise ValueError("Need enough curve points for short-rate model calibration.")
    return maturities, zero_yields


def vasicek_zero_yields(maturities: np.ndarray, kappa: float, theta: float, sigma: float, r0: float) -> np.ndarray:
    maturities = np.asarray(maturities, dtype=float)
    kappa = max(kappa, 1e-8)
    sigma = max(sigma, 1e-8)
    b_term = (1.0 - np.exp(-kappa * maturities)) / kappa
    a_term = np.exp(
        (theta - (sigma**2) / (2.0 * kappa**2)) * (b_term - maturities) - (sigma**2) * (b_term**2) / (4.0 * kappa)
    )
    prices = a_term * np.exp(-b_term * r0)
    prices = np.clip(prices, 1e-12, None)
    return -np.log(prices) / maturities


def cir_zero_yields(maturities: np.ndarray, kappa: float, theta: float, sigma: float, r0: float) -> np.ndarray:
    maturities = np.asarray(maturities, dtype=float)
    if min(kappa, theta, sigma, r0) <= 0.0:
        raise ValueError("CIR parameters must be strictly positive.")
    gamma = np.sqrt(kappa**2 + 2.0 * sigma**2)
    exp_term = np.exp(gamma * maturities) - 1.0
    denom = (gamma + kappa) * exp_term + 2.0 * gamma
    a_term = ((2.0 * gamma * np.exp((kappa + gamma) * maturities / 2.0)) / denom) ** (2.0 * kappa * theta / sigma**2)
    b_term = 2.0 * exp_term / denom
    prices = a_term * np.exp(-b_term * r0)
    prices = np.clip(prices, 1e-12, None)
    return -np.log(prices) / maturities


def fit_vasicek_curve(curve: pd.DataFrame) -> dict:
    maturities, targets = _eval_target_curve(curve)
    initial = np.array([0.30, float(targets[-1]), 0.02, float(targets[0])], dtype=float)
    lower = np.array([1e-4, 0.0, 1e-4, 0.0], dtype=float)
    upper = np.array([5.0, 0.20, 0.10, 0.20], dtype=float)

    def residuals(params: np.ndarray) -> np.ndarray:
        fitted = vasicek_zero_yields(maturities, *params)
        return fitted - targets

    result = least_squares(residuals, x0=initial, bounds=(lower, upper), max_nfev=10000)
    fitted = vasicek_zero_yields(maturities, *result.x)
    errors = fitted - targets
    return {
        "model": "Vasicek",
        "kappa": float(result.x[0]),
        "theta": float(result.x[1]),
        "sigma": float(result.x[2]),
        "r0": float(result.x[3]),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "mae": float(np.mean(np.abs(errors))),
        "success": bool(result.success),
        "nfev": int(result.nfev),
    }


def fit_cir_curve(curve: pd.DataFrame) -> dict:
    maturities, targets = _eval_target_curve(curve)
    initial = np.array([0.40, max(float(targets[-1]), 0.01), 0.04, max(float(targets[0]), 0.01)], dtype=float)
    lower = np.array([1e-4, 1e-4, 1e-4, 1e-6], dtype=float)
    upper = np.array([5.0, 0.20, 0.50, 0.20], dtype=float)

    def residuals(params: np.ndarray) -> np.ndarray:
        kappa, theta, sigma, r0 = params
        if 2.0 * kappa * theta <= sigma**2:
            return np.full_like(targets, 1e3)
        try:
            fitted = cir_zero_yields(maturities, kappa, theta, sigma, r0)
        except ValueError:
            return np.full_like(targets, 1e3)
        return fitted - targets

    result = least_squares(residuals, x0=initial, bounds=(lower, upper), max_nfev=15000)
    fitted = cir_zero_yields(maturities, *result.x)
    errors = fitted - targets
    return {
        "model": "CIR",
        "kappa": float(result.x[0]),
        "theta": float(result.x[1]),
        "sigma": float(result.x[2]),
        "r0": float(result.x[3]),
        "feller_margin": float(2.0 * result.x[0] * result.x[1] - result.x[2] ** 2),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "mae": float(np.mean(np.abs(errors))),
        "success": bool(result.success),
        "nfev": int(result.nfev),
    }


def estimate_ou_parameters(rate_series: pd.Series, dt_years: float = 1.0 / 12.0) -> dict:
    series = pd.to_numeric(rate_series, errors="coerce").dropna().astype(float)
    if len(series) < 12:
        return {
            "a": HULL_WHITE_DEFAULT_A,
            "sigma": HULL_WHITE_DEFAULT_SIGMA,
            "phi": math.exp(-HULL_WHITE_DEFAULT_A * dt_years),
            "used_default": True,
            "observations": int(len(series)),
        }
    x = series.iloc[:-1].to_numpy()
    y = series.iloc[1:].to_numpy()
    design = np.column_stack([np.ones_like(x), x])
    intercept, phi = np.linalg.lstsq(design, y, rcond=None)[0]
    phi = float(np.clip(phi, 1e-6, 0.9999))
    a_param = float(max(-math.log(phi) / dt_years, 1e-6))
    theta = float(intercept / max(1.0 - phi, 1e-8))
    residuals = y - (intercept + phi * x)
    resid_std = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else HULL_WHITE_DEFAULT_SIGMA
    sigma = float(resid_std * math.sqrt(max(2.0 * a_param / (1.0 - math.exp(-2.0 * a_param * dt_years)), 1e-8)))
    return {
        "a": a_param,
        "sigma": max(sigma, 1e-6),
        "theta_proxy": theta,
        "phi": phi,
        "used_default": False,
        "observations": int(len(series)),
    }


def build_vasicek_curve(params: dict, maturities: list[float] | None = None) -> pd.DataFrame:
    maturities = BOOTSTRAP_GRID if maturities is None else maturities
    zero = vasicek_zero_yields(np.array(maturities, dtype=float), params["kappa"], params["theta"], params["sigma"], params["r0"])
    return curve_frame_from_zero_yields(maturities, zero)


def build_cir_curve(params: dict, maturities: list[float] | None = None) -> pd.DataFrame:
    maturities = BOOTSTRAP_GRID if maturities is None else maturities
    zero = cir_zero_yields(np.array(maturities, dtype=float), params["kappa"], params["theta"], params["sigma"], params["r0"])
    return curve_frame_from_zero_yields(maturities, zero)


def build_hull_white_curve(benchmark_curve: pd.DataFrame) -> pd.DataFrame:
    return benchmark_curve[["maturity_years", "zero_yield", "discount_factor", "forward_rate"]].copy().reset_index(drop=True)
