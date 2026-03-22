from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import PROCESSED_DIR
from .phase2_config import (
    PARALLEL_SHOCK_BP,
    PHASE2_DEFAULT_FIXED_COUPON,
    PHASE2_DEFAULT_START_DATE,
    PHASE2_FIGURES_DIR,
    PHASE2_METRICS_DIR,
    PHASE2_NOTIONAL,
    PHASE2_PROCESSED_DIR,
    PHASE2_RAW_DIR,
    PHASE2_SMOKE_START_DATE,
    PHASE2_SUMMARIES_DIR,
    PHASE2_SWAP_MATURITIES,
    SOFR_SERIES_ID,
)
from .phase2_curves import build_discount_proxy_curve, build_projection_curve, parallel_shift_curve, slope_shift_curve
from .phase2_pricing import price_swap
from .rates import fetch_fred_series
from .reporting import write_csv, write_json


PHASE1_MONTHLY_RATES_PATH = PROCESSED_DIR / "monthly_rates.csv"
PHASE1_NSS_CURVES_PATH = PROCESSED_DIR / "nss_curves.csv"
PHASE1_BOOTSTRAP_CURVES_PATH = PROCESSED_DIR / "bootstrap_curves.csv"

SOFR_DAILY_PATH = PHASE2_RAW_DIR / "sofr_daily.csv"
DISCOUNT_CURVE_PATH = PHASE2_PROCESSED_DIR / "discount_proxy_curves.csv"
PROJECTION_CURVE_PATH = PHASE2_PROCESSED_DIR / "projection_curves.csv"
SWAP_PRICING_DETAILS_PATH = PHASE2_PROCESSED_DIR / "swap_pricing_details.csv"


def _load_phase1_artifacts(start_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not PHASE1_MONTHLY_RATES_PATH.exists() or not PHASE1_NSS_CURVES_PATH.exists():
        raise FileNotFoundError("Phase 1 artifacts are missing. Run the Phase 1 pipeline first.")
    monthly_rates = pd.read_csv(PHASE1_MONTHLY_RATES_PATH, parse_dates=["date"])
    nss_curves = pd.read_csv(PHASE1_NSS_CURVES_PATH, parse_dates=["date"])
    monthly_rates = monthly_rates[monthly_rates["date"] >= pd.Timestamp(start_date)].reset_index(drop=True)
    nss_curves = nss_curves[nss_curves["date"] >= pd.Timestamp(start_date)].reset_index(drop=True)
    return monthly_rates, nss_curves


def _load_phase1_bootstrap_curves(start_date: str) -> pd.DataFrame:
    if not PHASE1_BOOTSTRAP_CURVES_PATH.exists():
        raise FileNotFoundError("Phase 1 bootstrap curves are missing. Run the Phase 1 pipeline first.")
    curves = pd.read_csv(PHASE1_BOOTSTRAP_CURVES_PATH, parse_dates=["date"])
    return curves[curves["date"] >= pd.Timestamp(start_date)].reset_index(drop=True)


def phase_fetch_public_data(start_date: str, write_outputs: bool = True) -> pd.DataFrame:
    sofr = fetch_fred_series(SOFR_SERIES_ID)
    sofr.columns = ["date", "SOFR"]
    sofr["sofr_rate"] = pd.to_numeric(sofr["SOFR"], errors="coerce") / 100.0
    sofr = sofr[["date", "sofr_rate"]].dropna().reset_index(drop=True)
    monthly_rates, _ = _load_phase1_artifacts(start_date)
    common_dates = monthly_rates["date"][monthly_rates["date"] >= sofr["date"].min() + pd.DateOffset(months=12)].copy()
    common_dates = common_dates.reset_index(drop=True)
    summary = {
        "sofr_series_id": SOFR_SERIES_ID,
        "sofr_start_date": str(sofr["date"].min().date()),
        "sofr_end_date": str(sofr["date"].max().date()),
        "sofr_rows": int(len(sofr)),
        "phase2_sample_start": str(common_dates.min().date()) if not common_dates.empty else None,
        "phase2_sample_end": str(common_dates.max().date()) if not common_dates.empty else None,
        "phase2_snapshot_count": int(len(common_dates)),
        "note": "Phase 2 starts once there is a 12-month SOFR lookback window.",
    }
    if write_outputs:
        write_csv(SOFR_DAILY_PATH, sofr)
        write_json(PHASE2_SUMMARIES_DIR / "phase2_data_summary.json", summary)
    return sofr


def phase_build_discount_proxy_curve(start_date: str, write_outputs: bool = True) -> pd.DataFrame:
    sofr = phase_fetch_public_data(start_date, write_outputs=write_outputs)
    monthly_rates, nss_curves = _load_phase1_artifacts(start_date)
    valid_dates = monthly_rates["date"][monthly_rates["date"] >= sofr["date"].min() + pd.DateOffset(months=12)].tolist()

    curve_frames: list[pd.DataFrame] = []
    meta_records: list[dict] = []
    for snapshot_date in valid_dates:
        nss_curve = nss_curves[nss_curves["date"] == snapshot_date].copy()
        if nss_curve.empty:
            continue
        curve, meta = build_discount_proxy_curve(snapshot_date, sofr, nss_curve)
        curve_frames.append(curve)
        meta_records.append(meta)

    discount_curves = pd.concat(curve_frames, ignore_index=True)
    meta_frame = pd.DataFrame(meta_records)
    summary = {
        "snapshot_count": int(meta_frame["snapshot_date"].nunique()),
        "mean_join_shift_bps": float(meta_frame["join_shift"].mean() * 10000.0),
        "max_abs_join_shift_bps": float(meta_frame["join_shift"].abs().max() * 10000.0),
        "discount_positive_share": float(meta_frame["discount_positive"].mean()),
        "discount_monotonic_share": float(meta_frame["discount_monotonic"].mean()),
    }
    if write_outputs:
        write_csv(DISCOUNT_CURVE_PATH, discount_curves)
        write_json(PHASE2_SUMMARIES_DIR / "phase2_curve_summary.json", summary)
    return discount_curves


def phase_build_projection_curve(start_date: str, write_outputs: bool = True) -> pd.DataFrame:
    bootstrap_curves = _load_phase1_bootstrap_curves(start_date)
    sofr = phase_fetch_public_data(start_date, write_outputs=write_outputs)
    min_valid_date = sofr["date"].min() + pd.DateOffset(months=12)
    valid_dates = [date_value for date_value in sorted(set(pd.to_datetime(bootstrap_curves["date"]))) if date_value >= min_valid_date and date_value >= pd.Timestamp(start_date)]

    curve_frames: list[pd.DataFrame] = []
    for snapshot_date in valid_dates:
        bootstrap_curve = bootstrap_curves[bootstrap_curves["date"] == snapshot_date].copy()
        if bootstrap_curve.empty:
            continue
        curve_frames.append(build_projection_curve(snapshot_date, bootstrap_curve))

    projection_curves = pd.concat(curve_frames, ignore_index=True)
    if write_outputs:
        write_csv(PROJECTION_CURVE_PATH, projection_curves)
    return projection_curves


def _plot_phase2_figures(discount_curves: pd.DataFrame, projection_curves: pd.DataFrame, gap_table: pd.DataFrame, sensitivity: pd.DataFrame) -> None:
    PHASE2_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    latest_date = pd.to_datetime(discount_curves["date"]).max()
    discount_latest = discount_curves[pd.to_datetime(discount_curves["date"]) == latest_date].copy()
    projection_latest = projection_curves[pd.to_datetime(projection_curves["date"]) == latest_date].copy()

    plt.figure(figsize=(10, 6))
    plt.plot(discount_latest["maturity_years"], discount_latest["zero_yield"] * 100.0, label="Discount proxy", color="#0f766e", linewidth=2.2)
    plt.plot(projection_latest["maturity_years"], projection_latest["zero_yield"] * 100.0, label="Projection curve", color="#2563eb", linewidth=2.2)
    plt.title(f"Discount vs Projection Curves on {latest_date.date()}")
    plt.xlabel("Maturity (years)")
    plt.ylabel("Zero yield (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PHASE2_FIGURES_DIR / f"discount_vs_projection_{latest_date:%Y%m%d}.png", dpi=150)
    plt.close()

    for maturity in PHASE2_SWAP_MATURITIES:
        subset = gap_table[gap_table["maturity_years"] == maturity].copy()
        plt.figure(figsize=(10, 5))
        plt.plot(pd.to_datetime(subset["date"]), subset["total_gap_bps"], label="Total gap", color="#111827")
        plt.plot(pd.to_datetime(subset["date"]), subset["discount_effect_bps"], label="Discount effect", color="#0f766e")
        plt.plot(pd.to_datetime(subset["date"]), subset["projection_effect_bps"], label="Projection effect", color="#dc2626")
        plt.title(f"Single-Curve vs Multi-Curve Par Rate Gap: {int(maturity)}Y")
        plt.xlabel("Date")
        plt.ylabel("Gap (bps)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PHASE2_FIGURES_DIR / f"swap_rate_gap_{int(maturity)}y.png", dpi=150)
        plt.close()

        latest_sens = sensitivity[(sensitivity["maturity_years"] == maturity) & (pd.to_datetime(sensitivity["date"]) == pd.to_datetime(sensitivity["date"]).max())].copy()
        plt.figure(figsize=(10, 5))
        plt.bar(latest_sens["scenario"], latest_sens["swap_pv_change"], color="#0f4c81")
        plt.title(f"Latest Sensitivity Waterfall: {int(maturity)}Y")
        plt.ylabel("Swap PV change")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(PHASE2_FIGURES_DIR / f"sensitivity_waterfall_{int(maturity)}y_{latest_date:%Y%m%d}.png", dpi=150)
        plt.close()


def phase_price_multi_curve_swaps(start_date: str, write_outputs: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    discount_curves = phase_build_discount_proxy_curve(start_date, write_outputs=write_outputs)
    projection_curves = phase_build_projection_curve(start_date, write_outputs=write_outputs)
    _, nss_curves = _load_phase1_artifacts(start_date)

    pricing_records: list[dict] = []
    gap_records: list[dict] = []
    pv_records: list[dict] = []

    for snapshot_date in sorted(set(pd.to_datetime(discount_curves["date"]))):
        discount_curve = discount_curves[pd.to_datetime(discount_curves["date"]) == snapshot_date].copy()
        projection_curve = projection_curves[pd.to_datetime(projection_curves["date"]) == snapshot_date].copy()
        baseline_curve = nss_curves[pd.to_datetime(nss_curves["date"]) == snapshot_date].copy()
        if discount_curve.empty or projection_curve.empty or baseline_curve.empty:
            continue

        for maturity in PHASE2_SWAP_MATURITIES:
            baseline = price_swap(baseline_curve, baseline_curve, maturity, PHASE2_DEFAULT_FIXED_COUPON, PHASE2_NOTIONAL)
            discount_only = price_swap(discount_curve, baseline_curve, maturity, PHASE2_DEFAULT_FIXED_COUPON, PHASE2_NOTIONAL)
            multicurve = price_swap(discount_curve, projection_curve, maturity, PHASE2_DEFAULT_FIXED_COUPON, PHASE2_NOTIONAL)

            for setup_name, valuation in [
                ("phase1_single_curve", baseline),
                ("discount_only", discount_only),
                ("phase2_multicurve", multicurve),
            ]:
                pricing_records.append(
                    {
                        "date": snapshot_date,
                        "setup": setup_name,
                        "maturity_years": maturity,
                        "fixed_coupon": valuation.fixed_coupon,
                        "fixed_leg_pv": valuation.fixed_leg_pv,
                        "floating_leg_pv": valuation.floating_leg_pv,
                        "receiver_fixed_pv": valuation.receiver_fixed_pv,
                        "payer_fixed_pv": valuation.payer_fixed_pv,
                        "par_swap_rate": valuation.par_swap_rate,
                        "annuity": valuation.annuity,
                        "pv01": valuation.pv01,
                        "dv01": valuation.dv01,
                    }
                )

            baseline_par = price_swap(baseline_curve, baseline_curve, maturity, 0.0, PHASE2_NOTIONAL).par_swap_rate
            discount_only_par = price_swap(discount_curve, baseline_curve, maturity, 0.0, PHASE2_NOTIONAL).par_swap_rate
            multicurve_par = price_swap(discount_curve, projection_curve, maturity, 0.0, PHASE2_NOTIONAL).par_swap_rate
            gap_records.append(
                {
                    "date": snapshot_date,
                    "maturity_years": maturity,
                    "phase1_single_curve_par_rate": baseline_par,
                    "discount_only_par_rate": discount_only_par,
                    "phase2_multicurve_par_rate": multicurve_par,
                    "discount_effect_bps": (discount_only_par - baseline_par) * 10000.0,
                    "projection_effect_bps": (multicurve_par - discount_only_par) * 10000.0,
                    "total_gap_bps": (multicurve_par - baseline_par) * 10000.0,
                }
            )

            baseline_pv = price_swap(baseline_curve, baseline_curve, maturity, baseline_par, PHASE2_NOTIONAL).receiver_fixed_pv
            discount_only_pv = price_swap(discount_curve, baseline_curve, maturity, baseline_par, PHASE2_NOTIONAL).receiver_fixed_pv
            multicurve_pv = price_swap(discount_curve, projection_curve, maturity, baseline_par, PHASE2_NOTIONAL).receiver_fixed_pv
            pv_records.append(
                {
                    "date": snapshot_date,
                    "maturity_years": maturity,
                    "comparison_fixed_coupon": baseline_par,
                    "phase1_single_curve_pv": baseline_pv,
                    "discount_only_pv": discount_only_pv,
                    "phase2_multicurve_pv": multicurve_pv,
                    "discount_effect_pv": discount_only_pv - baseline_pv,
                    "projection_effect_pv": multicurve_pv - discount_only_pv,
                    "total_gap_pv": multicurve_pv - baseline_pv,
                }
            )

    pricing_details = pd.DataFrame(pricing_records)
    gap_table = pd.DataFrame(gap_records)
    pv_table = pd.DataFrame(pv_records)
    if write_outputs:
        write_csv(SWAP_PRICING_DETAILS_PATH, pricing_details)
        write_csv(PHASE2_METRICS_DIR / "baseline_vs_multicurve_swap_rates.csv", gap_table)
        write_csv(PHASE2_METRICS_DIR / "baseline_vs_multicurve_pv.csv", pv_table)
    return pricing_details, gap_table, pv_table


def phase_run_sensitivity_grid(start_date: str, write_outputs: bool = True) -> pd.DataFrame:
    discount_curves = phase_build_discount_proxy_curve(start_date, write_outputs=write_outputs)
    projection_curves = phase_build_projection_curve(start_date, write_outputs=write_outputs)
    sensitivity_records: list[dict] = []

    for snapshot_date in sorted(set(pd.to_datetime(discount_curves["date"]))):
        discount_curve = discount_curves[pd.to_datetime(discount_curves["date"]) == snapshot_date].copy()
        projection_curve = projection_curves[pd.to_datetime(projection_curves["date"]) == snapshot_date].copy()
        if discount_curve.empty or projection_curve.empty:
            continue
        for maturity in PHASE2_SWAP_MATURITIES:
            base_coupon = price_swap(discount_curve, projection_curve, maturity, 0.0, PHASE2_NOTIONAL).par_swap_rate
            base = price_swap(discount_curve, projection_curve, maturity, base_coupon, PHASE2_NOTIONAL)
            scenarios = {
                "discount_up_1bp": (parallel_shift_curve(discount_curve, PARALLEL_SHOCK_BP), projection_curve),
                "discount_down_1bp": (parallel_shift_curve(discount_curve, -PARALLEL_SHOCK_BP), projection_curve),
                "projection_up_1bp": (discount_curve, parallel_shift_curve(projection_curve, PARALLEL_SHOCK_BP)),
                "projection_down_1bp": (discount_curve, parallel_shift_curve(projection_curve, -PARALLEL_SHOCK_BP)),
                "projection_steepener": (discount_curve, slope_shift_curve(projection_curve, "steepener")),
                "projection_flattener": (discount_curve, slope_shift_curve(projection_curve, "flattener")),
            }
            for scenario_name, (shock_discount, shock_projection) in scenarios.items():
                shocked = price_swap(shock_discount, shock_projection, maturity, base_coupon, PHASE2_NOTIONAL)
                sensitivity_records.append(
                    {
                        "date": snapshot_date,
                        "maturity_years": maturity,
                        "scenario": scenario_name,
                        "fixed_coupon": base_coupon,
                        "base_par_rate": base.par_swap_rate,
                        "shocked_par_rate": shocked.par_swap_rate,
                        "par_rate_change_bps": (shocked.par_swap_rate - base.par_swap_rate) * 10000.0,
                        "base_fixed_leg_pv": base.fixed_leg_pv,
                        "base_floating_leg_pv": base.floating_leg_pv,
                        "base_swap_pv": base.receiver_fixed_pv,
                        "shocked_fixed_leg_pv": shocked.fixed_leg_pv,
                        "shocked_floating_leg_pv": shocked.floating_leg_pv,
                        "shocked_swap_pv": shocked.receiver_fixed_pv,
                        "fixed_leg_change": shocked.fixed_leg_pv - base.fixed_leg_pv,
                        "floating_leg_change": shocked.floating_leg_pv - base.floating_leg_pv,
                        "swap_pv_change": shocked.receiver_fixed_pv - base.receiver_fixed_pv,
                    }
                )

    sensitivity = pd.DataFrame(sensitivity_records)
    if write_outputs:
        write_csv(PHASE2_METRICS_DIR / "sensitivity_summary.csv", sensitivity)
    return sensitivity


def phase_results(start_date: str, write_outputs: bool = True) -> dict:
    PHASE2_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    PHASE2_RAW_DIR.mkdir(parents=True, exist_ok=True)
    PHASE2_SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
    PHASE2_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PHASE2_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    phase_fetch_public_data(start_date, write_outputs=write_outputs)
    discount_curves = phase_build_discount_proxy_curve(start_date, write_outputs=write_outputs)
    projection_curves = phase_build_projection_curve(start_date, write_outputs=write_outputs)
    _, gap_table, _ = phase_price_multi_curve_swaps(start_date, write_outputs=write_outputs)
    sensitivity = phase_run_sensitivity_grid(start_date, write_outputs=write_outputs)
    _plot_phase2_figures(discount_curves, projection_curves, gap_table, sensitivity)
    outputs = {
        "discount_proxy_curves": str(DISCOUNT_CURVE_PATH),
        "projection_curves": str(PROJECTION_CURVE_PATH),
        "swap_pricing_details": str(SWAP_PRICING_DETAILS_PATH),
        "phase2_metrics_dir": str(PHASE2_METRICS_DIR),
        "phase2_figures_dir": str(PHASE2_FIGURES_DIR),
        "phase2_summaries_dir": str(PHASE2_SUMMARIES_DIR),
    }
    print(json.dumps(outputs, indent=2))
    return outputs


def run_phase(phase: str) -> None:
    phase = phase.lower()
    start_date = PHASE2_SMOKE_START_DATE if phase == "phase2_smoke" else PHASE2_DEFAULT_START_DATE
    if phase == "fetch_phase2_public_data":
        phase_fetch_public_data(start_date)
    elif phase == "build_discount_proxy_curve":
        phase_build_discount_proxy_curve(start_date)
    elif phase == "build_projection_curve":
        phase_build_projection_curve(start_date)
    elif phase == "price_multi_curve_swaps":
        phase_price_multi_curve_swaps(start_date)
    elif phase == "run_sensitivity_grid":
        phase_run_sensitivity_grid(start_date)
    elif phase == "phase2_results":
        phase_results(start_date)
    elif phase == "phase2_smoke":
        phase_results(start_date)
    elif phase == "phase2_all":
        phase_results(PHASE2_DEFAULT_START_DATE)
    else:
        raise ValueError(f"Unknown phase: {phase}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 multi-curve swap pricing pipeline.")
    parser.add_argument("--phase", default="phase2_all", help="Phase 2 pipeline phase to run.")
    args = parser.parse_args()
    run_phase(args.phase)


if __name__ == "__main__":
    main()
