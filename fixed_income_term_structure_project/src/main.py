from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import (
    BOOTSTRAP_GRID,
    DEFAULT_START_DATE,
    FIGURES_DIR,
    METRICS_DIR,
    MODEL_EVAL_GRID,
    PROCESSED_DIR,
    RAW_DIR,
    SMOKE_START_DATE,
    SUMMARIES_DIR,
    TREASURY_SERIES,
)
from .curves import (
    bootstrap_curve_from_row,
    build_pricing_records,
    build_swap_rate_records,
    compute_curve_quality,
    par_swap_rate_from_discount_curve,
    quote_yield_from_discount_curve,
)
from .models import build_cir_curve, build_hull_white_curve, build_vasicek_curve, estimate_ou_parameters, fit_cir_curve, fit_vasicek_curve
from .nss import build_nss_curve, fit_nss_parameters
from .rates import build_monthly_snapshots, fetch_public_rates
from .reporting import write_csv, write_json


RAW_RATES_PATH = RAW_DIR / "public_rates_daily.csv"
MONTHLY_RATES_PATH = PROCESSED_DIR / "monthly_rates.csv"
BOOTSTRAP_CURVES_PATH = PROCESSED_DIR / "bootstrap_curves.csv"
NSS_PARAMETERS_PATH = PROCESSED_DIR / "nss_parameters.csv"
NSS_CURVES_PATH = PROCESSED_DIR / "nss_curves.csv"
MODEL_PARAMETERS_PATH = PROCESSED_DIR / "model_parameters.csv"
MODEL_CURVES_PATH = PROCESSED_DIR / "model_curves.csv"
PRICING_DETAILS_PATH = PROCESSED_DIR / "pricing_details.csv"
SWAP_DETAILS_PATH = PROCESSED_DIR / "swap_rate_details.csv"
OBSERVED_CURVE_DETAILS_PATH = PROCESSED_DIR / "observed_curve_comparison.csv"


def _load_frame(path: Path, parse_dates: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, parse_dates=parse_dates)


def _bootstrap_toy_check() -> dict:
    row = pd.Series(
        {
            "DGS1MO": 4.0,
            "DGS3MO": 4.1,
            "DGS6MO": 4.2,
            "DGS1": 4.25,
            "DGS2": 4.50,
            "DGS3": 4.60,
            "DGS5": 4.75,
            "DGS7": 4.80,
            "DGS10": 4.90,
            "DGS20": 5.00,
            "DGS30": 5.05,
        }
    )
    curve = bootstrap_curve_from_row(row)
    grid_map = dict(zip(curve["maturity_years"], curve["discount_factor"]))
    par_map = dict(zip(curve["maturity_years"], curve["par_yield"]))
    coupon = par_map[2.0] / 2.0
    lhs = coupon * (grid_map[0.5] + grid_map[1.0] + grid_map[1.5]) + (1.0 + coupon) * grid_map[2.0]
    return {
        "bootstrap_equation_residual_2y": float(lhs - 1.0),
        "discount_positive": bool((curve["discount_factor"] > 0.0).all()),
        "discount_monotonic": bool((curve["discount_factor"].diff().fillna(0.0) <= 1e-10).all()),
    }


def phase_fetch_public_rates(start_date: str, write_outputs: bool = True) -> pd.DataFrame:
    if write_outputs and RAW_RATES_PATH.exists():
        daily_rates = _load_frame(RAW_RATES_PATH, parse_dates=["date"])
        if not daily_rates.empty and str(daily_rates["date"].min().date()) <= start_date:
            return daily_rates[daily_rates["date"] >= pd.Timestamp(start_date)].reset_index(drop=True)
    daily_rates = fetch_public_rates(start_date=start_date)
    summary = {
        "start_date": str(daily_rates["date"].min().date()),
        "end_date": str(daily_rates["date"].max().date()),
        "rows": int(len(daily_rates)),
        "series_count": int(len(TREASURY_SERIES)),
        "series_ids": [item["series_id"] for item in TREASURY_SERIES],
        "missing_share_by_series": {
            item["series_id"]: float(daily_rates[item["series_id"]].isna().mean()) for item in TREASURY_SERIES
        },
    }
    if write_outputs:
        write_csv(RAW_RATES_PATH, daily_rates)
        write_json(SUMMARIES_DIR / "fetch_public_rates_summary.json", summary)
    return daily_rates


def phase_build_bootstrap_curves(start_date: str, write_outputs: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    if write_outputs and MONTHLY_RATES_PATH.exists() and BOOTSTRAP_CURVES_PATH.exists():
        monthly_rates = _load_frame(MONTHLY_RATES_PATH, parse_dates=["date"])
        bootstrap_curves = _load_frame(BOOTSTRAP_CURVES_PATH, parse_dates=["date"])
        if not monthly_rates.empty and str(monthly_rates["date"].min().date()) <= start_date:
            monthly_rates = monthly_rates[monthly_rates["date"] >= pd.Timestamp(start_date)].reset_index(drop=True)
            bootstrap_curves = bootstrap_curves[bootstrap_curves["date"] >= pd.Timestamp(start_date)].reset_index(drop=True)
            return monthly_rates, bootstrap_curves
    daily_rates = phase_fetch_public_rates(start_date=start_date, write_outputs=write_outputs)
    monthly_rates = build_monthly_snapshots(daily_rates)
    curve_frames: list[pd.DataFrame] = []
    quality_records: list[dict] = []
    for row in monthly_rates.itertuples(index=False):
        row_series = pd.Series(row._asdict())
        curve = bootstrap_curve_from_row(row_series)
        curve.insert(0, "date", row_series["date"])
        quality = compute_curve_quality(curve)
        quality_records.append({"date": row_series["date"], **quality})
        curve_frames.append(curve)
    bootstrap_curves = pd.concat(curve_frames, ignore_index=True)
    quality_frame = pd.DataFrame(quality_records)
    summary = {
        "monthly_snapshot_count": int(len(monthly_rates)),
        "curve_rows": int(len(bootstrap_curves)),
        "start_date": str(monthly_rates["date"].min().date()),
        "end_date": str(monthly_rates["date"].max().date()),
        "all_discount_positive": bool(quality_frame["discount_positive"].all()),
        "all_discount_monotonic": bool(quality_frame["discount_monotonic"].all()),
        "mean_forward_roughness": float(quality_frame["forward_roughness"].mean()),
        "toy_bootstrap_check": _bootstrap_toy_check(),
    }
    if write_outputs:
        write_csv(MONTHLY_RATES_PATH, monthly_rates)
        write_csv(BOOTSTRAP_CURVES_PATH, bootstrap_curves)
        write_json(SUMMARIES_DIR / "build_bootstrap_curves_summary.json", summary)
    return monthly_rates, bootstrap_curves


def phase_fit_nss_curve(start_date: str, write_outputs: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    if write_outputs and NSS_PARAMETERS_PATH.exists() and NSS_CURVES_PATH.exists():
        parameter_frame = _load_frame(NSS_PARAMETERS_PATH, parse_dates=["date"])
        nss_curves = _load_frame(NSS_CURVES_PATH, parse_dates=["date"])
        if not parameter_frame.empty and str(parameter_frame["date"].min().date()) <= start_date:
            parameter_frame = parameter_frame[parameter_frame["date"] >= pd.Timestamp(start_date)].reset_index(drop=True)
            nss_curves = nss_curves[nss_curves["date"] >= pd.Timestamp(start_date)].reset_index(drop=True)
            return parameter_frame, nss_curves
    _, bootstrap_curves = phase_build_bootstrap_curves(start_date=start_date, write_outputs=write_outputs)
    parameter_records: list[dict] = []
    curve_frames: list[pd.DataFrame] = []
    for date_value, curve in bootstrap_curves.groupby("date", sort=True):
        params = fit_nss_parameters(curve)
        parameter_records.append({"date": date_value, **params})
        fitted_curve = build_nss_curve(params, maturities=BOOTSTRAP_GRID)
        fitted_curve.insert(0, "date", date_value)
        curve_frames.append(fitted_curve)
    parameter_frame = pd.DataFrame(parameter_records)
    nss_curves = pd.concat(curve_frames, ignore_index=True)
    summary = {
        "snapshot_count": int(parameter_frame["date"].nunique()),
        "mean_rmse": float(parameter_frame["rmse"].mean()),
        "mean_mae": float(parameter_frame["mae"].mean()),
        "all_success": bool(parameter_frame["success"].all()),
    }
    if write_outputs:
        write_csv(NSS_PARAMETERS_PATH, parameter_frame)
        write_csv(NSS_CURVES_PATH, nss_curves)
        write_json(SUMMARIES_DIR / "fit_nss_curve_summary.json", summary)
    return parameter_frame, nss_curves


def phase_fit_short_rate_models(start_date: str, write_outputs: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    if write_outputs and MODEL_PARAMETERS_PATH.exists() and MODEL_CURVES_PATH.exists():
        parameter_frame = _load_frame(MODEL_PARAMETERS_PATH, parse_dates=["date"])
        model_curves = _load_frame(MODEL_CURVES_PATH, parse_dates=["date"])
        if not parameter_frame.empty and str(parameter_frame["date"].min().date()) <= start_date:
            parameter_frame = parameter_frame[parameter_frame["date"] >= pd.Timestamp(start_date)].reset_index(drop=True)
            model_curves = model_curves[model_curves["date"] >= pd.Timestamp(start_date)].reset_index(drop=True)
            return parameter_frame, model_curves
    monthly_rates, _ = phase_build_bootstrap_curves(start_date=start_date, write_outputs=write_outputs)
    _, nss_curves = phase_fit_nss_curve(start_date=start_date, write_outputs=write_outputs)
    hw_dynamic = estimate_ou_parameters(monthly_rates["DGS1"] / 100.0)
    parameter_records: list[dict] = []
    curve_records: list[pd.DataFrame] = []

    for date_value, curve in nss_curves.groupby("date", sort=True):
        benchmark_curve = curve.copy().reset_index(drop=True)

        vasicek_params = fit_vasicek_curve(benchmark_curve)
        vasicek_curve = build_vasicek_curve(vasicek_params, maturities=BOOTSTRAP_GRID)
        vasicek_curve.insert(0, "date", date_value)
        vasicek_curve.insert(1, "model", "Vasicek")
        curve_records.append(vasicek_curve)
        parameter_records.append({"date": date_value, **vasicek_params})

        cir_params = fit_cir_curve(benchmark_curve)
        cir_curve = build_cir_curve(cir_params, maturities=BOOTSTRAP_GRID)
        cir_curve.insert(0, "date", date_value)
        cir_curve.insert(1, "model", "CIR")
        curve_records.append(cir_curve)
        parameter_records.append({"date": date_value, **cir_params})

        short_rate = float(benchmark_curve.loc[benchmark_curve["maturity_years"] == 0.5, "zero_yield"].iloc[0])
        hw_params = {
            "model": "Hull-White 1F",
            "a": float(hw_dynamic["a"]),
            "sigma": float(hw_dynamic["sigma"]),
            "r0": short_rate,
            "success": True,
            "rmse": 0.0,
            "mae": 0.0,
            "used_default_dynamic_estimate": bool(hw_dynamic.get("used_default", False)),
        }
        hw_curve = build_hull_white_curve(benchmark_curve)
        hw_curve.insert(0, "date", date_value)
        hw_curve.insert(1, "model", "Hull-White 1F")
        curve_records.append(hw_curve)
        parameter_records.append({"date": date_value, **hw_params})

    parameter_frame = pd.DataFrame(parameter_records)
    model_curves = pd.concat(curve_records, ignore_index=True)
    summary = {
        "snapshot_count": int(parameter_frame["date"].nunique()),
        "model_count": int(parameter_frame["model"].nunique()),
        "hull_white_dynamic_estimate": hw_dynamic,
        "vasicek_mean_rmse": float(parameter_frame.loc[parameter_frame["model"] == "Vasicek", "rmse"].mean()),
        "cir_mean_rmse": float(parameter_frame.loc[parameter_frame["model"] == "CIR", "rmse"].mean()),
        "hull_white_max_abs_error": 0.0,
    }
    if write_outputs:
        write_csv(MODEL_PARAMETERS_PATH, parameter_frame)
        write_csv(MODEL_CURVES_PATH, model_curves)
        write_json(SUMMARIES_DIR / "fit_short_rate_models_summary.json", summary)
    return parameter_frame, model_curves


def _aggregate_fit_metrics(nss_curves: pd.DataFrame, model_curves: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_date_records: list[dict] = []
    for (date_value, model_name), curve in model_curves.groupby(["date", "model"], sort=True):
        benchmark = nss_curves[(nss_curves["date"] == date_value) & (nss_curves["maturity_years"].isin(MODEL_EVAL_GRID))].copy()
        model_eval = curve[curve["maturity_years"].isin(MODEL_EVAL_GRID)].copy()
        merged = benchmark.merge(model_eval, on="maturity_years", suffixes=("_benchmark", "_model"))
        errors = merged["zero_yield_model"] - merged["zero_yield_benchmark"]
        per_date_records.append(
            {
                "date": date_value,
                "model": model_name,
                "zero_yield_rmse": float(np.sqrt(np.mean(np.square(errors)))),
                "zero_yield_mae": float(np.mean(np.abs(errors))),
                "discount_positive": bool((curve["discount_factor"] > 0.0).all()),
                "discount_monotonic": bool((curve["discount_factor"].diff().fillna(0.0) <= 1e-10).all()),
                "forward_roughness": float(np.mean(np.square(np.diff(curve["forward_rate"].to_numpy())))),
            }
        )
    per_date = pd.DataFrame(per_date_records)
    aggregated = (
        per_date.groupby("model", as_index=False)
        .agg(
            zero_yield_rmse=("zero_yield_rmse", "mean"),
            zero_yield_mae=("zero_yield_mae", "mean"),
            forward_roughness=("forward_roughness", "mean"),
            discount_positive_share=("discount_positive", "mean"),
            discount_monotonic_share=("discount_monotonic", "mean"),
            snapshots=("date", "count"),
        )
        .sort_values("zero_yield_rmse")
        .reset_index(drop=True)
    )
    return aggregated, per_date


def _parameter_stability_metrics(parameter_frame: pd.DataFrame) -> pd.DataFrame:
    stability_records: list[dict] = []
    candidate_columns = [column for column in parameter_frame.columns if column not in {"date", "model", "success", "nfev", "used_default_dynamic_estimate"}]
    for model_name, group in parameter_frame.groupby("model", sort=True):
        group = group.sort_values("date").reset_index(drop=True)
        for column in candidate_columns:
            if column not in group.columns:
                continue
            series = pd.to_numeric(group[column], errors="coerce").dropna()
            if len(series) == 0:
                continue
            diffs = series.diff().dropna()
            stability_records.append(
                {
                    "model": model_name,
                    "parameter": column,
                    "mean": float(series.mean()),
                    "std": float(series.std(ddof=1)) if len(series) > 1 else 0.0,
                    "mean_abs_change": float(diffs.abs().mean()) if len(diffs) > 0 else 0.0,
                }
            )
    return pd.DataFrame(stability_records)


def _compute_observed_curve_fit(monthly_rates: pd.DataFrame, nss_curves: pd.DataFrame, model_curves: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    observed_specs = [(item["series_id"], float(item["maturity_years"])) for item in TREASURY_SERIES if float(item["maturity_years"]) >= 1.0]
    detail_records: list[dict] = []
    date_indexed_nss = {date_value: curve.reset_index(drop=True) for date_value, curve in nss_curves.groupby("date", sort=True)}
    model_indexed = {(date_value, model_name): curve.reset_index(drop=True) for (date_value, model_name), curve in model_curves.groupby(["date", "model"], sort=True)}

    for row in monthly_rates.itertuples(index=False):
        date_value = row.date
        benchmark_curve = date_indexed_nss.get(date_value)
        if benchmark_curve is None:
            continue
        curve_map = {"NSS benchmark": benchmark_curve}
        for model_name in ["Vasicek", "CIR", "Hull-White 1F"]:
            model_curve = model_indexed.get((date_value, model_name))
            if model_curve is not None:
                curve_map[model_name] = model_curve
        for series_id, maturity in observed_specs:
            observed_value = getattr(row, series_id)
            if pd.isna(observed_value):
                continue
            observed_yield = float(observed_value) / 100.0
            for model_name, curve in curve_map.items():
                implied_yield = quote_yield_from_discount_curve(curve, maturity)
                detail_records.append(
                    {
                        "date": date_value,
                        "model": model_name,
                        "series_id": series_id,
                        "maturity_years": maturity,
                        "observed_yield": observed_yield,
                        "implied_yield": implied_yield,
                        "yield_error": implied_yield - observed_yield,
                    }
                )
    details = pd.DataFrame(detail_records)
    metrics = (
        details.groupby("model", as_index=False)
        .agg(
            observed_yield_rmse=("yield_error", lambda values: float(np.sqrt(np.mean(np.square(values))))),
            observed_yield_mae=("yield_error", lambda values: float(np.mean(np.abs(values)))),
            points=("yield_error", "count"),
        )
        .sort_values("observed_yield_rmse")
        .reset_index(drop=True)
    )
    return details, metrics


def _plot_curve_examples(nss_curves: pd.DataFrame, model_curves: pd.DataFrame) -> None:
    latest_date = pd.to_datetime(nss_curves["date"]).max()
    benchmark = nss_curves[pd.to_datetime(nss_curves["date"]) == latest_date]
    plt.figure(figsize=(10, 6))
    plt.plot(benchmark["maturity_years"], benchmark["zero_yield"] * 100.0, label="NSS benchmark", linewidth=2.5, color="#111827")
    colors = {"Vasicek": "#1d4ed8", "CIR": "#dc2626", "Hull-White 1F": "#059669"}
    for model_name, curve in model_curves[pd.to_datetime(model_curves["date"]) == latest_date].groupby("model", sort=True):
        plt.plot(curve["maturity_years"], curve["zero_yield"] * 100.0, label=model_name, linewidth=1.8, color=colors.get(model_name))
    plt.title(f"Curve Comparison on {latest_date.date()}")
    plt.xlabel("Maturity (years)")
    plt.ylabel("Zero yield (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"curve_examples_{latest_date:%Y%m%d}.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(benchmark["maturity_years"], benchmark["forward_rate"] * 100.0, label="NSS benchmark", linewidth=2.5, color="#111827")
    for model_name, curve in model_curves[pd.to_datetime(model_curves["date"]) == latest_date].groupby("model", sort=True):
        plt.plot(curve["maturity_years"], curve["forward_rate"] * 100.0, label=model_name, linewidth=1.8, color=colors.get(model_name))
    plt.title(f"Forward Curve Comparison on {latest_date.date()}")
    plt.xlabel("Maturity (years)")
    plt.ylabel("Forward rate (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"forward_curve_comparison_{latest_date:%Y%m%d}.png", dpi=150)
    plt.close()


def _plot_fit_comparison(fit_metrics: pd.DataFrame) -> None:
    ordered = fit_metrics.sort_values("zero_yield_rmse").reset_index(drop=True)
    plt.figure(figsize=(8, 5))
    plt.bar(ordered["model"], ordered["zero_yield_rmse"] * 10000.0, color=["#111827", "#1d4ed8", "#dc2626"][: len(ordered)])
    plt.title("Average Zero-Yield RMSE by Model")
    plt.ylabel("RMSE (basis points)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_fit_comparison_zero_rmse.png", dpi=150)
    plt.close()


def phase_evaluate_models(start_date: str, write_outputs: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    parameter_frame, model_curves = phase_fit_short_rate_models(start_date=start_date, write_outputs=write_outputs)
    _, nss_curves = phase_fit_nss_curve(start_date=start_date, write_outputs=write_outputs)
    monthly_rates, _ = phase_build_bootstrap_curves(start_date=start_date, write_outputs=write_outputs)
    fit_metrics, fit_metrics_per_date = _aggregate_fit_metrics(nss_curves, model_curves)
    stability = _parameter_stability_metrics(parameter_frame)
    observed_details, observed_metrics = _compute_observed_curve_fit(monthly_rates, nss_curves, model_curves)
    _plot_curve_examples(nss_curves, model_curves)
    _plot_fit_comparison(fit_metrics)
    if write_outputs:
        write_csv(METRICS_DIR / "model_fit_metrics.csv", fit_metrics)
        write_csv(METRICS_DIR / "parameter_stability_metrics.csv", stability)
        write_csv(METRICS_DIR / "observed_curve_fit_metrics.csv", observed_metrics)
        write_csv(OBSERVED_CURVE_DETAILS_PATH, observed_details)
    return fit_metrics_per_date, stability


def phase_build_pricing_outputs(start_date: str, write_outputs: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    if write_outputs and PRICING_DETAILS_PATH.exists() and SWAP_DETAILS_PATH.exists():
        pricing_detail = _load_frame(PRICING_DETAILS_PATH, parse_dates=["date"])
        swap_detail = _load_frame(SWAP_DETAILS_PATH, parse_dates=["date"])
        if not pricing_detail.empty and str(pricing_detail["date"].min().date()) <= start_date:
            pricing_detail = pricing_detail[pricing_detail["date"] >= pd.Timestamp(start_date)].reset_index(drop=True)
            swap_detail = swap_detail[swap_detail["date"] >= pd.Timestamp(start_date)].reset_index(drop=True)
            pricing_metrics = (
                pricing_detail.groupby(["model", "maturity_years"], as_index=False)
                .agg(
                    pricing_mae=("pricing_error", lambda values: float(np.mean(np.abs(values)))),
                    pricing_rmse=("pricing_error", lambda values: float(np.sqrt(np.mean(np.square(values))))),
                    mean_model_price=("model_price", "mean"),
                )
                .sort_values(["model", "maturity_years"])
                .reset_index(drop=True)
            )
            swap_metrics = (
                swap_detail.groupby(["model", "maturity_years"], as_index=False)
                .agg(
                    swap_rate_mae=("swap_rate_error", lambda values: float(np.mean(np.abs(values)))),
                    swap_rate_rmse=("swap_rate_error", lambda values: float(np.sqrt(np.mean(np.square(values))))),
                    mean_benchmark_par_swap_rate=("benchmark_par_swap_rate", "mean"),
                    mean_model_par_swap_rate=("model_par_swap_rate", "mean"),
                )
                .sort_values(["model", "maturity_years"])
                .reset_index(drop=True)
            )
            return pricing_metrics, swap_metrics
    _, model_curves = phase_fit_short_rate_models(start_date=start_date, write_outputs=write_outputs)
    _, nss_curves = phase_fit_nss_curve(start_date=start_date, write_outputs=write_outputs)
    pricing_records: list[dict] = []
    swap_records: list[dict] = []
    for date_value, benchmark_curve in nss_curves.groupby("date", sort=True):
        for model_name, model_curve in model_curves[model_curves["date"] == date_value].groupby("model", sort=True):
            pricing_records.extend(build_pricing_records(pd.Timestamp(date_value), benchmark_curve.reset_index(drop=True), model_curve.reset_index(drop=True), model_name))
            swap_records.extend(build_swap_rate_records(pd.Timestamp(date_value), benchmark_curve.reset_index(drop=True), model_curve.reset_index(drop=True), model_name))
    pricing_detail = pd.DataFrame(pricing_records)
    swap_detail = pd.DataFrame(swap_records)
    pricing_metrics = (
        pricing_detail.groupby(["model", "maturity_years"], as_index=False)
        .agg(
            pricing_mae=("pricing_error", lambda values: float(np.mean(np.abs(values)))),
            pricing_rmse=("pricing_error", lambda values: float(np.sqrt(np.mean(np.square(values))))),
            mean_model_price=("model_price", "mean"),
        )
        .sort_values(["model", "maturity_years"])
        .reset_index(drop=True)
    )
    swap_metrics = (
        swap_detail.groupby(["model", "maturity_years"], as_index=False)
        .agg(
            swap_rate_mae=("swap_rate_error", lambda values: float(np.mean(np.abs(values)))),
            swap_rate_rmse=("swap_rate_error", lambda values: float(np.sqrt(np.mean(np.square(values))))),
            mean_benchmark_par_swap_rate=("benchmark_par_swap_rate", "mean"),
            mean_model_par_swap_rate=("model_par_swap_rate", "mean"),
        )
        .sort_values(["model", "maturity_years"])
        .reset_index(drop=True)
    )
    if write_outputs:
        write_csv(PRICING_DETAILS_PATH, pricing_detail)
        write_csv(SWAP_DETAILS_PATH, swap_detail)
        write_csv(METRICS_DIR / "model_pricing_metrics.csv", pricing_metrics)
        write_csv(METRICS_DIR / "swap_rate_comparison.csv", swap_metrics)
    return pricing_metrics, swap_metrics


def phase_results(start_date: str, write_outputs: bool = True) -> dict:
    phase_fetch_public_rates(start_date=start_date, write_outputs=write_outputs)
    phase_build_bootstrap_curves(start_date=start_date, write_outputs=write_outputs)
    phase_fit_nss_curve(start_date=start_date, write_outputs=write_outputs)
    phase_fit_short_rate_models(start_date=start_date, write_outputs=write_outputs)
    phase_evaluate_models(start_date=start_date, write_outputs=write_outputs)
    phase_build_pricing_outputs(start_date=start_date, write_outputs=write_outputs)
    outputs = {
        "raw_rates_path": str(RAW_RATES_PATH),
        "bootstrap_curves_path": str(BOOTSTRAP_CURVES_PATH),
        "nss_parameters_path": str(NSS_PARAMETERS_PATH),
        "model_parameters_path": str(MODEL_PARAMETERS_PATH),
        "metrics_dir": str(METRICS_DIR),
        "figures_dir": str(FIGURES_DIR),
        "summaries_dir": str(SUMMARIES_DIR),
    }
    print(json.dumps(outputs, indent=2))
    return outputs


def run_phase(phase: str) -> None:
    phase = phase.lower()
    start_date = SMOKE_START_DATE if phase == "smoke" else DEFAULT_START_DATE
    if phase == "fetch_public_rates":
        phase_fetch_public_rates(start_date)
    elif phase == "build_bootstrap_curves":
        phase_build_bootstrap_curves(start_date)
    elif phase == "fit_nss_curve":
        phase_fit_nss_curve(start_date)
    elif phase == "fit_short_rate_models":
        phase_fit_short_rate_models(start_date)
    elif phase == "evaluate_models":
        phase_evaluate_models(start_date)
    elif phase == "build_pricing_outputs":
        phase_build_pricing_outputs(start_date)
    elif phase == "results":
        phase_results(start_date)
    elif phase == "smoke":
        phase_results(start_date)
    elif phase == "all":
        phase_results(DEFAULT_START_DATE)
    else:
        raise ValueError(f"Unknown phase: {phase}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed-income term-structure project pipeline.")
    parser.add_argument("--phase", default="all", help="Pipeline phase to run.")
    args = parser.parse_args()
    run_phase(args.phase)


if __name__ == "__main__":
    main()
