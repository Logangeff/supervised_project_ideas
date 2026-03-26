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
    DOCS_DIR,
    FED_BENCHMARK_MATURITIES,
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
from .nss import build_nss_curve, fit_nss_parameters, nss_zero_yield
from .rates import build_monthly_snapshots, fetch_fed_nominal_curve, fetch_public_rates
from .reporting import write_csv, write_json


RAW_RATES_PATH = RAW_DIR / "public_rates_daily.csv"
FED_NOMINAL_DAILY_PATH = RAW_DIR / "fed_nominal_curve_daily.csv"
MONTHLY_RATES_PATH = PROCESSED_DIR / "monthly_rates.csv"
BOOTSTRAP_CURVES_PATH = PROCESSED_DIR / "bootstrap_curves.csv"
NSS_PARAMETERS_PATH = PROCESSED_DIR / "nss_parameters.csv"
NSS_CURVES_PATH = PROCESSED_DIR / "nss_curves.csv"
FED_MONTHLY_PATH = PROCESSED_DIR / "fed_nominal_curve_monthly.csv"
FED_CURVES_PATH = PROCESSED_DIR / "fed_benchmark_curves.csv"
MODEL_PARAMETERS_PATH = PROCESSED_DIR / "model_parameters.csv"
MODEL_CURVES_PATH = PROCESSED_DIR / "model_curves.csv"
PRICING_DETAILS_PATH = PROCESSED_DIR / "pricing_details.csv"
SWAP_DETAILS_PATH = PROCESSED_DIR / "swap_rate_details.csv"
OBSERVED_CURVE_DETAILS_PATH = PROCESSED_DIR / "observed_curve_comparison.csv"
FED_CURVE_DETAILS_PATH = PROCESSED_DIR / "fed_curve_comparison.csv"
TP1_VALIDATION_PATH = PROCESSED_DIR / "tp1_fed_validation.csv"


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


def _safe_fed_params(row: pd.Series) -> dict:
    beta3 = pd.to_numeric(row.get("BETA3"), errors="coerce")
    tau2 = pd.to_numeric(row.get("TAU2"), errors="coerce")
    if pd.isna(beta3):
        beta3 = 0.0
    if pd.isna(tau2) or float(tau2) <= 0.0 or float(tau2) >= 900.0:
        tau2 = max(float(pd.to_numeric(row.get("TAU1"), errors="coerce")), 1e-6)
        beta3 = 0.0
    return {
        "beta0": float(pd.to_numeric(row["BETA0"], errors="coerce")) / 100.0,
        "beta1": float(pd.to_numeric(row["BETA1"], errors="coerce")) / 100.0,
        "beta2": float(pd.to_numeric(row["BETA2"], errors="coerce")) / 100.0,
        "beta3": float(beta3) / 100.0,
        "tau1": float(pd.to_numeric(row["TAU1"], errors="coerce")),
        "tau2": float(tau2),
    }


def phase_fetch_fed_benchmark(start_date: str, write_outputs: bool = True) -> pd.DataFrame:
    if write_outputs and FED_NOMINAL_DAILY_PATH.exists():
        fed_daily = _load_frame(FED_NOMINAL_DAILY_PATH, parse_dates=["date"])
        if not fed_daily.empty and str(fed_daily["date"].min().date()) <= start_date:
            return fed_daily[fed_daily["date"] >= pd.Timestamp(start_date)].reset_index(drop=True)
    fed_daily = fetch_fed_nominal_curve(start_date=start_date)
    sveny_columns = [f"SVENY{maturity:02d}" for maturity in FED_BENCHMARK_MATURITIES]
    summary = {
        "start_date": str(fed_daily["date"].min().date()),
        "end_date": str(fed_daily["date"].max().date()),
        "rows": int(len(fed_daily)),
        "published_sveny_columns": sveny_columns,
        "missing_share_sveny": {column: float(fed_daily[column].isna().mean()) for column in sveny_columns if column in fed_daily.columns},
    }
    if write_outputs:
        write_csv(FED_NOMINAL_DAILY_PATH, fed_daily)
        write_json(SUMMARIES_DIR / "fetch_fed_benchmark_summary.json", summary)
    return fed_daily


def phase_build_fed_benchmark_curves(start_date: str, write_outputs: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if write_outputs and FED_MONTHLY_PATH.exists() and FED_CURVES_PATH.exists():
        fed_monthly = _load_frame(FED_MONTHLY_PATH, parse_dates=["date"])
        fed_curves = _load_frame(FED_CURVES_PATH, parse_dates=["date"])
        if not fed_monthly.empty and str(fed_monthly["date"].min().date()) <= start_date:
            fed_monthly = fed_monthly[fed_monthly["date"] >= pd.Timestamp(start_date)].reset_index(drop=True)
            fed_curves = fed_curves[fed_curves["date"] >= pd.Timestamp(start_date)].reset_index(drop=True)
            reconstruction_check = _load_frame(METRICS_DIR / "fed_reconstruction_check.csv", parse_dates=["date"])
            reconstruction_check = reconstruction_check[reconstruction_check["date"] >= pd.Timestamp(start_date)].reset_index(drop=True)
            return fed_monthly, fed_curves, reconstruction_check

    fed_daily = phase_fetch_fed_benchmark(start_date=start_date, write_outputs=write_outputs)
    fed_monthly = build_monthly_snapshots(fed_daily)
    curve_records: list[dict] = []
    reconstruction_records: list[dict] = []

    for row in fed_monthly.itertuples(index=False):
        row_series = pd.Series(row._asdict())
        params = _safe_fed_params(row_series)
        for maturity in BOOTSTRAP_GRID:
            zero_yield = nss_zero_yield(maturity, **params)
            curve_records.append(
                {
                    "date": row_series["date"],
                    "maturity_years": maturity,
                    "zero_yield": float(zero_yield),
                    "discount_factor": float(np.exp(-zero_yield * maturity)),
                }
            )
        for maturity in FED_BENCHMARK_MATURITIES:
            column = f"SVENY{maturity:02d}"
            published = pd.to_numeric(row_series.get(column), errors="coerce")
            if pd.isna(published):
                continue
            reconstructed = nss_zero_yield(float(maturity), **params) * 100.0
            reconstruction_records.append(
                {
                    "date": row_series["date"],
                    "maturity_years": float(maturity),
                    "published_sveny": float(published),
                    "reconstructed_sveny": float(reconstructed),
                    "reconstruction_error": float(reconstructed - published),
                }
            )

    fed_curves = pd.DataFrame(curve_records).sort_values(["date", "maturity_years"]).reset_index(drop=True)
    forward_frames: list[pd.DataFrame] = []
    for _, frame in fed_curves.groupby("date", sort=False):
        frame = frame.copy().reset_index(drop=True)
        forward_rates = [float(frame.iloc[0]["zero_yield"])]
        for idx in range(1, len(frame)):
            forward_rates.append(
                float(
                    -np.log(frame.iloc[idx]["discount_factor"] / frame.iloc[idx - 1]["discount_factor"])
                    / (frame.iloc[idx]["maturity_years"] - frame.iloc[idx - 1]["maturity_years"])
                )
            )
        frame["forward_rate"] = forward_rates
        forward_frames.append(frame)
    fed_curves = pd.concat(forward_frames, ignore_index=True)
    reconstruction_check = pd.DataFrame(reconstruction_records).sort_values(["date", "maturity_years"]).reset_index(drop=True)
    recon_summary = {
        "monthly_snapshot_count": int(fed_monthly["date"].nunique()),
        "reconstruction_rmse_bps": float(np.sqrt(np.mean(np.square(reconstruction_check["reconstruction_error"]))) * 100.0),
        "reconstruction_mae_bps": float(np.mean(np.abs(reconstruction_check["reconstruction_error"])) * 100.0),
        "overlap_points": int(len(reconstruction_check)),
    }

    tp1_path = DOCS_DIR / "TP1 - Submission" / "TP1 data 60201 W2026.xlsx"
    tp1_summary = {"available": False}
    if tp1_path.exists():
        try:
            tp1_data = pd.read_excel(tp1_path)
            tp1_data["Date"] = pd.to_datetime(tp1_data["Date"], errors="coerce")
            overlap = tp1_data.merge(
                fed_monthly[["date", "BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2"]],
                left_on="Date",
                right_on="date",
                how="inner",
            )
            if not overlap.empty:
                tp1_records: list[dict] = []
                for row in overlap.itertuples(index=False):
                    tp1_params = {
                        "beta0": float(row.BETA0_x) / 100.0,
                        "beta1": float(row.BETA1_x) / 100.0,
                        "beta2": float(row.BETA2_x) / 100.0,
                        "beta3": float(row.BETA3_x) / 100.0,
                        "tau1": float(row.TAU1_x),
                        "tau2": float(row.TAU2_x),
                    }
                    fed_params = _safe_fed_params(
                        pd.Series(
                            {
                                "BETA0": row.BETA0_y,
                                "BETA1": row.BETA1_y,
                                "BETA2": row.BETA2_y,
                                "BETA3": row.BETA3_y,
                                "TAU1": row.TAU1_y,
                                "TAU2": row.TAU2_y,
                            }
                        )
                    )
                    for maturity in FED_BENCHMARK_MATURITIES:
                        tp1_yield = nss_zero_yield(float(maturity), **tp1_params) * 100.0
                        fed_yield = nss_zero_yield(float(maturity), **fed_params) * 100.0
                        tp1_records.append(
                            {
                                "date": row.Date,
                                "maturity_years": float(maturity),
                                "tp1_sveny": float(tp1_yield),
                                "fed_sveny": float(fed_yield),
                                "yield_error": float(tp1_yield - fed_yield),
                            }
                        )
                tp1_validation = pd.DataFrame(tp1_records)
                tp1_summary = {
                    "available": True,
                    "overlap_dates": int(tp1_validation["date"].nunique()),
                    "yield_rmse_bps": float(np.sqrt(np.mean(np.square(tp1_validation["yield_error"]))) * 100.0),
                    "yield_mae_bps": float(np.mean(np.abs(tp1_validation["yield_error"])) * 100.0),
                }
                if write_outputs:
                    write_csv(TP1_VALIDATION_PATH, tp1_validation)
        except Exception as exc:
            tp1_summary = {"available": True, "error": str(exc)}

    if write_outputs:
        write_csv(FED_MONTHLY_PATH, fed_monthly)
        write_csv(FED_CURVES_PATH, fed_curves)
        write_csv(METRICS_DIR / "fed_reconstruction_check.csv", reconstruction_check)
        write_json(SUMMARIES_DIR / "build_fed_benchmark_summary.json", {**recon_summary, "tp1_validation": tp1_summary})
    return fed_monthly, fed_curves, reconstruction_check


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


def _compute_fed_curve_fit(fed_monthly: pd.DataFrame, fed_curves: pd.DataFrame, nss_curves: pd.DataFrame, model_curves: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_records: list[dict] = []
    fed_curve_map = {date_value: curve.reset_index(drop=True) for date_value, curve in fed_curves.groupby("date", sort=True)}
    nss_curve_map = {date_value: curve.reset_index(drop=True) for date_value, curve in nss_curves.groupby("date", sort=True)}
    model_curve_map = {(date_value, model_name): curve.reset_index(drop=True) for (date_value, model_name), curve in model_curves.groupby(["date", "model"], sort=True)}

    for row in fed_monthly.itertuples(index=False):
        date_value = row.date
        fed_curve = fed_curve_map.get(date_value)
        nss_curve = nss_curve_map.get(date_value)
        if fed_curve is None or nss_curve is None:
            continue
        curve_map = {"Fed published benchmark": fed_curve, "NSS benchmark": nss_curve}
        for model_name in ["Vasicek", "CIR", "Hull-White 1F"]:
            model_curve = model_curve_map.get((date_value, model_name))
            if model_curve is not None:
                curve_map[model_name] = model_curve
        for maturity in FED_BENCHMARK_MATURITIES:
            column = f"SVENY{maturity:02d}"
            published = pd.to_numeric(getattr(row, column), errors="coerce")
            if pd.isna(published):
                continue
            benchmark_yield = float(published) / 100.0
            for model_name, curve in curve_map.items():
                curve_slice = curve[curve["maturity_years"] == float(maturity)]
                if curve_slice.empty:
                    continue
                implied_yield = float(curve_slice["zero_yield"].iloc[0])
                detail_records.append(
                    {
                        "date": date_value,
                        "model": model_name,
                        "maturity_years": float(maturity),
                        "fed_benchmark_yield": benchmark_yield,
                        "implied_yield": implied_yield,
                        "yield_error": implied_yield - benchmark_yield,
                    }
                )
    details = pd.DataFrame(detail_records)
    metrics = (
        details.groupby("model", as_index=False)
        .agg(
            fed_yield_rmse=("yield_error", lambda values: float(np.sqrt(np.mean(np.square(values))))),
            fed_yield_mae=("yield_error", lambda values: float(np.mean(np.abs(values)))),
            points=("yield_error", "count"),
        )
        .sort_values("fed_yield_rmse")
        .reset_index(drop=True)
    )
    return details, metrics


def _plot_curve_examples(fed_curves: pd.DataFrame, nss_curves: pd.DataFrame, model_curves: pd.DataFrame) -> None:
    latest_date = pd.to_datetime(nss_curves["date"]).max()
    fed_benchmark = fed_curves[pd.to_datetime(fed_curves["date"]) == latest_date]
    benchmark = nss_curves[pd.to_datetime(nss_curves["date"]) == latest_date]
    plt.figure(figsize=(10, 6))
    plt.plot(fed_benchmark["maturity_years"], fed_benchmark["zero_yield"] * 100.0, label="Fed published benchmark", linewidth=2.5, color="#7c3aed")
    plt.plot(benchmark["maturity_years"], benchmark["zero_yield"] * 100.0, label="NSS benchmark", linewidth=2.5, color="#111827")
    colors = {"Vasicek": "#1d4ed8", "CIR": "#dc2626", "Hull-White 1F": "#059669"}
    for model_name, curve in model_curves[pd.to_datetime(model_curves["date"]) == latest_date].groupby("model", sort=True):
        plt.plot(curve["maturity_years"], curve["zero_yield"] * 100.0, label=model_name, linewidth=1.8, color=colors.get(model_name))
    plt.title(f"Fed / NSS / Model Curve Comparison on {latest_date.date()}")
    plt.xlabel("Maturity (years)")
    plt.ylabel("Zero yield (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"curve_examples_{latest_date:%Y%m%d}.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(fed_benchmark["maturity_years"], fed_benchmark["forward_rate"] * 100.0, label="Fed published benchmark", linewidth=2.5, color="#7c3aed")
    plt.plot(benchmark["maturity_years"], benchmark["forward_rate"] * 100.0, label="NSS benchmark", linewidth=2.5, color="#111827")
    for model_name, curve in model_curves[pd.to_datetime(model_curves["date"]) == latest_date].groupby("model", sort=True):
        plt.plot(curve["maturity_years"], curve["forward_rate"] * 100.0, label=model_name, linewidth=1.8, color=colors.get(model_name))
    plt.title(f"Fed / NSS / Model Forward Curve Comparison on {latest_date.date()}")
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
    fed_monthly, fed_curves, fed_reconstruction_check = phase_build_fed_benchmark_curves(start_date=start_date, write_outputs=write_outputs)
    fit_metrics, fit_metrics_per_date = _aggregate_fit_metrics(nss_curves, model_curves)
    stability = _parameter_stability_metrics(parameter_frame)
    observed_details, observed_metrics = _compute_observed_curve_fit(monthly_rates, nss_curves, model_curves)
    fed_details, fed_metrics = _compute_fed_curve_fit(fed_monthly, fed_curves, nss_curves, model_curves)
    _plot_curve_examples(fed_curves, nss_curves, model_curves)
    _plot_fit_comparison(fit_metrics)
    if write_outputs:
        write_csv(METRICS_DIR / "model_fit_metrics.csv", fit_metrics)
        write_csv(METRICS_DIR / "parameter_stability_metrics.csv", stability)
        write_csv(METRICS_DIR / "observed_curve_fit_metrics.csv", observed_metrics)
        write_csv(OBSERVED_CURVE_DETAILS_PATH, observed_details)
        write_csv(METRICS_DIR / "fed_curve_fit_metrics.csv", fed_metrics)
        write_csv(FED_CURVE_DETAILS_PATH, fed_details)
        write_json(
            SUMMARIES_DIR / "phase1_benchmark_summary.json",
            {
                "observed_curve_best_model": observed_metrics.iloc[0]["model"],
                "observed_curve_best_rmse_bps": float(observed_metrics.iloc[0]["observed_yield_rmse"] * 10000.0),
                "fed_curve_best_model": fed_metrics.iloc[0]["model"],
                "fed_curve_best_rmse_bps": float(fed_metrics.iloc[0]["fed_yield_rmse"] * 10000.0),
                "fed_reconstruction_rmse_bps": float(np.sqrt(np.mean(np.square(fed_reconstruction_check["reconstruction_error"]))) * 100.0),
            },
        )
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
    phase_fetch_fed_benchmark(start_date=start_date, write_outputs=write_outputs)
    phase_build_bootstrap_curves(start_date=start_date, write_outputs=write_outputs)
    phase_fit_nss_curve(start_date=start_date, write_outputs=write_outputs)
    phase_build_fed_benchmark_curves(start_date=start_date, write_outputs=write_outputs)
    phase_fit_short_rate_models(start_date=start_date, write_outputs=write_outputs)
    phase_evaluate_models(start_date=start_date, write_outputs=write_outputs)
    phase_build_pricing_outputs(start_date=start_date, write_outputs=write_outputs)
    outputs = {
        "raw_rates_path": str(RAW_RATES_PATH),
        "fed_nominal_path": str(FED_NOMINAL_DAILY_PATH),
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
    elif phase == "fetch_fed_benchmark":
        phase_fetch_fed_benchmark(start_date)
    elif phase == "build_bootstrap_curves":
        phase_build_bootstrap_curves(start_date)
    elif phase == "build_fed_benchmark_curves":
        phase_build_fed_benchmark_curves(start_date)
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
