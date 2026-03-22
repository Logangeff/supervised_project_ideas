from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import (
    CATALOG_CACHE_PATH,
    CYCLE_LABEL_PANEL_PATH,
    FORECAST_PAYLOAD_JSON,
    HISTORY_PAYLOAD_JSON,
    LATEST_SNAPSHOT_JSON,
    PHASE_CLASS_TO_INT,
    PHASE_CLASSES,
    PHASE_FORECAST_METRICS_JSON,
    PHASE_FORECAST_PREDICTIONS_CSV,
    PHASE_HORIZONS,
    PHASE_MODEL_PATH,
    RECESSION_MODEL_PATH,
    RECESSION_RISK_METRICS_JSON,
    RECESSION_RISK_PREDICTIONS_CSV,
)
from .utils import load_json, save_json


def _load_catalog_map() -> dict[str, str]:
    if not CATALOG_CACHE_PATH.exists():
        return {}
    catalog = pd.read_parquet(CATALOG_CACHE_PATH)
    return {row["series_id"]: row["display_name"] for _, row in catalog.iterrows()}


def _feature_source_name(feature_name: str, catalog_map: dict[str, str]) -> str:
    if feature_name in {"activity_score", "level_score", "momentum_score", "term_spread", "baa_aaa_spread", "sp500_ret_1m", "rolling_rv_3m", "drawdown_6m"}:
        return feature_name.replace("_", " ").title()
    for suffix in ("_current", "_chg3", "_chg6"):
        if feature_name.endswith(suffix):
            series_id = feature_name.removesuffix(suffix)
            return catalog_map.get(series_id, series_id)
    return feature_name


def _latest_phase_forecast(bundle: dict[str, object], latest_row: pd.Series) -> list[dict[str, object]]:
    model_name = str(bundle["model_name"])
    feature_cols = bundle.get("feature_cols", [])
    forecasts: list[dict[str, object]] = []
    for horizon in PHASE_HORIZONS:
        horizon_bundle = bundle["horizons"][horizon]
        if model_name == "phase_persistence":
            probabilities = np.zeros(len(PHASE_CLASSES))
            current_phase_int = int(latest_row["current_phase_int"])
            probabilities[current_phase_int] = 1.0
        elif model_name == "phase_markov_baseline":
            counts = np.asarray(horizon_bundle, dtype=float)
            current_phase_int = int(latest_row["current_phase_int"])
            probabilities = counts[current_phase_int] / counts[current_phase_int].sum()
        else:
            estimator = horizon_bundle["estimator"]
            usable_features = horizon_bundle["feature_cols"]
            probabilities = estimator.predict_proba(pd.DataFrame([latest_row[usable_features].to_dict()]))[0]
        best_index = int(np.argmax(probabilities))
        forecasts.append(
            {
                "horizon_months": horizon,
                "most_likely_phase": PHASE_CLASSES[best_index],
                "confidence": float(probabilities[best_index]),
                "phase_probabilities": {PHASE_CLASSES[idx]: float(probabilities[idx]) for idx in range(len(PHASE_CLASSES))},
            }
        )
    return forecasts


def _latest_recession_probability(bundle: dict[str, object], latest_row: pd.Series) -> float:
    estimator = bundle["estimator"]
    feature_cols = bundle["feature_cols"]
    calibrator = bundle["calibrator"]
    raw_probability = float(estimator.predict_proba(pd.DataFrame([latest_row[feature_cols].to_dict()]))[0, 1])
    return float(calibrator.transform([raw_probability])[0])


def _latest_driver_rows(bundle: dict[str, object], panel: pd.DataFrame, latest_row: pd.Series) -> list[dict[str, object]]:
    feature_cols = list(bundle["feature_cols"])
    catalog_map = _load_catalog_map()
    feature_frame = panel[feature_cols].copy()
    latest_values = latest_row[feature_cols].astype(float)
    z_scores = ((latest_values - feature_frame.mean()) / feature_frame.std(ddof=0)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    estimator = bundle["estimator"]
    if hasattr(estimator, "named_steps") and "model" in estimator.named_steps:
        model = estimator.named_steps["model"]
        if hasattr(model, "coef_"):
            transformed = estimator.named_steps["imputer"].transform(pd.DataFrame([latest_values.to_dict()]))
            if "scaler" in estimator.named_steps:
                transformed = estimator.named_steps["scaler"].transform(transformed)
            contributions = transformed[0] * model.coef_[0]
            importances = {feature_cols[idx]: float(contributions[idx]) for idx in range(len(feature_cols))}
        elif hasattr(model, "feature_importances_"):
            importances = {feature_cols[idx]: float(model.feature_importances_[idx]) for idx in range(len(feature_cols))}
        else:
            importances = {name: abs(float(z_scores[name])) for name in feature_cols}
    else:
        importances = {name: abs(float(z_scores[name])) for name in feature_cols}
    ranked = sorted(importances.items(), key=lambda item: abs(item[1]), reverse=True)[:12]
    return [
        {
            "feature": feature_name,
            "source_name": _feature_source_name(feature_name, catalog_map),
            "importance": float(importance),
            "latest_value": float(latest_values[feature_name]) if pd.notna(latest_values[feature_name]) else None,
            "z_score": float(z_scores[feature_name]) if pd.notna(z_scores[feature_name]) else None,
        }
        for feature_name, importance in ranked
    ]


def build_dashboard_payload() -> dict[str, object]:
    labeled_panel = pd.read_parquet(CYCLE_LABEL_PANEL_PATH)
    labeled_panel["date"] = pd.to_datetime(labeled_panel["date"])

    recession_metrics = load_json(RECESSION_RISK_METRICS_JSON)
    phase_metrics = load_json(PHASE_FORECAST_METRICS_JSON)
    if recession_metrics is None or phase_metrics is None:
        raise FileNotFoundError("Missing trained model metrics. Run training phases before building dashboard payloads.")

    selected_risk_model = str(recession_metrics["selected_model"])
    selected_phase_model = str(phase_metrics["selected_model"])
    risk_predictions = pd.read_csv(RECESSION_RISK_PREDICTIONS_CSV, parse_dates=["date"])
    phase_predictions = pd.read_csv(PHASE_FORECAST_PREDICTIONS_CSV, parse_dates=["date"])
    phase_bundle = joblib.load(PHASE_MODEL_PATH)
    recession_bundle = joblib.load(RECESSION_MODEL_PATH)

    selected_risk_predictions = risk_predictions[risk_predictions["model"] == selected_risk_model].copy()
    if "calibrated_probability" in selected_risk_predictions.columns:
        selected_risk_predictions["display_probability"] = selected_risk_predictions["calibrated_probability"].fillna(selected_risk_predictions["probability"])
    else:
        selected_risk_predictions["display_probability"] = selected_risk_predictions["probability"]

    history = labeled_panel[["date", "current_phase", "current_phase_int", "level_score", "momentum_score", "recession_start", "USRECDM_current"]].copy()
    history = history.merge(
        selected_risk_predictions[["date", "split", "display_probability"]].rename(columns={"display_probability": "recession_risk_probability"}),
        on="date",
        how="left",
    )
    history["is_recession_month"] = history["USRECDM_current"].fillna(0).astype(int)
    history_payload = {
        "selected_phase_model": selected_phase_model,
        "selected_recession_model": selected_risk_model,
        "rows": [
            {
                "date": row["date"].strftime("%Y-%m-%d"),
                "current_phase": row["current_phase"],
                "current_phase_int": int(row["current_phase_int"]) if pd.notna(row["current_phase_int"]) else None,
                "level_score": float(row["level_score"]) if pd.notna(row["level_score"]) else None,
                "momentum_score": float(row["momentum_score"]) if pd.notna(row["momentum_score"]) else None,
                "is_recession_month": int(row["is_recession_month"]),
                "recession_risk_probability": float(row["recession_risk_probability"]) if pd.notna(row["recession_risk_probability"]) else None,
                "split": (None if ("split" not in history.columns or pd.isna(row["split"])) else str(row["split"])),
            }
            for _, row in history.iterrows()
        ],
    }
    save_json(HISTORY_PAYLOAD_JSON, history_payload)

    latest_row = labeled_panel.dropna(subset=["current_phase_int"]).sort_values("date").iloc[-1]
    forecast_rows = _latest_phase_forecast(phase_bundle, latest_row)
    recession_probability = _latest_recession_probability(recession_bundle, latest_row)
    driver_rows = _latest_driver_rows(recession_bundle, labeled_panel, latest_row)
    next_phase = forecast_rows[0]

    latest_snapshot = {
        "date": latest_row["date"].strftime("%Y-%m-%d"),
        "current_phase": latest_row["current_phase"],
        "current_phase_int": int(latest_row["current_phase_int"]),
        "level_score": float(latest_row["level_score"]),
        "momentum_score": float(latest_row["momentum_score"]),
        "recession_within_6m_probability": recession_probability,
        "selected_phase_model": selected_phase_model,
        "selected_recession_model": selected_risk_model,
        "most_likely_next_phase": next_phase["most_likely_phase"],
        "next_phase_confidence": next_phase["confidence"],
        "driver_rows": driver_rows,
    }
    save_json(LATEST_SNAPSHOT_JSON, latest_snapshot)

    forecast_payload = {
        "generated_for_date": latest_row["date"].strftime("%Y-%m-%d"),
        "selected_phase_model": selected_phase_model,
        "forecasts": forecast_rows,
        "phase_probabilities_long": [
            {
                "horizon_months": row["horizon_months"],
                "phase": phase_name,
                "probability": probability,
            }
            for row in forecast_rows
            for phase_name, probability in row["phase_probabilities"].items()
        ],
    }
    save_json(FORECAST_PAYLOAD_JSON, forecast_payload)

    return {
        "latest_snapshot": latest_snapshot,
        "forecast_count": len(forecast_rows),
        "history_rows": len(history_payload["rows"]),
    }
