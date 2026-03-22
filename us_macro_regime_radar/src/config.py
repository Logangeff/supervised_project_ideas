from __future__ import annotations

import random
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_FRED_DIR = RAW_DATA_DIR / "fred"
CATALOG_DIR = DATA_DIR / "catalog"
PROCESSED_DIR = DATA_DIR / "processed"

OUTPUT_DIR = ROOT_DIR / "outputs"
SUMMARIES_DIR = OUTPUT_DIR / "summaries"
METRICS_DIR = OUTPUT_DIR / "metrics"
DASHBOARD_DIR = OUTPUT_DIR / "dashboard"
FIGURES_DIR = OUTPUT_DIR / "figures"
MODELS_DIR = OUTPUT_DIR / "models"

APP_DIR = ROOT_DIR / "app"
LOCAL_CONFIG_DIR = ROOT_DIR / "config"

FRED_API_KEY_JSON = LOCAL_CONFIG_DIR / "fred_api_key.json"
FRED_API_KEY_TEMPLATE_JSON = LOCAL_CONFIG_DIR / "fred_api_key.template.json"
SERIES_CATALOG_CSV = CATALOG_DIR / "us_series_catalog.csv"

SEED = 42
MONTHS_PER_YEAR = 12

RAW_FETCH_START_DATE = "1960-01-01"
WARMUP_TRAIN_END = "1989-12-31"
VALIDATION_END = "2009-12-31"

CATALOG_CACHE_PATH = PROCESSED_DIR / "series_catalog_snapshot.parquet"
MONTHLY_PANEL_PATH = PROCESSED_DIR / "monthly_feature_panel.parquet"
CYCLE_LABEL_PANEL_PATH = PROCESSED_DIR / "cycle_labeled_panel.parquet"

FETCH_MACRO_DATA_SUMMARY_PATH = SUMMARIES_DIR / "fetch_macro_data_summary.json"
BUILD_MONTHLY_PANEL_SUMMARY_PATH = SUMMARIES_DIR / "build_monthly_panel_summary.json"
BUILD_CYCLE_LABELS_SUMMARY_PATH = SUMMARIES_DIR / "build_cycle_labels_summary.json"
SMOKE_SUMMARY_PATH = SUMMARIES_DIR / "smoke_summary.json"

PHASE_FORECAST_METRICS_JSON = METRICS_DIR / "phase_forecast_metrics.json"
PHASE_FORECAST_METRICS_CSV = METRICS_DIR / "phase_forecast_metrics.csv"
PHASE_FORECAST_PREDICTIONS_CSV = METRICS_DIR / "phase_forecast_predictions.csv"

RECESSION_RISK_METRICS_JSON = METRICS_DIR / "recession_risk_metrics.json"
RECESSION_RISK_METRICS_CSV = METRICS_DIR / "recession_risk_metrics.csv"
RECESSION_RISK_PREDICTIONS_CSV = METRICS_DIR / "recession_risk_predictions.csv"

HISTORY_PAYLOAD_JSON = DASHBOARD_DIR / "history_payload.json"
LATEST_SNAPSHOT_JSON = DASHBOARD_DIR / "latest_snapshot.json"
FORECAST_PAYLOAD_JSON = DASHBOARD_DIR / "forecast_payload.json"

PHASE_CONFUSION_DIR = FIGURES_DIR / "phase_confusion"
RECESSION_PR_CURVE_FIGURE = FIGURES_DIR / "recession_risk_pr_curve.png"
RECESSION_ROC_CURVE_FIGURE = FIGURES_DIR / "recession_risk_roc_curve.png"
PHASE_SCORE_TIMELINE_FIGURE = FIGURES_DIR / "phase_score_timeline.png"

PHASE_MODEL_PATH = MODELS_DIR / "selected_phase_model_bundle.joblib"
RECESSION_MODEL_PATH = MODELS_DIR / "selected_recession_model_bundle.joblib"

PHASE_CLASSES = ["Expansion", "Slowdown", "Contraction", "Recovery"]
PHASE_CLASS_TO_INT = {label: index for index, label in enumerate(PHASE_CLASSES)}
PHASE_INT_TO_CLASS = {index: label for label, index in PHASE_CLASS_TO_INT.items()}
PHASE_HORIZONS = [1, 2, 3, 4, 5, 6]
FIXED_BINARY_THRESHOLDS = [0.30, 0.50]


def ensure_project_dirs() -> None:
    for path in (
        DATA_DIR,
        RAW_DATA_DIR,
        RAW_FRED_DIR,
        CATALOG_DIR,
        PROCESSED_DIR,
        OUTPUT_DIR,
        SUMMARIES_DIR,
        METRICS_DIR,
        DASHBOARD_DIR,
        FIGURES_DIR,
        PHASE_CONFUSION_DIR,
        MODELS_DIR,
        APP_DIR,
        LOCAL_CONFIG_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
