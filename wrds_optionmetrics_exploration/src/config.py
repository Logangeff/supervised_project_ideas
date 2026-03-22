from __future__ import annotations

import random
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_STOCK_DIR = RAW_DATA_DIR / "stock"
RAW_OPTIONS_DIR = RAW_DATA_DIR / "options"
UNIVERSE_DIR = DATA_DIR / "universe"
PROCESSED_DIR = DATA_DIR / "processed"

OUTPUT_DIR = ROOT_DIR / "outputs"
SUMMARIES_DIR = OUTPUT_DIR / "summaries"
METRICS_DIR = OUTPUT_DIR / "metrics"
FIGURES_DIR = OUTPUT_DIR / "figures"
MODELS_DIR = OUTPUT_DIR / "models"

LOCAL_CONFIG_DIR = ROOT_DIR / "config"
WRDS_CREDENTIALS_JSON = LOCAL_CONFIG_DIR / "wrds_credentials.json"
WRDS_CREDENTIALS_TEMPLATE_JSON = LOCAL_CONFIG_DIR / "wrds_credentials.template.json"

SEED = 42
TRADING_DAYS_PER_YEAR = 252

UNIVERSE_CSV = UNIVERSE_DIR / "current_sp100_like_tickers.csv"

PROJECT_START_DATE = "2016-01-01"
TRAIN_END_DATE = "2021-12-31"
VALIDATION_END_DATE = "2022-12-31"
PROJECT_END_DATE = "2024-12-31"

FORWARD_RV_WINDOW = 20
SHORT_RETURN_WINDOW = 5
LONG_RETURN_WINDOW = 20
HIGH_RV_QUANTILE = 0.80

CLASSICAL_C_GRID = [0.01, 0.1, 1.0, 10.0]
TRAILING_WINDOW_YEARS = [2, 5]
TRAILING_RETRAIN_MONTHS = 3
XGBOOST_PARAM_GRID = [
    {
        "max_depth": 2,
        "learning_rate": 0.05,
        "n_estimators": 200,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
    },
    {
        "max_depth": 3,
        "learning_rate": 0.05,
        "n_estimators": 300,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
    },
    {
        "max_depth": 3,
        "learning_rate": 0.10,
        "n_estimators": 200,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 3,
    },
]

STOCK_FEATURES = ["ret_1d", "ret_5d", "ret_20d", "rv_5d", "rv_20d", "turnover_20d"]
OPTION_FEATURES = [
    "iv_atm_short",
    "iv_atm_long",
    "iv_term_slope",
    "put_skew_short",
    "pc_oi_ratio_short",
    "pc_vol_ratio_short",
    "log_total_oi_short",
]
SURFACE_FEATURES_RAW = [
    "surface_atm_short_raw",
    "surface_atm_long_raw",
    "surface_term_slope_raw",
    "surface_put_25_short_raw",
    "surface_call_25_short_raw",
    "surface_rr_25_short_raw",
    "surface_bf_25_short_raw",
]
SURFACE_FEATURES = [
    "surface_atm_short",
    "surface_atm_long",
    "surface_term_slope",
    "surface_put_25_short",
    "surface_call_25_short",
    "surface_rr_25_short",
    "surface_bf_25_short",
]
CALIBRATED_SURFACE_FEATURES_RAW = [
    "surface_beta_1_raw",
    "surface_beta_2_raw",
    "surface_beta_3_raw",
    "surface_beta_4_raw",
    "surface_beta_5_raw",
]
CALIBRATED_SURFACE_FEATURES = [
    "surface_beta_1",
    "surface_beta_2",
    "surface_beta_3",
    "surface_beta_4",
    "surface_beta_5",
]
TEXT_NEWS_START_DATE = "2020-01-01"
TEXT_NEWS_END_DATE = "2023-12-31"
TEXT_NEWS_TIMEZONE = "America/New_York"
TEXT_NEWS_ARCHIVE_DIR = ROOT_DIR.parent / "numerical_claim_detection_project" / "data" / "raw" / "financial_news"
LOCKED_NUMCLAIM_PROJECT = ROOT_DIR.parent / "numerical_claim_detection_project"
LOCKED_CLAIM_DETECTOR_META = LOCKED_NUMCLAIM_PROJECT / "outputs" / "models" / "stage1_best_detector.json"
TEXT_FINBERT_MODEL_ID = "ProsusAI/finbert"
TEXT_FINBERT_MAX_LENGTH = 128
TEXT_FINBERT_BATCH_SIZE = 128
TEXT_NEWS_FEATURES = [
    "news_count",
    "news_count_log1p",
    "claim_prob_mean",
    "claim_prob_max",
    "claim_count_above_05",
    "sentiment_neg_mean",
    "sentiment_neu_mean",
    "sentiment_pos_mean",
    "sentiment_net_mean",
]

CRSP_NAMES_TABLE_CANDIDATES = ["crsp.stocknames", "crsp.dsenames"]
CRSP_DAILY_TABLE = "crsp.dsf"
OPTION_SECURITIES_TABLE = "optionm.secnmd"
OPTION_PRICES_TABLE = "optionm.opprcd"

RAW_STOCK_NAMES_PATH = RAW_STOCK_DIR / "crsp_stock_names.parquet"
RAW_STOCK_DAILY_PATH = RAW_STOCK_DIR / "crsp_daily_stock.parquet"
RAW_OPTION_SECURITIES_PATH = RAW_OPTIONS_DIR / "optionm_securities.parquet"
RAW_OPTION_PRICES_PATH = RAW_OPTIONS_DIR / "optionm_option_prices.parquet"

STOCK_IDENTIFIER_MAP_PATH = PROCESSED_DIR / "stock_identifier_map.parquet"
OPTION_SECURITY_LINK_PATH = PROCESSED_DIR / "option_security_link.parquet"
STOCK_PANEL_PATH = PROCESSED_DIR / "stock_panel.parquet"
OPTION_FEATURE_PANEL_PATH = PROCESSED_DIR / "option_feature_panel.parquet"
COMPLETE_CASE_PANEL_PATH = PROCESSED_DIR / "complete_case_panel.parquet"
SURFACE_FACTOR_PANEL_PATH = PROCESSED_DIR / "surface_factor_panel.parquet"
SURFACE_EXTENSION_PANEL_PATH = PROCESSED_DIR / "surface_extension_panel.parquet"
CALIBRATED_SURFACE_PANEL_PATH = PROCESSED_DIR / "calibrated_surface_panel.parquet"
CALIBRATED_SURFACE_EXTENSION_PANEL_PATH = PROCESSED_DIR / "calibrated_surface_extension_panel.parquet"
TEXT_NEWS_ARTICLE_PANEL_PATH = PROCESSED_DIR / "text_news_article_panel.parquet"
TEXT_NEWS_DAILY_FEATURE_PANEL_PATH = PROCESSED_DIR / "text_news_daily_feature_panel.parquet"
TEXT_NEWS_EXTENSION_PANEL_PATH = PROCESSED_DIR / "text_news_extension_panel.parquet"
TEXT_NEWS_EVENT_PANEL_PATH = PROCESSED_DIR / "text_news_event_panel.parquet"

EXTRACT_STOCK_SUMMARY_PATH = SUMMARIES_DIR / "extract_stock_data_summary.json"
BUILD_STOCK_PANEL_SUMMARY_PATH = SUMMARIES_DIR / "build_stock_panel_summary.json"
EXTRACT_OPTION_SUMMARY_PATH = SUMMARIES_DIR / "extract_option_data_summary.json"
BUILD_OPTION_FEATURES_SUMMARY_PATH = SUMMARIES_DIR / "build_option_features_summary.json"
BUILD_SURFACE_FACTORS_SUMMARY_PATH = SUMMARIES_DIR / "build_surface_factors_summary.json"
EXTRACT_CALIBRATED_SURFACE_INPUTS_SUMMARY_PATH = SUMMARIES_DIR / "extract_calibrated_surface_inputs_summary.json"
BUILD_CALIBRATED_SURFACE_SUMMARY_PATH = SUMMARIES_DIR / "build_calibrated_surface_summary.json"
BUILD_TEXT_NEWS_PANEL_SUMMARY_PATH = SUMMARIES_DIR / "build_text_news_panel_summary.json"
SMOKE_SUMMARY_PATH = SUMMARIES_DIR / "smoke_summary.json"

PART1_METRICS_JSON = METRICS_DIR / "part1_metrics.json"
PART1_METRICS_CSV = METRICS_DIR / "part1_metrics.csv"
PART1_PREDICTIONS_CSV = METRICS_DIR / "part1_predictions.csv"

PART2_METRICS_JSON = METRICS_DIR / "part2_metrics.json"
PART2_METRICS_CSV = METRICS_DIR / "part2_metrics.csv"
PART2_PREDICTIONS_CSV = METRICS_DIR / "part2_predictions.csv"
SURFACE_EXTENSION_METRICS_JSON = METRICS_DIR / "surface_extension_metrics.json"
SURFACE_EXTENSION_METRICS_CSV = METRICS_DIR / "surface_extension_metrics.csv"
SURFACE_EXTENSION_PREDICTIONS_CSV = METRICS_DIR / "surface_extension_predictions.csv"
CALIBRATED_SURFACE_EXTENSION_METRICS_JSON = METRICS_DIR / "calibrated_surface_extension_metrics.json"
CALIBRATED_SURFACE_EXTENSION_METRICS_CSV = METRICS_DIR / "calibrated_surface_extension_metrics.csv"
CALIBRATED_SURFACE_EXTENSION_PREDICTIONS_CSV = METRICS_DIR / "calibrated_surface_extension_predictions.csv"
PHASE2_DECISION_METRICS_JSON = METRICS_DIR / "phase2_decision_metrics.json"
PHASE2_DECISION_METRICS_CSV = METRICS_DIR / "phase2_decision_metrics.csv"
PHASE2_DECISION_DAILY_RETURNS_CSV = METRICS_DIR / "phase2_decision_daily_returns.csv"
PHASE2_DECISION_EXTENSION_METRICS_JSON = METRICS_DIR / "phase2_decision_extension_metrics.json"
PHASE2_DECISION_EXTENSION_METRICS_CSV = METRICS_DIR / "phase2_decision_extension_metrics.csv"
PHASE2_BUCKET_METRICS_JSON = METRICS_DIR / "phase2_bucket_analysis_metrics.json"
PHASE2_BUCKET_METRICS_CSV = METRICS_DIR / "phase2_bucket_analysis_metrics.csv"
PHASE2_BUCKET_DIAGNOSTICS_CSV = METRICS_DIR / "phase2_bucket_analysis_diagnostics.csv"
PHASE2_BUCKET_EXTENSION_METRICS_JSON = METRICS_DIR / "phase2_bucket_analysis_extension_metrics.json"
PHASE2_BUCKET_EXTENSION_METRICS_CSV = METRICS_DIR / "phase2_bucket_analysis_extension_metrics.csv"
PHASE2_BUCKET_EXTENSION_DIAGNOSTICS_CSV = METRICS_DIR / "phase2_bucket_analysis_extension_diagnostics.csv"
TEXT_NEWS_EXTENSION_METRICS_JSON = METRICS_DIR / "text_news_extension_metrics.json"
TEXT_NEWS_EXTENSION_METRICS_CSV = METRICS_DIR / "text_news_extension_metrics.csv"
TEXT_NEWS_EXTENSION_PREDICTIONS_CSV = METRICS_DIR / "text_news_extension_predictions.csv"

PART1_PERSISTENCE_FIGURE = FIGURES_DIR / "part1_persistence_test_confusion.png"
PART1_STOCK_LOGREG_FIGURE = FIGURES_DIR / "part1_stock_only_test_confusion.png"
PART2_STOCK_LOGREG_FIGURE = FIGURES_DIR / "part2_stock_only_complete_case_test_confusion.png"
PART2_OPTION_LOGREG_FIGURE = FIGURES_DIR / "part2_option_only_test_confusion.png"
PART2_COMBINED_LOGREG_FIGURE = FIGURES_DIR / "part2_combined_test_confusion.png"
SURFACE_ONLY_FIGURE = FIGURES_DIR / "surface_extension_surface_only_test_confusion.png"
STOCK_SURFACE_FIGURE = FIGURES_DIR / "surface_extension_stock_surface_test_confusion.png"
ALL_FEATURES_FIGURE = FIGURES_DIR / "surface_extension_all_features_test_confusion.png"
CALIBRATED_BETA_ONLY_FIGURE = FIGURES_DIR / "calibrated_surface_extension_beta_only_test_confusion.png"
OPTION_BETA_FIGURE = FIGURES_DIR / "calibrated_surface_extension_option_beta_test_confusion.png"
ALL_EXTENSIONS_FIGURE = FIGURES_DIR / "calibrated_surface_extension_all_extensions_test_confusion.png"
PHASE2_CUMULATIVE_VALUE_FIGURE = FIGURES_DIR / "phase2_cumulative_value.png"
PHASE2_DRAWDOWN_FIGURE = FIGURES_DIR / "phase2_drawdown.png"
PHASE2_EXPOSURE_FIGURE = FIGURES_DIR / "phase2_exposure.png"
PHASE2_BUCKET_FUTURE_RV_FIGURE = FIGURES_DIR / "phase2_bucket_future_rv.png"
PHASE2_BUCKET_HIT_RATE_FIGURE = FIGURES_DIR / "phase2_bucket_hit_rate.png"
PHASE2_BUCKET_RANK_IC_FIGURE = FIGURES_DIR / "phase2_bucket_rank_ic.png"
TEXT_NEWS_CORE_FIGURE = FIGURES_DIR / "text_news_extension_core_test_confusion.png"
TEXT_NEWS_NEWS_ONLY_FIGURE = FIGURES_DIR / "text_news_extension_news_only_test_confusion.png"
TEXT_NEWS_SENTIMENT_FIGURE = FIGURES_DIR / "text_news_extension_sentiment_test_confusion.png"
TEXT_NEWS_CLAIM_SENTIMENT_FIGURE = FIGURES_DIR / "text_news_extension_claim_sentiment_test_confusion.png"
TEXT_NEWS_FINBERT_FIGURE = FIGURES_DIR / "text_news_extension_finbert_test_confusion.png"
TEXT_NEWS_CLAIM_FINBERT_FIGURE = FIGURES_DIR / "text_news_extension_claim_finbert_test_confusion.png"
TEXT_NEWS_EVENT_CORE_FIGURE = FIGURES_DIR / "text_news_event_core_test_confusion.png"
TEXT_NEWS_EVENT_FINBERT_FIGURE = FIGURES_DIR / "text_news_event_finbert_test_confusion.png"
TEXT_NEWS_EVENT_CLAIM_FINBERT_FIGURE = FIGURES_DIR / "text_news_event_claim_finbert_test_confusion.png"

PART1_STOCK_MODEL_PATH = MODELS_DIR / "part1_stock_only_logreg.joblib"
PART2_STOCK_MODEL_PATH = MODELS_DIR / "part2_stock_only_complete_case_logreg.joblib"
PART2_OPTION_MODEL_PATH = MODELS_DIR / "part2_option_only_logreg.joblib"
PART2_COMBINED_MODEL_PATH = MODELS_DIR / "part2_combined_logreg.joblib"
SURFACE_ONLY_MODEL_PATH = MODELS_DIR / "surface_extension_surface_only_logreg.joblib"
STOCK_SURFACE_MODEL_PATH = MODELS_DIR / "surface_extension_stock_surface_logreg.joblib"
ALL_FEATURES_MODEL_PATH = MODELS_DIR / "surface_extension_all_features_logreg.joblib"
STOCK_ONLY_XGB_MODEL_PATH = MODELS_DIR / "surface_extension_stock_only_xgb.joblib"
OPTION_ONLY_XGB_MODEL_PATH = MODELS_DIR / "surface_extension_option_only_xgb.joblib"
ALL_FEATURES_XGB_MODEL_PATH = MODELS_DIR / "surface_extension_all_features_xgb.joblib"
CALIBRATED_BETA_ONLY_MODEL_PATH = MODELS_DIR / "calibrated_surface_extension_beta_only_logreg.joblib"
OPTION_BETA_MODEL_PATH = MODELS_DIR / "calibrated_surface_extension_option_beta_logreg.joblib"
ALL_EXTENSIONS_MODEL_PATH = MODELS_DIR / "calibrated_surface_extension_all_extensions_logreg.joblib"
TEXT_NEWS_NEWS_ONLY_MODEL_PATH = MODELS_DIR / "text_news_extension_news_only_logreg.joblib"
TEXT_NEWS_SENTIMENT_MODEL_PATH = MODELS_DIR / "text_news_extension_all_features_plus_sentiment_logreg.joblib"
TEXT_NEWS_CLAIM_SENTIMENT_MODEL_PATH = MODELS_DIR / "text_news_extension_all_features_plus_claim_sentiment_logreg.joblib"
TEXT_NEWS_FINBERT_MODEL_PATH = MODELS_DIR / "text_news_extension_all_features_plus_finbert_logreg.joblib"
TEXT_NEWS_CLAIM_FINBERT_MODEL_PATH = MODELS_DIR / "text_news_extension_all_features_plus_claim_finbert_logreg.joblib"
TEXT_NEWS_EVENT_FINBERT_MODEL_PATH = MODELS_DIR / "text_news_event_all_features_plus_event_finbert_logreg.joblib"
TEXT_NEWS_EVENT_CLAIM_FINBERT_MODEL_PATH = MODELS_DIR / "text_news_event_all_features_plus_event_claim_finbert_logreg.joblib"

PART1_LABEL_NAMES = ["normal_rv", "high_rv"]


def ensure_project_dirs() -> None:
    for path in (
        DATA_DIR,
        RAW_DATA_DIR,
        RAW_STOCK_DIR,
        RAW_OPTIONS_DIR,
        UNIVERSE_DIR,
        PROCESSED_DIR,
        OUTPUT_DIR,
        SUMMARIES_DIR,
        METRICS_DIR,
        FIGURES_DIR,
        MODELS_DIR,
        LOCAL_CONFIG_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
