from __future__ import annotations

import random
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_PRICES_DIR = RAW_DATA_DIR / "prices"
RAW_NEWS_DIR = RAW_DATA_DIR / "news"

OUTPUT_DIR = ROOT_DIR / "outputs"
SUMMARIES_DIR = OUTPUT_DIR / "summaries"
METRICS_DIR = OUTPUT_DIR / "metrics"
FIGURES_DIR = OUTPUT_DIR / "figures"
MODELS_DIR = OUTPUT_DIR / "models"
INTERMEDIATE_DIR = OUTPUT_DIR / "intermediate"

SEED = 42
TICKER = "TSLA"
COMPANY_NAME = "Tesla"
TIMEZONE_NAME = "America/New_York"
NEWS_LOCALE_PARAMS = {"hl": "en-US", "gl": "US", "ceid": "US:en"}
NEWS_QUERY = "Tesla OR TSLA"
NEWS_MONTH_SLEEP_SECONDS = 0.2
YAHOO_PRICE_URL = "https://query1.finance.yahoo.com/v8/finance/chart/TSLA"
GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search"
HISTORY_RANGE = "2y"
PRICE_INTERVAL = "1d"

DIRECTION_LABEL_NAMES = ["down", "up"]
AMPLITUDE_LABEL_NAMES = ["normal_move", "large_move"]
STAGE2_TRAIN_RATIO = 0.70
STAGE2_VALIDATION_RATIO = 0.15
STAGE2_TEST_RATIO = 0.15
CLASSICAL_C_GRID = [0.1, 1.0, 10.0]
CLASSICAL_MIN_DF = 2
AMPLITUDE_QUANTILE = 0.80

FINBERT_MODEL_ID = "ProsusAI/finbert"
FINBERT_MAX_LENGTH = 128
FINBERT_BATCH_SIZE = 64

LOCKED_NUMCLAIM_PROJECT = ROOT_DIR.parent / "numerical_claim_detection_project"
LOCKED_CLAIM_DETECTOR_META = LOCKED_NUMCLAIM_PROJECT / "outputs" / "models" / "stage1_best_detector.json"

RAW_PRICE_JSON = RAW_PRICES_DIR / "tsla_chart_raw.json"
PRICE_HISTORY_PARQUET = RAW_PRICES_DIR / "tsla_price_history.parquet"
PRICE_HISTORY_CSV = RAW_PRICES_DIR / "tsla_price_history.csv"
RAW_NEWS_PARQUET = RAW_NEWS_DIR / "tsla_headlines_raw.parquet"
RAW_NEWS_CSV = RAW_NEWS_DIR / "tsla_headlines_raw.csv"

DAILY_DATASET_PARQUET = INTERMEDIATE_DIR / "tsla_daily_dataset.parquet"
HEADLINE_CLAIM_SCORES_PARQUET = INTERMEDIATE_DIR / "headline_claim_scores.parquet"
HEADLINE_FINBERT_SCORES_PARQUET = INTERMEDIATE_DIR / "headline_finbert_scores.parquet"
ENRICHED_DAILY_DATASET_PARQUET = INTERMEDIATE_DIR / "tsla_daily_dataset_enriched.parquet"

COLLECTION_SUMMARY_PATH = SUMMARIES_DIR / "collection_summary.json"
DATASET_SUMMARY_PATH = SUMMARIES_DIR / "daily_dataset_summary.json"
MATERIALITY_SUMMARY_PATH = SUMMARIES_DIR / "stage1_materiality_summary.json"

STAGE2_DIRECTION_SUMMARY_PATH = METRICS_DIR / "stage2_direction_summary.json"
STAGE2_DIRECTION_CSV = METRICS_DIR / "stage2_direction_summary.csv"
STAGE2_DIRECTION_PREDICTIONS_CSV = METRICS_DIR / "stage2_direction_predictions.csv"

STAGE3_SENTIMENT_SUMMARY_PATH = METRICS_DIR / "stage3_sentiment_summary.json"
STAGE3_SENTIMENT_CSV = METRICS_DIR / "stage3_sentiment_summary.csv"
STAGE3_SENTIMENT_PREDICTIONS_CSV = METRICS_DIR / "stage3_sentiment_predictions.csv"

AMPLITUDE_SUMMARY_PATH = METRICS_DIR / "stage3_amplitude_summary.json"
AMPLITUDE_CSV = METRICS_DIR / "stage3_amplitude_summary.csv"
AMPLITUDE_PREDICTIONS_CSV = METRICS_DIR / "stage3_amplitude_predictions.csv"

STAGE2_MARKET_MODEL_PATH = MODELS_DIR / "stage2_market_only.joblib"
STAGE2_FULL_TEXT_MODEL_PATH = MODELS_DIR / "stage2_full_text.joblib"
STAGE2_CLAIM_AWARE_MODEL_PATH = MODELS_DIR / "stage2_claim_aware.joblib"
STAGE3_CLAIM_SENTIMENT_MODEL_PATH = MODELS_DIR / "stage3_claim_sentiment_aware.joblib"
AMPLITUDE_MARKET_MODEL_PATH = MODELS_DIR / "stage3_market_only_amp.joblib"
AMPLITUDE_STRUCTURED_MODEL_PATH = MODELS_DIR / "stage3_claim_sentiment_structured_amp.joblib"

STAGE2_MARKET_FIGURE_PATH = FIGURES_DIR / "stage2_market_only_confusion_matrix.png"
STAGE2_FULL_TEXT_FIGURE_PATH = FIGURES_DIR / "stage2_full_text_confusion_matrix.png"
STAGE2_CLAIM_AWARE_FIGURE_PATH = FIGURES_DIR / "stage2_claim_aware_confusion_matrix.png"
STAGE3_SENTIMENT_FIGURE_PATH = FIGURES_DIR / "stage3_claim_sentiment_aware_confusion_matrix.png"
AMPLITUDE_MARKET_FIGURE_PATH = FIGURES_DIR / "stage3_market_only_amp_confusion_matrix.png"
AMPLITUDE_STRUCTURED_FIGURE_PATH = FIGURES_DIR / "stage3_claim_sentiment_structured_amp_confusion_matrix.png"


def ensure_project_dirs() -> None:
    for path in (
        DATA_DIR,
        RAW_DATA_DIR,
        RAW_PRICES_DIR,
        RAW_NEWS_DIR,
        OUTPUT_DIR,
        SUMMARIES_DIR,
        METRICS_DIR,
        FIGURES_DIR,
        MODELS_DIR,
        INTERMEDIATE_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
