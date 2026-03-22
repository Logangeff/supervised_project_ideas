from __future__ import annotations

import random
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
NUMCLAIM_RAW_DIR = RAW_DATA_DIR / "numclaim"
NUMCLAIM_CACHE_DIR = NUMCLAIM_RAW_DIR / "hf_cache"
FINANCIAL_NEWS_RAW_DIR = RAW_DATA_DIR / "financial_news"

OUTPUT_DIR = ROOT_DIR / "outputs"
SPLITS_DIR = OUTPUT_DIR / "splits"
SUMMARIES_DIR = OUTPUT_DIR / "summaries"
METRICS_DIR = OUTPUT_DIR / "metrics"
FIGURES_DIR = OUTPUT_DIR / "figures"
MODELS_DIR = OUTPUT_DIR / "models"
INTERMEDIATE_DIR = OUTPUT_DIR / "intermediate"

SEED = 42

STAGE1_LABEL_TO_ID = {"OUTOFCLAIM": 0, "INCLAIM": 1}
STAGE1_ID_TO_LABEL = {value: key for key, value in STAGE1_LABEL_TO_ID.items()}
STAGE1_LABEL_NAMES = [STAGE1_ID_TO_LABEL[idx] for idx in sorted(STAGE1_ID_TO_LABEL)]
STAGE2_LABEL_NAMES = ["down", "up"]

NUMCLAIM_DATASET_ID = "gtfintechlab/Numclaim"
FINANCIAL_NEWS_DATASET_ID = "luckycat37/financial-news-dataset"

STAGE1_VALIDATION_SIZE = 0.15
STAGE2_TRAIN_RATIO = 0.70
STAGE2_VALIDATION_RATIO = 0.15
STAGE2_TEST_RATIO = 0.15

CLASSICAL_C_GRID = [0.1, 1.0, 10.0]
CLASSICAL_MIN_DF = 2

STAGE1_CLASSICAL_MODEL_PATH = MODELS_DIR / "stage1_tfidf_logreg.joblib"
STAGE1_GRU_MODEL_PATH = MODELS_DIR / "stage1_gru.pt"
STAGE1_BEST_DETECTOR_PATH = MODELS_DIR / "stage1_best_detector.json"

STAGE2_MARKET_MODEL_PATH = MODELS_DIR / "stage2_market_only.joblib"
STAGE2_ALL_TEXT_MODEL_PATH = MODELS_DIR / "stage2_all_text.joblib"
STAGE2_CLAIM_AWARE_MODEL_PATH = MODELS_DIR / "stage2_claim_aware.joblib"

STAGE1_DATA_SUMMARY_PATH = SUMMARIES_DIR / "stage1_data_summary.json"
STAGE2_DATA_SUMMARY_PATH = SUMMARIES_DIR / "stage2_data_summary.json"

STAGE1_CLASSICAL_SUMMARY_PATH = METRICS_DIR / "stage1_classical_summary.json"
STAGE1_NEURAL_SUMMARY_PATH = METRICS_DIR / "stage1_neural_summary.json"
STAGE1_EVALUATION_JSON = METRICS_DIR / "stage1_evaluation_summary.json"
STAGE1_EVALUATION_CSV = METRICS_DIR / "stage1_evaluation_summary.csv"

STAGE2_MARKET_SUMMARY_PATH = METRICS_DIR / "stage2_market_only_summary.json"
STAGE2_TEXT_SUMMARY_PATH = METRICS_DIR / "stage2_text_models_summary.json"
STAGE2_EVALUATION_JSON = METRICS_DIR / "stage2_evaluation_summary.json"
STAGE2_EVALUATION_CSV = METRICS_DIR / "stage2_evaluation_summary.csv"
PROJECT_SUMMARY_JSON = METRICS_DIR / "project_summary.json"

STAGE1_CLASSICAL_FIGURE_PATH = FIGURES_DIR / "stage1_classical_confusion_matrix.png"
STAGE1_GRU_FIGURE_PATH = FIGURES_DIR / "stage1_gru_confusion_matrix.png"
STAGE2_MARKET_FIGURE_PATH = FIGURES_DIR / "stage2_market_only_confusion_matrix.png"
STAGE2_ALL_TEXT_FIGURE_PATH = FIGURES_DIR / "stage2_all_text_confusion_matrix.png"
STAGE2_CLAIM_AWARE_FIGURE_PATH = FIGURES_DIR / "stage2_claim_aware_confusion_matrix.png"

STAGE2_FILTERED_DATA_PATH = INTERMEDIATE_DIR / "stage2_filtered_dataset.parquet"
STAGE2_CLAIM_SCORES_PATH = INTERMEDIATE_DIR / "stage2_claim_scores.parquet"

STAGE1_TRAIN_SPLIT_PATH = SPLITS_DIR / "stage1_train_indices.csv"
STAGE1_VALIDATION_SPLIT_PATH = SPLITS_DIR / "stage1_validation_indices.csv"
STAGE1_TEST_SPLIT_PATH = SPLITS_DIR / "stage1_test_indices.csv"

STAGE2_TRAIN_SPLIT_PATH = SPLITS_DIR / "stage2_train_indices.csv"
STAGE2_VALIDATION_SPLIT_PATH = SPLITS_DIR / "stage2_validation_indices.csv"
STAGE2_TEST_SPLIT_PATH = SPLITS_DIR / "stage2_test_indices.csv"

GRU_MAX_LENGTH = 64
GRU_EMBED_DIM = 100
GRU_HIDDEN_SIZE = 128
GRU_DROPOUT = 0.2
GRU_LEARNING_RATE = 1e-3
GRU_BATCH_SIZE = 64
GRU_MAX_EPOCHS = 20
GRU_PATIENCE = 3


def ensure_project_dirs() -> None:
    for path in (
        DATA_DIR,
        RAW_DATA_DIR,
        NUMCLAIM_RAW_DIR,
        FINANCIAL_NEWS_RAW_DIR,
        OUTPUT_DIR,
        SPLITS_DIR,
        SUMMARIES_DIR,
        METRICS_DIR,
        FIGURES_DIR,
        MODELS_DIR,
        INTERMEDIATE_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def relative_to_root(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT_DIR.resolve()))
    except ValueError:
        return str(path)


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
