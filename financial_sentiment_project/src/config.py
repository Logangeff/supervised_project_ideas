from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"
SPLITS_DIR = OUTPUT_DIR / "splits"
SUMMARIES_DIR = OUTPUT_DIR / "summaries"
METRICS_DIR = OUTPUT_DIR / "metrics"
FIGURES_DIR = OUTPUT_DIR / "figures"
MODELS_DIR = OUTPUT_DIR / "models"

SEED = 42
LABELS = ["negative", "neutral", "positive"]

PHRASEBANK_REPO_ID = "takala/financial_phrasebank"
PHRASEBANK_ZIP_FILENAME = "data/FinancialPhraseBank-v1.0.zip"
PHRASEBANK_ZIP_MEMBER = "FinancialPhraseBank-v1.0/Sentences_50Agree.txt"

FIQA_DIR = DATA_DIR / "fiqa-sentiment-classification" / "data"
LM_CSV_PATH = DATA_DIR / "loughran_mcdonald" / "Loughran-McDonald_MasterDictionary_1993-2024.csv"

FIQA_NEGATIVE_THRESHOLD = -0.1
FIQA_POSITIVE_THRESHOLD = 0.1

LM_CATEGORY_COLUMNS = [
    "Negative",
    "Positive",
    "Uncertainty",
    "Litigious",
    "Strong_Modal",
    "Weak_Modal",
    "Constraining",
]

PHASE1_SUMMARY_PATH = SUMMARIES_DIR / "phase1_summary.json"
NOSIBLE_SUMMARY_PATH = SUMMARIES_DIR / "nosible_summary.json"

CLASSICAL_MODEL_FILENAMES = {
    "standard_bow": MODELS_DIR / "standard_bow.joblib",
    "lm_restricted_bow": MODELS_DIR / "lm_restricted_bow.joblib",
}
CLASSICAL_TRAINING_SUMMARY_PATH = METRICS_DIR / "classical_training_summary.json"
NOSIBLE_CLASSICAL_MODEL_FILENAMES = {
    "standard_bow": MODELS_DIR / "nosible_standard_bow.joblib",
    "lm_restricted_bow": MODELS_DIR / "nosible_lm_restricted_bow.joblib",
}
NOSIBLE_CLASSICAL_TRAINING_SUMMARY_PATH = METRICS_DIR / "nosible_classical_training_summary.json"

CLASSICAL_C_GRID = [0.1, 1.0, 10.0]
TFIDF_MIN_DF = 2

GRU_MODEL_PATH = MODELS_DIR / "gru_classifier.pt"
GRU_TRAINING_SUMMARY_PATH = METRICS_DIR / "gru_training_summary.json"
NOSIBLE_GRU_MODEL_PATH = MODELS_DIR / "nosible_gru_classifier.pt"
NOSIBLE_GRU_TRAINING_SUMMARY_PATH = METRICS_DIR / "nosible_gru_training_summary.json"
GRU_MAX_LENGTH = 50
GRU_EMBED_DIM = 100
GRU_HIDDEN_SIZE = 128
GRU_DROPOUT = 0.2
GRU_LEARNING_RATE = 1e-3
GRU_BATCH_SIZE = 64
GRU_MAX_EPOCHS = 20
GRU_PATIENCE = 3

EVALUATION_SUMMARY_JSON = METRICS_DIR / "evaluation_summary.json"
EVALUATION_SUMMARY_CSV = METRICS_DIR / "evaluation_summary.csv"
NOSIBLE_EVALUATION_SUMMARY_JSON = METRICS_DIR / "nosible_evaluation_summary.json"
NOSIBLE_EVALUATION_SUMMARY_CSV = METRICS_DIR / "nosible_evaluation_summary.csv"

NOSIBLE_PARQUET_PATH = DATA_DIR / "financial-sentiment" / "data.parquet"

FINBERT_MODEL_ID = "ProsusAI/finbert"
RAW_FINBERT_SUMMARY_PATH = METRICS_DIR / "raw_finbert_summary.json"
FINBERT_FINETUNED_DIR = MODELS_DIR / "finbert_finetuned"
FINBERT_FINETUNED_SUMMARY_PATH = METRICS_DIR / "finbert_finetuned_training_summary.json"
FINBERT_MAX_LENGTH = 128
FINBERT_EVAL_BATCH_SIZE = 32
FINBERT_TRAIN_BATCH_SIZE = 16
FINBERT_LEARNING_RATE = 2e-5
FINBERT_NUM_EPOCHS = 5
FINBERT_WEIGHT_DECAY = 0.01
FINBERT_EARLY_STOPPING_PATIENCE = 2


def ensure_output_dirs() -> None:
    for path in (OUTPUT_DIR, SPLITS_DIR, SUMMARIES_DIR, METRICS_DIR, FIGURES_DIR, MODELS_DIR):
        path.mkdir(parents=True, exist_ok=True)
