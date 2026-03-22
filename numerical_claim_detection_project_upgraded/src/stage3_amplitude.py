from __future__ import annotations

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import (
    CLASSICAL_C_GRID,
    SEED,
    STAGE2_FINBERT_SIGNALS_PATH,
    STAGE3_AMPLITUDE_CSV,
    STAGE3_AMPLITUDE_SUMMARY_PATH,
    STAGE3_LABEL_NAMES,
    STAGE3_LARGE_MOVE_QUANTILE,
    STAGE3_MARKET_ONLY_FIGURE_PATH,
    STAGE3_MARKET_ONLY_MODEL_PATH,
    STAGE3_STRUCTURED_LOGREG_FIGURE_PATH,
    STAGE3_STRUCTURED_LOGREG_MODEL_PATH,
    STAGE3_STRUCTURED_MLP_FIGURE_PATH,
    STAGE3_STRUCTURED_MLP_MODEL_PATH,
    ensure_project_dirs,
    set_global_seed,
)
from .evaluation import compute_metrics, save_confusion_matrix_figure, write_json, write_rows_csv
from .stage2_data import load_stage2_dataset, load_stage2_splits, run_stage2_data
from .stage2_models import _ensure_claim_scores, _ensure_finbert_signals


MARKET_FEATURES = ["same_day_return", "abs_same_day_return"]
STRUCTURED_FEATURES = [
    "same_day_return",
    "abs_same_day_return",
    "claim_prob",
    "finbert_positive_prob",
    "finbert_negative_prob",
    "finbert_neutral_prob",
    "finbert_sentiment_score",
    "material_sentiment_score",
]


def _score_tuple(metrics: dict[str, object], model_name: str) -> tuple[float, float, str]:
    return (float(metrics["macro_f1"]), float(metrics["accuracy"]), model_name)


def _ensure_enriched_stage2_dataset() -> pd.DataFrame:
    base_dataset = load_stage2_dataset()
    enriched = _ensure_finbert_signals(_ensure_claim_scores(base_dataset))
    enriched["material_sentiment_score"] = enriched["claim_prob"] * enriched["finbert_sentiment_score"]
    return enriched


def _build_amplitude_splits() -> tuple[dict[str, pd.DataFrame], float]:
    if not STAGE2_FINBERT_SIGNALS_PATH.exists():
        run_stage2_data()

    enriched = _ensure_enriched_stage2_dataset()
    split_frames = load_stage2_splits()
    split_frames = {
        split_name: enriched[enriched["event_id"].isin(split_frame["event_id"])]
        .sort_values(["publication_timestamp", "event_id"])
        .reset_index(drop=True)
        for split_name, split_frame in split_frames.items()
    }

    threshold = float(split_frames["train"]["abs_next_day_return"].quantile(STAGE3_LARGE_MOVE_QUANTILE))
    for frame in split_frames.values():
        frame["amp_label_id"] = (frame["abs_next_day_return"] >= threshold).astype(int)
        frame["amp_label_name"] = frame["amp_label_id"].map({0: STAGE3_LABEL_NAMES[0], 1: STAGE3_LABEL_NAMES[1]})
    return split_frames, threshold


def _train_logreg_model(model_name: str, feature_columns: list[str], splits: dict[str, pd.DataFrame]):
    X_train = splits["train"][feature_columns].to_numpy()
    y_train = splits["train"]["amp_label_id"].astype(int).tolist()
    X_validation = splits["validation"][feature_columns].to_numpy()
    y_validation = splits["validation"]["amp_label_id"].astype(int).tolist()
    X_test = splits["test"][feature_columns].to_numpy()
    y_test = splits["test"]["amp_label_id"].astype(int).tolist()

    best_pipeline = None
    best_c = None
    best_metrics = None
    for c_value in CLASSICAL_C_GRID:
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=2000, C=c_value, random_state=SEED)),
            ]
        )
        pipeline.fit(X_train, y_train)
        validation_predictions = pipeline.predict(X_validation).tolist()
        metrics = compute_metrics(y_validation, validation_predictions, STAGE3_LABEL_NAMES)
        if best_metrics is None or _score_tuple(metrics, model_name) > _score_tuple(best_metrics, model_name):
            best_pipeline = pipeline
            best_c = c_value
            best_metrics = metrics

    assert best_pipeline is not None and best_c is not None
    validation_predictions = best_pipeline.predict(X_validation).tolist()
    test_predictions = best_pipeline.predict(X_test).tolist()
    summary = {
        "model": model_name,
        "feature_columns": feature_columns,
        "best_c": best_c,
        "validation": compute_metrics(y_validation, validation_predictions, STAGE3_LABEL_NAMES),
        "test": compute_metrics(y_test, test_predictions, STAGE3_LABEL_NAMES),
    }
    return best_pipeline, summary, y_test, test_predictions


def _train_mlp_model(feature_columns: list[str], splits: dict[str, pd.DataFrame]):
    X_train = splits["train"][feature_columns].to_numpy()
    y_train = splits["train"]["amp_label_id"].astype(int).tolist()
    X_validation = splits["validation"][feature_columns].to_numpy()
    y_validation = splits["validation"]["amp_label_id"].astype(int).tolist()
    X_test = splits["test"][feature_columns].to_numpy()
    y_test = splits["test"]["amp_label_id"].astype(int).tolist()

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                MLPClassifier(
                    hidden_layer_sizes=(16,),
                    random_state=SEED,
                    max_iter=300,
                    early_stopping=True,
                    n_iter_no_change=10,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    validation_predictions = pipeline.predict(X_validation).tolist()
    test_predictions = pipeline.predict(X_test).tolist()
    summary = {
        "model": "structured_mlp",
        "feature_columns": feature_columns,
        "validation": compute_metrics(y_validation, validation_predictions, STAGE3_LABEL_NAMES),
        "test": compute_metrics(y_test, test_predictions, STAGE3_LABEL_NAMES),
    }
    return pipeline, summary, y_test, test_predictions


def run_stage3_amplitude() -> dict[str, object]:
    ensure_project_dirs()
    set_global_seed(SEED)
    split_frames, threshold = _build_amplitude_splits()

    market_model, market_summary, market_y_test, market_test_preds = _train_logreg_model(
        "market_only_amp",
        MARKET_FEATURES,
        split_frames,
    )
    structured_logreg_model, structured_logreg_summary, logreg_y_test, logreg_test_preds = _train_logreg_model(
        "structured_logreg",
        STRUCTURED_FEATURES,
        split_frames,
    )
    structured_mlp_model, structured_mlp_summary, mlp_y_test, mlp_test_preds = _train_mlp_model(
        STRUCTURED_FEATURES,
        split_frames,
    )

    joblib.dump({"model_name": "market_only_amp", "pipeline": market_model, "feature_columns": MARKET_FEATURES}, STAGE3_MARKET_ONLY_MODEL_PATH)
    joblib.dump({"model_name": "structured_logreg", "pipeline": structured_logreg_model, "feature_columns": STRUCTURED_FEATURES}, STAGE3_STRUCTURED_LOGREG_MODEL_PATH)
    joblib.dump({"model_name": "structured_mlp", "pipeline": structured_mlp_model, "feature_columns": STRUCTURED_FEATURES}, STAGE3_STRUCTURED_MLP_MODEL_PATH)

    save_confusion_matrix_figure(
        market_y_test,
        market_test_preds,
        STAGE3_LABEL_NAMES,
        STAGE3_MARKET_ONLY_FIGURE_PATH,
        "Stage 3 Market-only Amplitude Baseline (Test)",
    )
    save_confusion_matrix_figure(
        logreg_y_test,
        logreg_test_preds,
        STAGE3_LABEL_NAMES,
        STAGE3_STRUCTURED_LOGREG_FIGURE_PATH,
        "Stage 3 Structured Logistic Regression (Test)",
    )
    save_confusion_matrix_figure(
        mlp_y_test,
        mlp_test_preds,
        STAGE3_LABEL_NAMES,
        STAGE3_STRUCTURED_MLP_FIGURE_PATH,
        "Stage 3 Structured MLP (Test)",
    )

    rows: list[dict[str, object]] = []
    for summary in (market_summary, structured_logreg_summary, structured_mlp_summary):
        for split_name in ("validation", "test"):
            rows.append(
                {
                    "model": summary["model"],
                    "split": split_name,
                    "accuracy": summary[split_name]["accuracy"],
                    "macro_f1": summary[split_name]["macro_f1"],
                }
            )
    write_rows_csv(STAGE3_AMPLITUDE_CSV, rows)

    amplitude_summary = {
        "target": {
            "name": "large_move",
            "definition": "abs(next_day_return) >= train_quantile_threshold",
            "train_quantile": STAGE3_LARGE_MOVE_QUANTILE,
            "threshold": threshold,
        },
        "split_label_counts": {
            split_name: {
                STAGE3_LABEL_NAMES[0]: int((frame["amp_label_id"] == 0).sum()),
                STAGE3_LABEL_NAMES[1]: int((frame["amp_label_id"] == 1).sum()),
            }
            for split_name, frame in split_frames.items()
        },
        "models": {
            "market_only_amp": market_summary,
            "structured_logreg": structured_logreg_summary,
            "structured_mlp": structured_mlp_summary,
        },
        "rows": rows,
    }
    write_json(STAGE3_AMPLITUDE_SUMMARY_PATH, amplitude_summary)
    return amplitude_summary
