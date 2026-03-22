from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import (
    CLASSICAL_C_GRID,
    CLASSICAL_MIN_DF,
    FINBERT_MODEL_ID,
    PROJECT_SUMMARY_JSON,
    SEED,
    STAGE1_BEST_DETECTOR_PATH,
    STAGE1_EVALUATION_JSON,
    STAGE2_ALL_TEXT_FIGURE_PATH,
    STAGE2_ALL_TEXT_MODEL_PATH,
    STAGE2_CLAIM_AWARE_FIGURE_PATH,
    STAGE2_CLAIM_AWARE_MODEL_PATH,
    STAGE2_CLAIM_FINBERT_FIGURE_PATH,
    STAGE2_CLAIM_FINBERT_MODEL_PATH,
    STAGE2_CLAIM_SCORES_PATH,
    STAGE2_EVALUATION_CSV,
    STAGE2_EVALUATION_JSON,
    STAGE2_FINBERT_SIGNALS_PATH,
    STAGE2_MATERIAL_TONE_FIGURE_PATH,
    STAGE2_MATERIAL_TONE_MODEL_PATH,
    STAGE2_MARKET_FIGURE_PATH,
    STAGE2_MARKET_MODEL_PATH,
    STAGE2_MARKET_SUMMARY_PATH,
    STAGE2_TEXT_SUMMARY_PATH,
    STAGE2_LABEL_NAMES,
    ensure_project_dirs,
    set_global_seed,
)
from .evaluation import compute_metrics, save_confusion_matrix_figure, write_json, write_rows_csv
from .stage1_models import run_stage1_evaluate, score_texts_with_best_detector
from .stage2_data import load_stage2_dataset, load_stage2_splits
from .text_utils import tokenize
from .transformer_features import score_finbert_sentiment


def _score_tuple(metrics: dict[str, object], model_name: str) -> tuple[float, float, str]:
    return (float(metrics["macro_f1"]), float(metrics["accuracy"]), model_name)


def _build_text_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        tokenizer=tokenize,
        preprocessor=None,
        lowercase=False,
        token_pattern=None,
        ngram_range=(1, 1),
        min_df=CLASSICAL_MIN_DF,
    )


def _ensure_claim_scores(dataset: pd.DataFrame) -> pd.DataFrame:
    if not STAGE2_CLAIM_SCORES_PATH.exists():
        if not STAGE1_BEST_DETECTOR_PATH.exists():
            run_stage1_evaluate()
        probabilities = score_texts_with_best_detector(dataset["title"].tolist())
        claim_scores = pd.DataFrame({"event_id": dataset["event_id"], "claim_prob": probabilities})
        claim_scores.to_parquet(STAGE2_CLAIM_SCORES_PATH, index=False)
    else:
        claim_scores = pd.read_parquet(STAGE2_CLAIM_SCORES_PATH)

    merged = dataset.merge(claim_scores, on="event_id", how="left")
    if merged["claim_prob"].isna().any():
        raise RuntimeError("Missing claim probabilities for some Stage 2 rows.")
    return merged


def _ensure_finbert_signals(dataset: pd.DataFrame) -> pd.DataFrame:
    if not STAGE2_FINBERT_SIGNALS_PATH.exists():
        finbert_signals = score_finbert_sentiment(dataset["title"].tolist())
        finbert_signals.insert(0, "event_id", dataset["event_id"].tolist())
        finbert_signals.to_parquet(STAGE2_FINBERT_SIGNALS_PATH, index=False)
    else:
        finbert_signals = pd.read_parquet(STAGE2_FINBERT_SIGNALS_PATH)

    merged = dataset.merge(finbert_signals, on="event_id", how="left")
    required_columns = [
        "finbert_positive_prob",
        "finbert_negative_prob",
        "finbert_neutral_prob",
        "finbert_sentiment_score",
        "finbert_label",
    ]
    if merged[required_columns].isna().any().any():
        raise RuntimeError("Missing FinBERT signals for some Stage 2 rows.")
    return merged


def _train_market_only_model(splits: dict[str, pd.DataFrame]) -> tuple[dict[str, object], dict[str, object]]:
    X_train = splits["train"][["same_day_return"]].to_numpy()
    y_train = splits["train"]["label_id"].astype(int).tolist()
    X_validation = splits["validation"][["same_day_return"]].to_numpy()
    y_validation = splits["validation"]["label_id"].astype(int).tolist()
    X_test = splits["test"][["same_day_return"]].to_numpy()
    y_test = splits["test"]["label_id"].astype(int).tolist()

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
        metrics = compute_metrics(y_validation, validation_predictions, STAGE2_LABEL_NAMES)
        if best_metrics is None or _score_tuple(metrics, "market_only") > _score_tuple(best_metrics, "market_only"):
            best_pipeline = pipeline
            best_c = c_value
            best_metrics = metrics

    assert best_pipeline is not None and best_c is not None

    model_bundle = {"model_type": "market_only", "pipeline": best_pipeline, "model_name": "market_only"}
    validation_predictions = best_pipeline.predict(X_validation).tolist()
    test_predictions = best_pipeline.predict(X_test).tolist()
    summary = {
        "model": "market_only",
        "best_c": best_c,
        "validation": compute_metrics(y_validation, validation_predictions, STAGE2_LABEL_NAMES),
        "test": compute_metrics(y_test, test_predictions, STAGE2_LABEL_NAMES),
    }
    save_confusion_matrix_figure(
        y_test,
        test_predictions,
        STAGE2_LABEL_NAMES,
        STAGE2_MARKET_FIGURE_PATH,
        "Stage 2 Market-only Baseline (Test)",
    )
    return model_bundle, summary


def _train_scalar_feature_model(
    model_name: str,
    feature_column: str,
    splits: dict[str, pd.DataFrame],
) -> tuple[dict[str, object], dict[str, object]]:
    X_train = splits["train"][[feature_column]].to_numpy()
    y_train = splits["train"]["label_id"].astype(int).tolist()
    X_validation = splits["validation"][[feature_column]].to_numpy()
    y_validation = splits["validation"]["label_id"].astype(int).tolist()
    X_test = splits["test"][[feature_column]].to_numpy()
    y_test = splits["test"]["label_id"].astype(int).tolist()

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
        metrics = compute_metrics(y_validation, validation_predictions, STAGE2_LABEL_NAMES)
        if best_metrics is None or _score_tuple(metrics, model_name) > _score_tuple(best_metrics, model_name):
            best_pipeline = pipeline
            best_c = c_value
            best_metrics = metrics

    assert best_pipeline is not None and best_c is not None

    model_bundle = {
        "model_type": model_name,
        "pipeline": best_pipeline,
        "model_name": model_name,
        "feature_column": feature_column,
    }
    validation_predictions = best_pipeline.predict(X_validation).tolist()
    test_predictions = best_pipeline.predict(X_test).tolist()
    summary = {
        "model": model_name,
        "feature_column": feature_column,
        "best_c": best_c,
        "validation": compute_metrics(y_validation, validation_predictions, STAGE2_LABEL_NAMES),
        "test": compute_metrics(y_test, test_predictions, STAGE2_LABEL_NAMES),
    }
    save_confusion_matrix_figure(
        y_test,
        test_predictions,
        STAGE2_LABEL_NAMES,
        STAGE2_MATERIAL_TONE_FIGURE_PATH,
        "Stage 2 Material-Tone Only Baseline (Test)",
    )
    return model_bundle, summary


def _train_text_model(
    model_name: str,
    splits: dict[str, pd.DataFrame],
    extra_feature_columns: list[str] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    train_texts = splits["train"]["title"].tolist()
    validation_texts = splits["validation"]["title"].tolist()
    test_texts = splits["test"]["title"].tolist()
    y_train = splits["train"]["label_id"].astype(int).tolist()
    y_validation = splits["validation"]["label_id"].astype(int).tolist()
    y_test = splits["test"]["label_id"].astype(int).tolist()

    vectorizer = _build_text_vectorizer()
    X_train_text = vectorizer.fit_transform(train_texts)
    X_validation_text = vectorizer.transform(validation_texts)
    X_test_text = vectorizer.transform(test_texts)

    extra_feature_columns = extra_feature_columns or []
    if extra_feature_columns:
        X_train = sparse.hstack(
            [X_train_text, sparse.csr_matrix(splits["train"][extra_feature_columns].to_numpy())],
            format="csr",
        )
        X_validation = sparse.hstack(
            [X_validation_text, sparse.csr_matrix(splits["validation"][extra_feature_columns].to_numpy())],
            format="csr",
        )
        X_test = sparse.hstack(
            [X_test_text, sparse.csr_matrix(splits["test"][extra_feature_columns].to_numpy())],
            format="csr",
        )
    else:
        X_train, X_validation, X_test = X_train_text, X_validation_text, X_test_text

    best_classifier = None
    best_c = None
    best_metrics = None
    for c_value in CLASSICAL_C_GRID:
        classifier = LogisticRegression(max_iter=2000, C=c_value, random_state=SEED)
        classifier.fit(X_train, y_train)
        validation_predictions = classifier.predict(X_validation).tolist()
        metrics = compute_metrics(y_validation, validation_predictions, STAGE2_LABEL_NAMES)
        if best_metrics is None or _score_tuple(metrics, model_name) > _score_tuple(best_metrics, model_name):
            best_classifier = classifier
            best_c = c_value
            best_metrics = metrics

    assert best_classifier is not None and best_c is not None
    model_bundle = {
        "model_type": model_name,
        "model_name": model_name,
        "vectorizer": vectorizer,
        "classifier": best_classifier,
        "extra_feature_columns": extra_feature_columns,
    }
    validation_predictions = best_classifier.predict(X_validation).tolist()
    test_predictions = best_classifier.predict(X_test).tolist()
    summary = {
        "model": model_name,
        "best_c": best_c,
        "feature_count": int(X_train.shape[1]),
        "extra_feature_columns": extra_feature_columns,
        "validation": compute_metrics(y_validation, validation_predictions, STAGE2_LABEL_NAMES),
        "test": compute_metrics(y_test, test_predictions, STAGE2_LABEL_NAMES),
    }
    figure_paths = {
        "all_text": STAGE2_ALL_TEXT_FIGURE_PATH,
        "claim_aware": STAGE2_CLAIM_AWARE_FIGURE_PATH,
        "claim_finbert_aware": STAGE2_CLAIM_FINBERT_FIGURE_PATH,
    }
    figure_path = figure_paths[model_name]
    save_confusion_matrix_figure(
        y_test,
        test_predictions,
        STAGE2_LABEL_NAMES,
        figure_path,
        f"Stage 2 {model_name.replace('_', ' ').title()} (Test)",
    )
    return model_bundle, summary


def run_stage2_models() -> dict[str, object]:
    ensure_project_dirs()
    set_global_seed(SEED)
    base_dataset = load_stage2_dataset()
    claim_scored_dataset = _ensure_claim_scores(base_dataset)
    sentiment_scored_dataset = _ensure_finbert_signals(claim_scored_dataset)
    sentiment_scored_dataset["material_sentiment_score"] = (
        sentiment_scored_dataset["claim_prob"] * sentiment_scored_dataset["finbert_sentiment_score"]
    )
    split_frames = load_stage2_splits()
    split_frames = {
        split_name: sentiment_scored_dataset[sentiment_scored_dataset["event_id"].isin(split_frame["event_id"])]
        .sort_values(["publication_timestamp", "event_id"])
        .reset_index(drop=True)
        for split_name, split_frame in split_frames.items()
    }

    market_model, market_summary = _train_market_only_model(split_frames)
    joblib.dump(market_model, STAGE2_MARKET_MODEL_PATH)
    write_json(STAGE2_MARKET_SUMMARY_PATH, market_summary)

    material_tone_model, material_tone_summary = _train_scalar_feature_model(
        "material_tone_only",
        "material_sentiment_score",
        split_frames,
    )
    joblib.dump(material_tone_model, STAGE2_MATERIAL_TONE_MODEL_PATH)

    all_text_model, all_text_summary = _train_text_model("all_text", split_frames, extra_feature_columns=[])
    claim_aware_model, claim_aware_summary = _train_text_model(
        "claim_aware",
        split_frames,
        extra_feature_columns=["claim_prob"],
    )
    claim_finbert_model, claim_finbert_summary = _train_text_model(
        "claim_finbert_aware",
        split_frames,
        extra_feature_columns=[
            "claim_prob",
            "finbert_positive_prob",
            "finbert_negative_prob",
            "finbert_neutral_prob",
            "finbert_sentiment_score",
            "material_sentiment_score",
        ],
    )
    joblib.dump(all_text_model, STAGE2_ALL_TEXT_MODEL_PATH)
    joblib.dump(claim_aware_model, STAGE2_CLAIM_AWARE_MODEL_PATH)
    joblib.dump(claim_finbert_model, STAGE2_CLAIM_FINBERT_MODEL_PATH)
    write_json(
        STAGE2_TEXT_SUMMARY_PATH,
        {
            "material_tone_only": material_tone_summary,
            "all_text": all_text_summary,
            "claim_aware": claim_aware_summary,
            "claim_finbert_aware": claim_finbert_summary,
            "sentiment_model_id": FINBERT_MODEL_ID,
        },
    )
    return {
        "market_only": market_summary,
        "material_tone_only": material_tone_summary,
        "all_text": all_text_summary,
        "claim_aware": claim_aware_summary,
        "claim_finbert_aware": claim_finbert_summary,
    }


def _predict_stage2_model(model_bundle: dict[str, object], frame: pd.DataFrame) -> list[int]:
    if model_bundle["model_type"] == "market_only":
        return model_bundle["pipeline"].predict(frame[["same_day_return"]].to_numpy()).tolist()
    if model_bundle["model_type"] == "material_tone_only":
        return model_bundle["pipeline"].predict(frame[[model_bundle["feature_column"]]].to_numpy()).tolist()

    X_text = model_bundle["vectorizer"].transform(frame["title"].tolist())
    extra_feature_columns = model_bundle.get("extra_feature_columns", [])
    if extra_feature_columns:
        X = sparse.hstack([X_text, sparse.csr_matrix(frame[extra_feature_columns].to_numpy())], format="csr")
    else:
        X = X_text
    return model_bundle["classifier"].predict(X).tolist()


def run_stage2_evaluate() -> dict[str, object]:
    ensure_project_dirs()
    set_global_seed(SEED)
    if (
        not STAGE2_MARKET_MODEL_PATH.exists()
        or not STAGE2_MATERIAL_TONE_MODEL_PATH.exists()
        or not STAGE2_ALL_TEXT_MODEL_PATH.exists()
        or not STAGE2_CLAIM_AWARE_MODEL_PATH.exists()
        or not STAGE2_CLAIM_FINBERT_MODEL_PATH.exists()
    ):
        run_stage2_models()

    base_dataset = load_stage2_dataset()
    dataset = _ensure_finbert_signals(_ensure_claim_scores(base_dataset))
    dataset["material_sentiment_score"] = dataset["claim_prob"] * dataset["finbert_sentiment_score"]
    split_frames = load_stage2_splits()
    split_frames = {
        split_name: dataset[dataset["event_id"].isin(split_frame["event_id"])]
        .sort_values(["publication_timestamp", "event_id"])
        .reset_index(drop=True)
        for split_name, split_frame in split_frames.items()
    }

    models = {
        "market_only": joblib.load(STAGE2_MARKET_MODEL_PATH),
        "material_tone_only": joblib.load(STAGE2_MATERIAL_TONE_MODEL_PATH),
        "all_text": joblib.load(STAGE2_ALL_TEXT_MODEL_PATH),
        "claim_aware": joblib.load(STAGE2_CLAIM_AWARE_MODEL_PATH),
        "claim_finbert_aware": joblib.load(STAGE2_CLAIM_FINBERT_MODEL_PATH),
    }

    rows: list[dict[str, object]] = []
    for model_name, model_bundle in models.items():
        for split_name, frame in split_frames.items():
            predictions = _predict_stage2_model(model_bundle, frame)
            metrics = compute_metrics(frame["label_id"].astype(int).tolist(), predictions, STAGE2_LABEL_NAMES)
            rows.append(
                {
                    "model": model_name,
                    "split": split_name,
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                }
            )

    write_json(STAGE2_EVALUATION_JSON, {"rows": rows})
    write_rows_csv(STAGE2_EVALUATION_CSV, rows)

    if not STAGE1_EVALUATION_JSON.exists():
        run_stage1_evaluate()
    stage1_summary = json.loads(STAGE1_EVALUATION_JSON.read_text(encoding="utf-8"))
    best_detector = json.loads(STAGE1_BEST_DETECTOR_PATH.read_text(encoding="utf-8"))
    project_summary = {
        "stage1_best_detector": best_detector,
        "stage1_rows": stage1_summary["rows"],
        "stage2_rows": rows,
    }
    write_json(PROJECT_SUMMARY_JSON, project_summary)
    return project_summary
