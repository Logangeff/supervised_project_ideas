from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from .config import (
    CLASSICAL_C_GRID,
    CLASSICAL_MODEL_FILENAMES,
    CLASSICAL_TRAINING_SUMMARY_PATH,
    TFIDF_MIN_DF,
    ensure_output_dirs,
)
from .data_pipeline import load_project_data, set_global_seed, tokenize


def _build_train_document_frequency(texts: list[str]) -> Counter:
    document_frequency: Counter = Counter()
    for text in texts:
        document_frequency.update(set(tokenize(text)))
    return document_frequency


def _fit_and_score_pipeline(
    train_texts: list[str],
    train_labels: list[str],
    validation_texts: list[str],
    validation_labels: list[str],
    c_value: float,
    vocabulary: list[str] | None = None,
) -> tuple[Pipeline, dict[str, float]]:
    vectorizer_kwargs = {
        "analyzer": tokenize,
        "lowercase": False,
        "token_pattern": None,
        "ngram_range": (1, 1),
    }
    if vocabulary is None:
        vectorizer_kwargs["min_df"] = TFIDF_MIN_DF
    else:
        vectorizer_kwargs["vocabulary"] = vocabulary

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(**vectorizer_kwargs)),
            (
                "classifier",
                LogisticRegression(
                    C=c_value,
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )
    pipeline.fit(train_texts, train_labels)
    predictions = pipeline.predict(validation_texts)
    metrics = {
        "accuracy": float(accuracy_score(validation_labels, predictions)),
        "macro_f1": float(f1_score(validation_labels, predictions, average="macro")),
    }
    return pipeline, metrics


def _train_best_logistic_pipeline(
    train_texts: list[str],
    train_labels: list[str],
    validation_texts: list[str],
    validation_labels: list[str],
    vocabulary: list[str] | None = None,
) -> tuple[Pipeline, dict[str, object]]:
    best_pipeline: Pipeline | None = None
    best_metadata: dict[str, object] | None = None

    for c_value in CLASSICAL_C_GRID:
        pipeline, metrics = _fit_and_score_pipeline(
            train_texts=train_texts,
            train_labels=train_labels,
            validation_texts=validation_texts,
            validation_labels=validation_labels,
            c_value=c_value,
            vocabulary=vocabulary,
        )
        metadata = {
            "selected_c": c_value,
            "validation_accuracy": metrics["accuracy"],
            "validation_macro_f1": metrics["macro_f1"],
            "feature_count": int(len(pipeline.named_steps["tfidf"].vocabulary_)),
        }
        if best_metadata is None or metadata["validation_macro_f1"] > best_metadata["validation_macro_f1"]:
            best_pipeline = pipeline
            best_metadata = metadata

    if best_pipeline is None or best_metadata is None:
        raise RuntimeError("Failed to train a classical baseline.")
    return best_pipeline, best_metadata


def train_classical_models(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    lm_vocabulary: set[str],
    model_filenames: dict[str, Path],
    summary_path: Path,
) -> dict[str, object]:
    train_texts = train_frame["text"].tolist()
    train_labels = train_frame["label"].tolist()
    validation_texts = validation_frame["text"].tolist()
    validation_labels = validation_frame["label"].tolist()

    standard_pipeline, standard_metadata = _train_best_logistic_pipeline(
        train_texts=train_texts,
        train_labels=train_labels,
        validation_texts=validation_texts,
        validation_labels=validation_labels,
    )
    joblib.dump(standard_pipeline, model_filenames["standard_bow"])

    document_frequency = _build_train_document_frequency(train_texts)
    restricted_vocabulary = sorted(
        token
        for token, count in document_frequency.items()
        if count >= TFIDF_MIN_DF and token in lm_vocabulary
    )
    if not restricted_vocabulary:
        raise RuntimeError("LM-restricted vocabulary is empty after applying training-set document frequency filtering.")

    lm_pipeline, lm_metadata = _train_best_logistic_pipeline(
        train_texts=train_texts,
        train_labels=train_labels,
        validation_texts=validation_texts,
        validation_labels=validation_labels,
        vocabulary=restricted_vocabulary,
    )
    joblib.dump(lm_pipeline, model_filenames["lm_restricted_bow"])

    summary = {
        "standard_bow": standard_metadata,
        "lm_restricted_bow": {
            **lm_metadata,
            "restricted_vocabulary_size": len(restricted_vocabulary),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_classical_phase() -> dict[str, object]:
    ensure_output_dirs()
    set_global_seed()

    project_data = load_project_data()
    phrasebank = project_data["phrasebank"]
    lm_vocabulary = project_data["lm_vocabulary"]

    return train_classical_models(
        train_frame=phrasebank["train"],
        validation_frame=phrasebank["validation"],
        lm_vocabulary=lm_vocabulary,
        model_filenames=CLASSICAL_MODEL_FILENAMES,
        summary_path=CLASSICAL_TRAINING_SUMMARY_PATH,
    )
