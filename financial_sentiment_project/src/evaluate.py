from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from .config import (
    CLASSICAL_MODEL_FILENAMES,
    EVALUATION_SUMMARY_CSV,
    EVALUATION_SUMMARY_JSON,
    FINBERT_FINETUNED_DIR,
    FIGURES_DIR,
    GRU_MODEL_PATH,
    LABELS,
    RAW_FINBERT_SUMMARY_PATH,
    ensure_output_dirs,
)
from .data_pipeline import load_project_data
from .neural_model import load_gru_checkpoint, predict_gru_dataframe


def compute_classification_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, object]:
    matrix = confusion_matrix(y_true, y_pred, labels=LABELS)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix": matrix.tolist(),
        "labels": LABELS,
    }


def save_confusion_matrix_figure(matrix: list[list[int]], title: str, output_path) -> None:
    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(LABELS)))
    axis.set_yticks(range(len(LABELS)))
    axis.set_xticklabels(LABELS, rotation=45, ha="right")
    axis.set_yticklabels(LABELS)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title(title)

    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            axis.text(col_index, row_index, str(value), ha="center", va="center", color="black")

    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def add_model_predictions_to_summary(
    evaluation_summary: dict[str, dict[str, object]],
    model_name: str,
    datasets: dict[str, pd.DataFrame],
    predictions_by_split: dict[str, list[str]],
    figure_prefix: str = "",
) -> dict[str, dict[str, object]]:
    evaluation_summary[model_name] = {}
    for split_name, frame in datasets.items():
        metrics = compute_classification_metrics(frame["label"].tolist(), predictions_by_split[split_name])
        evaluation_summary[model_name][split_name] = metrics
        save_confusion_matrix_figure(
            metrics["confusion_matrix"],
            title=f"{model_name} - {split_name}",
            output_path=FIGURES_DIR / f"{figure_prefix}{model_name}_{split_name}_confusion_matrix.png",
        )
    return evaluation_summary


def save_evaluation_summary(
    evaluation_summary: dict[str, dict[str, object]],
    output_json: Path,
    output_csv: Path,
) -> dict[str, object]:
    flat_rows: list[dict[str, object]] = []
    for model_name, split_results in evaluation_summary.items():
        for split_name, metrics in split_results.items():
            flat_rows.append(
                {
                    "model": model_name,
                    "split": split_name,
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                }
            )

    output_json.write_text(json.dumps(evaluation_summary, indent=2), encoding="utf-8")
    pd.DataFrame(flat_rows).to_csv(output_csv, index=False)
    return evaluation_summary


def evaluate_saved_models(
    model_filenames: dict[str, Path],
    gru_model_path: Path,
    datasets: dict[str, pd.DataFrame],
    output_json: Path,
    output_csv: Path,
    figure_prefix: str = "",
) -> dict[str, object]:
    evaluation_summary: dict[str, dict[str, object]] = {}

    for model_name, model_path in model_filenames.items():
        if not model_path.exists():
            raise FileNotFoundError(f"Missing classical model artifact: {model_path}")
        model = joblib.load(model_path)
        evaluation_summary[model_name] = {}
        for split_name, frame in datasets.items():
            predictions = model.predict(frame["text"].tolist()).tolist()
            metrics = compute_classification_metrics(frame["label"].tolist(), predictions)
            evaluation_summary[model_name][split_name] = metrics
            save_confusion_matrix_figure(
                metrics["confusion_matrix"],
                title=f"{model_name} - {split_name}",
                output_path=FIGURES_DIR / f"{figure_prefix}{model_name}_{split_name}_confusion_matrix.png",
            )

    if not gru_model_path.exists():
        raise FileNotFoundError(f"Missing GRU model artifact: {gru_model_path}")

    gru_model, gru_vocab, gru_device = load_gru_checkpoint(checkpoint_path=gru_model_path)
    evaluation_summary["gru"] = {}
    for split_name, frame in datasets.items():
        predictions = predict_gru_dataframe(gru_model, frame, gru_vocab, gru_device)
        metrics = compute_classification_metrics(frame["label"].tolist(), predictions)
        evaluation_summary["gru"][split_name] = metrics
        save_confusion_matrix_figure(
            metrics["confusion_matrix"],
            title=f"gru - {split_name}",
            output_path=FIGURES_DIR / f"{figure_prefix}gru_{split_name}_confusion_matrix.png",
        )

    return save_evaluation_summary(evaluation_summary, output_json, output_csv)


def run_evaluation_phase() -> dict[str, object]:
    ensure_output_dirs()
    project_data = load_project_data()
    datasets = {
        "phrasebank_validation": project_data["phrasebank"]["validation"],
        "phrasebank_test": project_data["phrasebank"]["test"],
        "fiqa_test": project_data["fiqa_test"],
    }

    evaluation_summary = evaluate_saved_models(
        model_filenames=CLASSICAL_MODEL_FILENAMES,
        gru_model_path=GRU_MODEL_PATH,
        datasets=datasets,
        output_json=EVALUATION_SUMMARY_JSON,
        output_csv=EVALUATION_SUMMARY_CSV,
    )

    if RAW_FINBERT_SUMMARY_PATH.exists() or (FINBERT_FINETUNED_DIR / "config.json").exists():
        from .config import FINBERT_MODEL_ID
        from .transformer_models import predict_finbert_texts

        raw_predictions = {
            split_name: predict_finbert_texts(frame["text"].tolist(), FINBERT_MODEL_ID)[0]
            for split_name, frame in datasets.items()
        }
        add_model_predictions_to_summary(
            evaluation_summary=evaluation_summary,
            model_name="raw_finbert",
            datasets=datasets,
            predictions_by_split=raw_predictions,
        )

        if (FINBERT_FINETUNED_DIR / "config.json").exists():
            finetuned_predictions = {
                split_name: predict_finbert_texts(frame["text"].tolist(), FINBERT_FINETUNED_DIR)[0]
                for split_name, frame in datasets.items()
            }
            add_model_predictions_to_summary(
                evaluation_summary=evaluation_summary,
                model_name="finbert_finetuned",
                datasets=datasets,
                predictions_by_split=finetuned_predictions,
            )

        save_evaluation_summary(evaluation_summary, EVALUATION_SUMMARY_JSON, EVALUATION_SUMMARY_CSV)

    return evaluation_summary
