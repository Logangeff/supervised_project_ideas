from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score


def compute_metrics(y_true: list[int], y_pred: list[int], label_names: list[str]) -> dict[str, object]:
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names))))
    label_counts = pd.Series(y_true).value_counts().reindex(range(len(label_names)), fill_value=0)
    prediction_counts = pd.Series(y_pred).value_counts().reindex(range(len(label_names)), fill_value=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "class_counts": {label_names[idx]: int(label_counts[idx]) for idx in range(len(label_names))},
        "prediction_counts": {label_names[idx]: int(prediction_counts[idx]) for idx in range(len(label_names))},
        "confusion_matrix": matrix.tolist(),
    }


def save_confusion_matrix_figure(
    y_true: list[int],
    y_pred: list[int],
    label_names: list[str],
    figure_path: Path,
    title: str,
) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names))))
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=label_names).plot(ax=ax, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)


def write_json(path: Path, payload: dict[str, object] | list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_rows_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
