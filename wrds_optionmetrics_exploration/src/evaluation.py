from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def compute_metrics(y_true: list[int], y_pred: list[int], y_prob: list[float], label_names: list[str]) -> dict[str, object]:
    labels = list(range(len(label_names)))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    metrics: dict[str, object] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "class_counts": {label_names[idx]: int(sum(1 for value in y_true if value == idx)) for idx in labels},
        "prediction_counts": {label_names[idx]: int(sum(1 for value in y_pred if value == idx)) for idx in labels},
        "confusion_matrix": matrix.astype(int).tolist(),
    }
    try:
        metrics["auroc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["auroc"] = None
    try:
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    except ValueError:
        metrics["pr_auc"] = None
    return metrics


def save_confusion_matrix_figure(
    y_true: list[int],
    y_pred: list[int],
    label_names: list[str],
    output_path: Path,
    title: str,
) -> None:
    labels = list(range(len(label_names)))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(title)
    fig.colorbar(image, ax=ax)
    ax.set_xticks(np.arange(len(label_names)))
    ax.set_yticks(np.arange(len(label_names)))
    ax.set_xticklabels(label_names)
    ax.set_yticklabels(label_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, int(matrix[row, col]), ha="center", va="center", color="black")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
