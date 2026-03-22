from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import (
    CLASSICAL_MODEL_FILENAMES,
    EVALUATION_SUMMARY_CSV,
    EVALUATION_SUMMARY_JSON,
    FINBERT_EARLY_STOPPING_PATIENCE,
    FINBERT_EVAL_BATCH_SIZE,
    FINBERT_FINETUNED_DIR,
    FINBERT_FINETUNED_SUMMARY_PATH,
    FINBERT_LEARNING_RATE,
    FINBERT_MAX_LENGTH,
    FINBERT_MODEL_ID,
    FINBERT_NUM_EPOCHS,
    FINBERT_TRAIN_BATCH_SIZE,
    FINBERT_WEIGHT_DECAY,
    GRU_MODEL_PATH,
    LABELS,
    RAW_FINBERT_SUMMARY_PATH,
    SEED,
    ensure_output_dirs,
)
from .data_pipeline import load_project_data, set_global_seed
from .evaluate import (
    add_model_predictions_to_summary,
    evaluate_saved_models,
    save_evaluation_summary,
)


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _normalize_finbert_label(label: str) -> str:
    label = label.strip().lower()
    if label not in LABELS:
        raise ValueError(f"Unexpected FinBERT label: {label}")
    return label


def _label_from_prediction_index(model_config, prediction_index: int) -> str:
    raw_label = model_config.id2label[int(prediction_index)]
    return _normalize_finbert_label(raw_label)


def _load_finbert_for_inference(model_name_or_path: str | Path) -> tuple[AutoTokenizer, AutoModelForSequenceClassification, torch.device]:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
    device = _get_device()
    model.to(device)
    model.eval()
    return tokenizer, model, device


def predict_finbert_texts(
    texts: list[str],
    model_name_or_path: str | Path,
    batch_size: int = FINBERT_EVAL_BATCH_SIZE,
) -> tuple[list[str], dict[str, str]]:
    tokenizer, model, device = _load_finbert_for_inference(model_name_or_path)
    predictions: list[str] = []

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=FINBERT_MAX_LENGTH,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits
            batch_predictions = logits.argmax(dim=-1).cpu().tolist()
            predictions.extend(_label_from_prediction_index(model.config, idx) for idx in batch_predictions)

    mapping = {
        str(key): _normalize_finbert_label(value)
        for key, value in model.config.id2label.items()
    }
    return predictions, mapping


def _build_trainer_dataset(frame, tokenizer, label2id: dict[str, int]) -> Dataset:
    return FinbertTextDataset(
        texts=frame["text"].tolist(),
        labels=[label2id[label] for label in frame["label"].tolist()],
    )


class FinbertTextDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int]) -> None:
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> tuple[str, int]:
        return self.texts[index], int(self.labels[index])


def _build_collate_fn(tokenizer):
    def collate(batch):
        texts, labels = zip(*batch)
        encoded = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=FINBERT_MAX_LENGTH,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor(labels, dtype=torch.long)
        return encoded

    return collate


def _evaluate_finetuned_model(model, data_loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    predictions: list[int] = []
    labels: list[int] = []

    with torch.no_grad():
        for batch in data_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            losses.append(float(outputs.loss.item()))
            predictions.extend(outputs.logits.argmax(dim=-1).cpu().tolist())
            labels.extend(batch["labels"].cpu().tolist())

    return {
        "loss": float(np.mean(losses)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
    }


def _build_phrasebank_eval_datasets():
    project_data = load_project_data()
    datasets = {
        "phrasebank_validation": project_data["phrasebank"]["validation"],
        "phrasebank_test": project_data["phrasebank"]["test"],
        "fiqa_test": project_data["fiqa_test"],
    }
    return project_data, datasets


def run_raw_finbert_phase() -> dict[str, object]:
    ensure_output_dirs()
    set_global_seed(SEED)

    project_data, datasets = _build_phrasebank_eval_datasets()
    base_summary = evaluate_saved_models(
        model_filenames=CLASSICAL_MODEL_FILENAMES,
        gru_model_path=GRU_MODEL_PATH,
        datasets=datasets,
        output_json=EVALUATION_SUMMARY_JSON,
        output_csv=EVALUATION_SUMMARY_CSV,
    )

    predictions_by_split: dict[str, list[str]] = {}
    label_mapping: dict[str, str] | None = None
    for split_name, frame in datasets.items():
        predictions, mapping = predict_finbert_texts(frame["text"].tolist(), FINBERT_MODEL_ID)
        predictions_by_split[split_name] = predictions
        if label_mapping is None:
            label_mapping = mapping

    add_model_predictions_to_summary(
        evaluation_summary=base_summary,
        model_name="raw_finbert",
        datasets=datasets,
        predictions_by_split=predictions_by_split,
    )
    save_evaluation_summary(base_summary, EVALUATION_SUMMARY_JSON, EVALUATION_SUMMARY_CSV)

    raw_summary = {
        "model_id": FINBERT_MODEL_ID,
        "device": str(_get_device()),
        "checkpoint_id2label": label_mapping,
        "evaluation": base_summary["raw_finbert"],
    }
    RAW_FINBERT_SUMMARY_PATH.write_text(json.dumps(raw_summary, indent=2), encoding="utf-8")
    return raw_summary


def run_finbert_finetuned_phase() -> dict[str, object]:
    ensure_output_dirs()
    set_global_seed(SEED)

    project_data, datasets = _build_phrasebank_eval_datasets()
    phrasebank = project_data["phrasebank"]

    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL_ID)

    label2id = {label: int(idx) for label, idx in model.config.label2id.items()}
    id2label = {int(idx): label for idx, label in model.config.id2label.items()}
    label2id = {_normalize_finbert_label(label): idx for label, idx in label2id.items()}
    id2label = {idx: _normalize_finbert_label(label) for idx, label in id2label.items()}

    model.config.label2id = label2id
    model.config.id2label = {idx: label for idx, label in id2label.items()}

    train_dataset = _build_trainer_dataset(phrasebank["train"], tokenizer, label2id)
    validation_dataset = _build_trainer_dataset(phrasebank["validation"], tokenizer, label2id)
    collate_fn = _build_collate_fn(tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=FINBERT_TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=FINBERT_EVAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

    device = _get_device()
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=FINBERT_LEARNING_RATE,
        weight_decay=FINBERT_WEIGHT_DECAY,
    )

    best_state_dict = None
    best_eval = None
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, FINBERT_NUM_EPOCHS + 1):
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            outputs.loss.backward()
            optimizer.step()
            train_losses.append(float(outputs.loss.item()))

        validation_metrics = _evaluate_finetuned_model(model, validation_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "validation_loss": validation_metrics["loss"],
                "validation_accuracy": validation_metrics["accuracy"],
                "validation_macro_f1": validation_metrics["macro_f1"],
            }
        )

        if best_eval is None or validation_metrics["macro_f1"] > best_eval["macro_f1"]:
            best_eval = validation_metrics
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= FINBERT_EARLY_STOPPING_PATIENCE:
            break

    if best_state_dict is None or best_eval is None:
        raise RuntimeError("FinBERT fine-tuning did not produce a valid checkpoint.")

    model.load_state_dict(best_state_dict)
    FINBERT_FINETUNED_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(FINBERT_FINETUNED_DIR))
    tokenizer.save_pretrained(str(FINBERT_FINETUNED_DIR))

    finetune_summary = {
        "model_id": FINBERT_MODEL_ID,
        "output_dir": str(FINBERT_FINETUNED_DIR),
        "device": str(device),
        "checkpoint_label2id": label2id,
        "checkpoint_id2label": {str(key): value for key, value in id2label.items()},
        "epochs_run": len(history),
        "history": history,
        "best_validation": {
            "loss": float(best_eval["loss"]),
            "accuracy": float(best_eval["accuracy"]),
            "macro_f1": float(best_eval["macro_f1"]),
        },
    }
    FINBERT_FINETUNED_SUMMARY_PATH.write_text(json.dumps(finetune_summary, indent=2), encoding="utf-8")

    base_summary = evaluate_saved_models(
        model_filenames=CLASSICAL_MODEL_FILENAMES,
        gru_model_path=GRU_MODEL_PATH,
        datasets=datasets,
        output_json=EVALUATION_SUMMARY_JSON,
        output_csv=EVALUATION_SUMMARY_CSV,
    )

    raw_predictions_by_split: dict[str, list[str]] = {}
    finetuned_predictions_by_split: dict[str, list[str]] = {}
    for split_name, frame in datasets.items():
        raw_predictions, _ = predict_finbert_texts(frame["text"].tolist(), FINBERT_MODEL_ID)
        raw_predictions_by_split[split_name] = raw_predictions

        finetuned_predictions, _ = predict_finbert_texts(frame["text"].tolist(), FINBERT_FINETUNED_DIR)
        finetuned_predictions_by_split[split_name] = finetuned_predictions

    add_model_predictions_to_summary(
        evaluation_summary=base_summary,
        model_name="raw_finbert",
        datasets=datasets,
        predictions_by_split=raw_predictions_by_split,
    )
    add_model_predictions_to_summary(
        evaluation_summary=base_summary,
        model_name="finbert_finetuned",
        datasets=datasets,
        predictions_by_split=finetuned_predictions_by_split,
    )
    save_evaluation_summary(base_summary, EVALUATION_SUMMARY_JSON, EVALUATION_SUMMARY_CSV)

    return {
        "training": finetune_summary,
        "evaluation": {
            "raw_finbert": base_summary["raw_finbert"],
            "finbert_finetuned": base_summary["finbert_finetuned"],
        },
    }
