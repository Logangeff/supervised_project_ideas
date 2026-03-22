from __future__ import annotations

import copy
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import (
    CLASSICAL_C_GRID,
    CLASSICAL_MIN_DF,
    GRU_BATCH_SIZE,
    GRU_DROPOUT,
    GRU_EMBED_DIM,
    GRU_HIDDEN_SIZE,
    GRU_LEARNING_RATE,
    GRU_MAX_EPOCHS,
    GRU_MAX_LENGTH,
    GRU_PATIENCE,
    SEED,
    STAGE1_BEST_DETECTOR_PATH,
    STAGE1_CLASSICAL_FIGURE_PATH,
    STAGE1_CLASSICAL_MODEL_PATH,
    STAGE1_CLASSICAL_SUMMARY_PATH,
    STAGE1_EVALUATION_CSV,
    STAGE1_EVALUATION_JSON,
    STAGE1_GRU_FIGURE_PATH,
    STAGE1_GRU_MODEL_PATH,
    STAGE1_ID_TO_LABEL,
    STAGE1_LABEL_NAMES,
    STAGE1_NEURAL_SUMMARY_PATH,
    ensure_project_dirs,
    relative_to_root,
    set_global_seed,
)
from .evaluation import compute_metrics, save_confusion_matrix_figure, write_json, write_rows_csv
from .stage1_data import load_stage1_splits
from .text_utils import tokenize


class EncodedTextDataset(Dataset):
    def __init__(self, sequences: list[list[int]], labels: list[int] | None = None) -> None:
        self.sequences = [torch.tensor(sequence, dtype=torch.long) for sequence in sequences]
        self.labels = None if labels is None else torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int):
        if self.labels is None:
            return self.sequences[index]
        return self.sequences[index], self.labels[index]


class GRUClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, len(STAGE1_LABEL_NAMES))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        _, hidden = self.gru(embedded)
        hidden = self.dropout(hidden[-1])
        return self.classifier(hidden)


def _model_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _score_tuple(metrics: dict[str, object], model_name: str) -> tuple[float, float, str]:
    return (float(metrics["macro_f1"]), float(metrics["accuracy"]), model_name)


def _build_tfidf_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        tokenizer=tokenize,
        preprocessor=None,
        lowercase=False,
        token_pattern=None,
        ngram_range=(1, 1),
        min_df=CLASSICAL_MIN_DF,
    )


def _build_vocab(texts: list[str]) -> dict[str, int]:
    vocab = {"<pad>": 0, "<unk>": 1}
    for text in texts:
        for token in tokenize(text):
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def _encode_text(text: str, vocab: dict[str, int], max_length: int = GRU_MAX_LENGTH) -> list[int]:
    token_ids = [vocab.get(token, 1) for token in tokenize(text)[:max_length]]
    if len(token_ids) < max_length:
        token_ids.extend([0] * (max_length - len(token_ids)))
    return token_ids


def _predict_classical_probabilities(model_bundle: dict[str, object], texts: list[str]) -> np.ndarray:
    vectorizer = model_bundle["vectorizer"]
    classifier = model_bundle["classifier"]
    features = vectorizer.transform(texts)
    return classifier.predict_proba(features)[:, 1]


def _build_gru_checkpoint(model: GRUClassifier, vocab: dict[str, int]) -> dict[str, object]:
    return {
        "model_type": "gru",
        "state_dict": model.state_dict(),
        "vocab": vocab,
        "config": {
            "embed_dim": GRU_EMBED_DIM,
            "hidden_size": GRU_HIDDEN_SIZE,
            "dropout": GRU_DROPOUT,
            "max_length": GRU_MAX_LENGTH,
        },
        "label_names": STAGE1_LABEL_NAMES,
    }


def _load_gru_from_checkpoint(checkpoint: dict[str, object], device: torch.device) -> GRUClassifier:
    model = GRUClassifier(
        vocab_size=len(checkpoint["vocab"]),
        embed_dim=checkpoint["config"]["embed_dim"],
        hidden_size=checkpoint["config"]["hidden_size"],
        dropout=checkpoint["config"]["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def _predict_gru_probabilities(checkpoint: dict[str, object], texts: list[str], batch_size: int = GRU_BATCH_SIZE) -> np.ndarray:
    if not texts:
        return np.array([], dtype=np.float32)

    device = _model_device()
    model = _load_gru_from_checkpoint(checkpoint, device)
    vocab = checkpoint["vocab"]
    max_length = checkpoint["config"]["max_length"]
    sequences = [_encode_text(text, vocab, max_length=max_length) for text in texts]
    dataloader = DataLoader(EncodedTextDataset(sequences), batch_size=batch_size, shuffle=False)

    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().tolist()
            probabilities.extend(float(value) for value in probs)

    return np.asarray(probabilities, dtype=np.float32)


def _classification_rows(model_name: str, split_name: str, y_true: list[int], y_pred: list[int]) -> dict[str, object]:
    metrics = compute_metrics(y_true, y_pred, STAGE1_LABEL_NAMES)
    return {
        "model": model_name,
        "split": split_name,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
    }


def run_stage1_classical() -> dict[str, object]:
    ensure_project_dirs()
    set_global_seed(SEED)
    splits = load_stage1_splits()

    train_texts = splits["train"]["text"].tolist()
    validation_texts = splits["validation"]["text"].tolist()
    test_texts = splits["test"]["text"].tolist()
    y_train = splits["train"]["label_id"].astype(int).tolist()
    y_validation = splits["validation"]["label_id"].astype(int).tolist()
    y_test = splits["test"]["label_id"].astype(int).tolist()

    best_bundle: dict[str, object] | None = None
    best_metrics: dict[str, object] | None = None
    best_c = None

    for c_value in CLASSICAL_C_GRID:
        vectorizer = _build_tfidf_vectorizer()
        train_features = vectorizer.fit_transform(train_texts)
        classifier = LogisticRegression(max_iter=2000, C=c_value, random_state=SEED)
        classifier.fit(train_features, y_train)
        validation_predictions = classifier.predict(vectorizer.transform(validation_texts))
        metrics = compute_metrics(y_validation, validation_predictions.tolist(), STAGE1_LABEL_NAMES)
        if best_metrics is None or _score_tuple(metrics, "tfidf_logreg") > _score_tuple(best_metrics, "tfidf_logreg"):
            best_bundle = {
                "model_type": "classical",
                "model_name": "tfidf_logreg",
                "vectorizer": vectorizer,
                "classifier": classifier,
                "label_names": STAGE1_LABEL_NAMES,
            }
            best_metrics = metrics
            best_c = c_value
            best_feature_count = int(train_features.shape[1])

    assert best_bundle is not None and best_metrics is not None and best_c is not None

    joblib.dump(best_bundle, STAGE1_CLASSICAL_MODEL_PATH)

    validation_predictions = best_bundle["classifier"].predict(best_bundle["vectorizer"].transform(validation_texts)).tolist()
    test_predictions = best_bundle["classifier"].predict(best_bundle["vectorizer"].transform(test_texts)).tolist()
    validation_metrics = compute_metrics(y_validation, validation_predictions, STAGE1_LABEL_NAMES)
    test_metrics = compute_metrics(y_test, test_predictions, STAGE1_LABEL_NAMES)

    save_confusion_matrix_figure(
        y_test,
        test_predictions,
        STAGE1_LABEL_NAMES,
        STAGE1_CLASSICAL_FIGURE_PATH,
        "Stage 1 TF-IDF + Logistic Regression (Test)",
    )

    summary = {
        "model": "tfidf_logreg",
        "best_c": best_c,
        "feature_count": best_feature_count,
        "validation": validation_metrics,
        "test": test_metrics,
    }
    write_json(STAGE1_CLASSICAL_SUMMARY_PATH, summary)
    return summary


def run_stage1_neural() -> dict[str, object]:
    ensure_project_dirs()
    set_global_seed(SEED)
    splits = load_stage1_splits()

    train_texts = splits["train"]["text"].tolist()
    validation_texts = splits["validation"]["text"].tolist()
    test_texts = splits["test"]["text"].tolist()
    y_train = splits["train"]["label_id"].astype(int).tolist()
    y_validation = splits["validation"]["label_id"].astype(int).tolist()
    y_test = splits["test"]["label_id"].astype(int).tolist()

    vocab = _build_vocab(train_texts)
    train_sequences = [_encode_text(text, vocab) for text in train_texts]
    validation_sequences = [_encode_text(text, vocab) for text in validation_texts]

    train_loader = DataLoader(
        EncodedTextDataset(train_sequences, y_train),
        batch_size=GRU_BATCH_SIZE,
        shuffle=True,
    )
    validation_loader = DataLoader(
        EncodedTextDataset(validation_sequences, y_validation),
        batch_size=GRU_BATCH_SIZE,
        shuffle=False,
    )

    device = _model_device()
    model = GRUClassifier(
        vocab_size=len(vocab),
        embed_dim=GRU_EMBED_DIM,
        hidden_size=GRU_HIDDEN_SIZE,
        dropout=GRU_DROPOUT,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=GRU_LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    best_checkpoint: dict[str, object] | None = None
    best_validation_metrics: dict[str, object] | None = None
    best_epoch = 0
    patience_counter = 0
    history: list[dict[str, object]] = []

    for epoch in range(1, GRU_MAX_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        sample_count = 0
        for batch_inputs, batch_labels in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            logits = model(batch_inputs)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * batch_labels.size(0)
            sample_count += int(batch_labels.size(0))

        model.eval()
        validation_probabilities: list[float] = []
        with torch.no_grad():
            for batch_inputs, _ in validation_loader:
                batch_inputs = batch_inputs.to(device)
                logits = model(batch_inputs)
                probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().tolist()
                validation_probabilities.extend(float(value) for value in probs)

        validation_probabilities_array = np.asarray(validation_probabilities, dtype=np.float32)
        validation_predictions = (validation_probabilities_array >= 0.5).astype(int).tolist()
        validation_metrics = compute_metrics(y_validation, validation_predictions, STAGE1_LABEL_NAMES)
        epoch_record = {
            "epoch": epoch,
            "train_loss": running_loss / max(sample_count, 1),
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_macro_f1": validation_metrics["macro_f1"],
        }
        history.append(epoch_record)

        if best_validation_metrics is None or _score_tuple(validation_metrics, "gru") > _score_tuple(best_validation_metrics, "gru"):
            best_validation_metrics = validation_metrics
            best_checkpoint = _build_gru_checkpoint(copy.deepcopy(model).cpu(), vocab)
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= GRU_PATIENCE:
            break

    assert best_checkpoint is not None and best_validation_metrics is not None
    torch.save(best_checkpoint, STAGE1_GRU_MODEL_PATH)

    validation_probabilities = _predict_gru_probabilities(best_checkpoint, validation_texts)
    test_probabilities = _predict_gru_probabilities(best_checkpoint, test_texts)
    validation_predictions = (validation_probabilities >= 0.5).astype(int).tolist()
    test_predictions = (test_probabilities >= 0.5).astype(int).tolist()
    validation_metrics = compute_metrics(y_validation, validation_predictions, STAGE1_LABEL_NAMES)
    test_metrics = compute_metrics(y_test, test_predictions, STAGE1_LABEL_NAMES)

    save_confusion_matrix_figure(
        y_test,
        test_predictions,
        STAGE1_LABEL_NAMES,
        STAGE1_GRU_FIGURE_PATH,
        "Stage 1 GRU (Test)",
    )

    summary = {
        "model": "gru",
        "device": str(device),
        "best_epoch": best_epoch,
        "vocab_size": len(vocab),
        "history": history,
        "validation": validation_metrics,
        "test": test_metrics,
    }
    write_json(STAGE1_NEURAL_SUMMARY_PATH, summary)
    return summary


def _load_classical_bundle() -> dict[str, object]:
    if not STAGE1_CLASSICAL_MODEL_PATH.exists():
        run_stage1_classical()
    return joblib.load(STAGE1_CLASSICAL_MODEL_PATH)


def _load_gru_checkpoint() -> dict[str, object]:
    if not STAGE1_GRU_MODEL_PATH.exists():
        run_stage1_neural()
    return torch.load(STAGE1_GRU_MODEL_PATH, map_location="cpu")


def _evaluate_classical_bundle(bundle: dict[str, object], split_frames: dict[str, object]) -> tuple[list[dict[str, object]], dict[str, list[int]]]:
    rows: list[dict[str, object]] = []
    predictions: dict[str, list[int]] = {}
    for split_name, frame in split_frames.items():
        probs = _predict_classical_probabilities(bundle, frame["text"].tolist())
        preds = (probs >= 0.5).astype(int).tolist()
        rows.append(_classification_rows("tfidf_logreg", split_name, frame["label_id"].astype(int).tolist(), preds))
        predictions[split_name] = preds
    return rows, predictions


def _evaluate_gru_checkpoint(checkpoint: dict[str, object], split_frames: dict[str, object]) -> tuple[list[dict[str, object]], dict[str, list[int]]]:
    rows: list[dict[str, object]] = []
    predictions: dict[str, list[int]] = {}
    for split_name, frame in split_frames.items():
        probs = _predict_gru_probabilities(checkpoint, frame["text"].tolist())
        preds = (probs >= 0.5).astype(int).tolist()
        rows.append(_classification_rows("gru", split_name, frame["label_id"].astype(int).tolist(), preds))
        predictions[split_name] = preds
    return rows, predictions


def run_stage1_evaluate() -> dict[str, object]:
    ensure_project_dirs()
    set_global_seed(SEED)
    split_frames = load_stage1_splits()
    classical_bundle = _load_classical_bundle()
    gru_checkpoint = _load_gru_checkpoint()

    classical_rows, classical_predictions = _evaluate_classical_bundle(classical_bundle, split_frames)
    gru_rows, gru_predictions = _evaluate_gru_checkpoint(gru_checkpoint, split_frames)
    all_rows = classical_rows + gru_rows
    write_rows_csv(STAGE1_EVALUATION_CSV, all_rows)
    write_json(STAGE1_EVALUATION_JSON, {"rows": all_rows})

    validation_rows = [row for row in all_rows if row["split"] == "validation"]
    best_row = max(
        validation_rows,
        key=lambda row: (float(row["macro_f1"]), float(row["accuracy"]), row["model"]),
    )
    best_detector = {
        "model_name": best_row["model"],
        "model_type": "classical" if best_row["model"] == "tfidf_logreg" else "gru",
        "path": relative_to_root(
            STAGE1_CLASSICAL_MODEL_PATH if best_row["model"] == "tfidf_logreg" else STAGE1_GRU_MODEL_PATH
        ),
        "positive_label_name": STAGE1_ID_TO_LABEL[1],
        "positive_label_id": 1,
        "selection_split": "validation",
        "selection_metric": "macro_f1",
        "validation_macro_f1": best_row["macro_f1"],
        "validation_accuracy": best_row["accuracy"],
    }
    write_json(STAGE1_BEST_DETECTOR_PATH, best_detector)

    return {"rows": all_rows, "best_detector": best_detector}


def score_texts_with_best_detector(texts: list[str]) -> np.ndarray:
    if not STAGE1_BEST_DETECTOR_PATH.exists():
        run_stage1_evaluate()
    metadata = json.loads(STAGE1_BEST_DETECTOR_PATH.read_text(encoding="utf-8"))
    if metadata["model_type"] == "classical":
        return _predict_classical_probabilities(_load_classical_bundle(), texts)
    return _predict_gru_probabilities(_load_gru_checkpoint(), texts)
