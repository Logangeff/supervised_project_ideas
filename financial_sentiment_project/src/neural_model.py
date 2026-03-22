from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import DataLoader, Dataset

from .config import (
    GRU_BATCH_SIZE,
    GRU_DROPOUT,
    GRU_EMBED_DIM,
    GRU_HIDDEN_SIZE,
    GRU_LEARNING_RATE,
    GRU_MAX_EPOCHS,
    GRU_MAX_LENGTH,
    GRU_MODEL_PATH,
    GRU_PATIENCE,
    GRU_TRAINING_SUMMARY_PATH,
    LABELS,
    SEED,
    ensure_output_dirs,
)
from .data_pipeline import load_project_data, set_global_seed, tokenize


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


class EncodedTextDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], vocab: dict[str, int], max_length: int) -> None:
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        tokens = tokenize(self.texts[index])[: self.max_length]
        ids = [self.vocab.get(token, self.vocab[UNK_TOKEN]) for token in tokens]
        if not ids:
            ids = [self.vocab[UNK_TOKEN]]
        return torch.tensor(ids, dtype=torch.long), int(self.labels[index])


def collate_batch(batch: list[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
    padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    return padded, lengths, label_tensor


class GRUClassifier(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, len(LABELS))

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.gru(packed)
        features = self.dropout(hidden[-1])
        return self.classifier(features)


def build_vocab(texts: list[str]) -> dict[str, int]:
    counter: Counter = Counter()
    for text in texts:
        counter.update(tokenize(text))

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token in sorted(counter):
        vocab[token] = len(vocab)
    return vocab


def build_loader(
    texts: list[str],
    label_ids: list[int],
    vocab: dict[str, int],
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = EncodedTextDataset(texts=texts, labels=label_ids, vocab=vocab, max_length=GRU_MAX_LENGTH)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_batch,
    )


def evaluate_gru_model(model: nn.Module, data_loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    predictions: list[int] = []
    labels: list[int] = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for input_ids, lengths, batch_labels in data_loader:
            input_ids = input_ids.to(device)
            lengths = lengths.to(device)
            batch_labels = batch_labels.to(device)
            logits = model(input_ids, lengths)
            loss = criterion(logits, batch_labels)
            total_loss += float(loss.item()) * len(batch_labels)
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            labels.extend(batch_labels.cpu().tolist())

    return {
        "loss": total_loss / max(len(labels), 1),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
    }


def load_gru_checkpoint(
    checkpoint_path: str | Path = GRU_MODEL_PATH,
    device: str | None = None,
) -> tuple[nn.Module, dict[str, int], torch.device]:
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(checkpoint_path, map_location=target_device)
    model = GRUClassifier(
        vocab_size=len(checkpoint["vocab"]),
        embedding_dim=checkpoint["config"]["embedding_dim"],
        hidden_size=checkpoint["config"]["hidden_size"],
        dropout=checkpoint["config"]["dropout"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(target_device)
    model.eval()
    return model, checkpoint["vocab"], target_device


def predict_gru_dataframe(model: nn.Module, frame, vocab: dict[str, int], device: torch.device) -> list[str]:
    loader = build_loader(
        texts=frame["text"].tolist(),
        label_ids=frame["label_id"].tolist(),
        vocab=vocab,
        batch_size=GRU_BATCH_SIZE,
        shuffle=False,
    )
    predictions: list[int] = []
    with torch.no_grad():
        for input_ids, lengths, _ in loader:
            logits = model(input_ids.to(device), lengths.to(device))
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
    return [LABELS[index] for index in predictions]


def train_gru_model(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    model_path: Path,
    summary_path: Path,
) -> dict[str, object]:
    train_texts = train_frame["text"].tolist()
    train_label_ids = train_frame["label_id"].tolist()
    validation_texts = validation_frame["text"].tolist()
    validation_label_ids = validation_frame["label_id"].tolist()

    vocab = build_vocab(train_texts)
    train_loader = build_loader(train_texts, train_label_ids, vocab, batch_size=GRU_BATCH_SIZE, shuffle=True)
    validation_loader = build_loader(
        validation_texts,
        validation_label_ids,
        vocab,
        batch_size=GRU_BATCH_SIZE,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRUClassifier(
        vocab_size=len(vocab),
        embedding_dim=GRU_EMBED_DIM,
        hidden_size=GRU_HIDDEN_SIZE,
        dropout=GRU_DROPOUT,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=GRU_LEARNING_RATE)

    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, float] | None = None
    training_history: list[dict[str, float]] = []
    epochs_without_improvement = 0

    for epoch in range(1, GRU_MAX_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        example_count = 0

        for input_ids, lengths, batch_labels in train_loader:
            input_ids = input_ids.to(device)
            lengths = lengths.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            logits = model(input_ids, lengths)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * len(batch_labels)
            example_count += len(batch_labels)

        validation_metrics = evaluate_gru_model(model, validation_loader, device)
        epoch_summary = {
            "epoch": epoch,
            "train_loss": running_loss / max(example_count, 1),
            "validation_loss": validation_metrics["loss"],
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_macro_f1": validation_metrics["macro_f1"],
        }
        training_history.append(epoch_summary)

        if best_metrics is None or validation_metrics["macro_f1"] > best_metrics["macro_f1"]:
            best_metrics = validation_metrics
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= GRU_PATIENCE:
            break

    if best_state is None or best_metrics is None:
        raise RuntimeError("GRU training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)
    checkpoint = {
        "state_dict": best_state,
        "vocab": vocab,
        "config": {
            "embedding_dim": GRU_EMBED_DIM,
            "hidden_size": GRU_HIDDEN_SIZE,
            "dropout": GRU_DROPOUT,
            "learning_rate": GRU_LEARNING_RATE,
            "batch_size": GRU_BATCH_SIZE,
            "max_length": GRU_MAX_LENGTH,
            "patience": GRU_PATIENCE,
            "seed": SEED,
        },
        "best_validation": best_metrics,
    }
    torch.save(checkpoint, model_path)

    summary = {
        "device": str(device),
        "vocab_size": len(vocab),
        "epochs_run": len(training_history),
        "best_validation_accuracy": best_metrics["accuracy"],
        "best_validation_macro_f1": best_metrics["macro_f1"],
        "history": training_history,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_neural_phase() -> dict[str, object]:
    ensure_output_dirs()
    set_global_seed(SEED)

    project_data = load_project_data()
    phrasebank = project_data["phrasebank"]
    return train_gru_model(
        train_frame=phrasebank["train"],
        validation_frame=phrasebank["validation"],
        model_path=GRU_MODEL_PATH,
        summary_path=GRU_TRAINING_SUMMARY_PATH,
    )
