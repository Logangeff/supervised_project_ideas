from __future__ import annotations

import os

import pandas as pd
import torch

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import FINBERT_BATCH_SIZE, FINBERT_MAX_LENGTH, FINBERT_MODEL_ID


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def score_finbert_sentiment(texts: list[str]) -> pd.DataFrame:
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL_ID)
    device = _device()
    model.to(device)
    model.eval()

    rows: list[dict[str, float | str]] = []
    id2label = {int(idx): str(label).strip().lower() for idx, label in model.config.id2label.items()}

    with torch.no_grad():
        for start in range(0, len(texts), FINBERT_BATCH_SIZE):
            batch_texts = texts[start : start + FINBERT_BATCH_SIZE]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=FINBERT_MAX_LENGTH,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            probabilities = torch.softmax(model(**encoded).logits, dim=-1).detach().cpu()

            for probs in probabilities.tolist():
                prob_map = {
                    id2label[idx]: float(prob)
                    for idx, prob in enumerate(probs)
                }
                positive_prob = prob_map.get("positive", 0.0)
                negative_prob = prob_map.get("negative", 0.0)
                neutral_prob = prob_map.get("neutral", 0.0)
                rows.append(
                    {
                        "finbert_positive_prob": positive_prob,
                        "finbert_negative_prob": negative_prob,
                        "finbert_neutral_prob": neutral_prob,
                        "finbert_sentiment_score": positive_prob - negative_prob,
                        "finbert_label": max(
                            (
                                ("positive", positive_prob),
                                ("neutral", neutral_prob),
                                ("negative", negative_prob),
                            ),
                            key=lambda item: item[1],
                        )[0],
                    }
                )

    return pd.DataFrame(rows)
