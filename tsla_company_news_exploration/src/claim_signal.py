from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from .config import LOCKED_CLAIM_DETECTOR_META


def _resolve_locked_model_path() -> Path:
    metadata = json.loads(LOCKED_CLAIM_DETECTOR_META.read_text(encoding="utf-8"))
    if metadata["model_type"] != "classical":
        raise RuntimeError("The locked Stage 1 detector is not a classical model in the current environment.")
    return LOCKED_CLAIM_DETECTOR_META.parent.parent.parent / metadata["path"]


def load_claim_detector_bundle() -> dict[str, object]:
    return joblib.load(_resolve_locked_model_path())


def score_claim_probabilities(texts: list[str]) -> np.ndarray:
    bundle = load_claim_detector_bundle()
    features = bundle["vectorizer"].transform(texts)
    return np.asarray(bundle["classifier"].predict_proba(features)[:, 1], dtype=np.float32)
