from __future__ import annotations

import hashlib
import json
import random
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split

from .config import (
    FIQA_DIR,
    FIQA_NEGATIVE_THRESHOLD,
    FIQA_POSITIVE_THRESHOLD,
    LABELS,
    LM_CATEGORY_COLUMNS,
    LM_CSV_PATH,
    NOSIBLE_PARQUET_PATH,
    NOSIBLE_SUMMARY_PATH,
    PHASE1_SUMMARY_PATH,
    PHRASEBANK_REPO_ID,
    PHRASEBANK_ZIP_FILENAME,
    PHRASEBANK_ZIP_MEMBER,
    SEED,
    SPLITS_DIR,
    SUMMARIES_DIR,
    ensure_output_dirs,
)


TOKEN_PATTERN = re.compile(r"[a-z]+(?:['-][a-z]+)*")
NORMALIZE_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
    }
)


def set_global_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_text(text: str) -> str:
    normalized = text.translate(NORMALIZE_TRANSLATION).lower()
    return re.sub(r"\s+", " ", normalized).strip()


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(normalize_text(text))


def _split_hash(indices: list[int]) -> str:
    payload = json.dumps(indices, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def download_phrasebank_zip() -> Path:
    zip_path = hf_hub_download(
        repo_id=PHRASEBANK_REPO_ID,
        repo_type="dataset",
        filename=PHRASEBANK_ZIP_FILENAME,
    )
    return Path(zip_path)


def load_phrasebank() -> pd.DataFrame:
    zip_path = download_phrasebank_zip()
    records: list[dict[str, object]] = []
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(PHRASEBANK_ZIP_MEMBER) as handle:
            for source_index, raw_line in enumerate(handle):
                line = raw_line.decode("latin1").strip()
                if not line:
                    continue
                sentence, label = line.rsplit("@", 1)
                label = label.strip().lower()
                if label not in LABELS:
                    raise ValueError(f"Unexpected PhraseBank label: {label}")
                records.append(
                    {
                        "source_index": source_index,
                        "text": sentence.strip(),
                        "normalized_text": normalize_text(sentence),
                        "label": label,
                        "label_id": LABELS.index(label),
                    }
                )

    frame = pd.DataFrame.from_records(records)
    if len(frame) != 4846:
        raise ValueError(f"Expected 4846 PhraseBank rows, found {len(frame)}")
    return frame


def stratified_phrasebank_split(
    phrasebank_df: pd.DataFrame, seed: int = SEED
) -> dict[str, pd.DataFrame]:
    train_df, temp_df = train_test_split(
        phrasebank_df,
        test_size=0.2,
        random_state=seed,
        stratify=phrasebank_df["label"],
    )
    validation_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=seed,
        stratify=temp_df["label"],
    )

    split_frames = {
        "train": train_df.sort_values("source_index").reset_index(drop=True),
        "validation": validation_df.sort_values("source_index").reset_index(drop=True),
        "test": test_df.sort_values("source_index").reset_index(drop=True),
    }
    return split_frames


def save_phrasebank_split_indices(split_frames: dict[str, pd.DataFrame]) -> dict[str, dict[str, object]]:
    saved: dict[str, dict[str, object]] = {}
    for split_name, frame in split_frames.items():
        indices = frame["source_index"].astype(int).tolist()
        split_path = SPLITS_DIR / f"phrasebank_{split_name}_indices.csv"
        pd.DataFrame({"source_index": indices}).to_csv(split_path, index=False)
        saved[split_name] = {
            "path": str(split_path),
            "size": len(indices),
            "sha256": _split_hash(indices),
        }
    return saved


def load_fiqa_split(split_name: str) -> pd.DataFrame:
    parquet_paths = sorted(FIQA_DIR.glob(f"{split_name}-*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"Could not find FiQA split: {split_name}")
    frame = pd.read_parquet(parquet_paths[0]).copy()
    frame["text"] = frame["sentence"].astype(str)
    frame["normalized_text"] = frame["text"].map(normalize_text)
    frame["label"] = frame["score"].map(map_fiqa_score)
    frame["label_id"] = frame["label"].map(LABELS.index)
    if not frame["label"].isin(LABELS).all():
        raise ValueError(f"Unexpected mapped FiQA labels in split {split_name}")
    return frame


def load_fiqa_all() -> dict[str, pd.DataFrame]:
    return {split_name: load_fiqa_split(split_name) for split_name in ("train", "valid", "test")}


def map_fiqa_score(score: float) -> str:
    if score >= FIQA_POSITIVE_THRESHOLD:
        return "positive"
    if score <= FIQA_NEGATIVE_THRESHOLD:
        return "negative"
    return "neutral"


def load_lm_restricted_vocabulary() -> set[str]:
    dictionary_df = pd.read_csv(LM_CSV_PATH)
    mask = dictionary_df[LM_CATEGORY_COLUMNS].gt(0).any(axis=1)
    candidate_words = dictionary_df.loc[mask, "Word"].dropna().astype(str)
    vocabulary: set[str] = set()
    for word in candidate_words:
        vocabulary.update(tokenize(word))
    if not vocabulary:
        raise ValueError("Loughran-McDonald restricted vocabulary is empty")
    return vocabulary


def _label_counts(frame: pd.DataFrame, label_column: str) -> dict[str, int]:
    counts = frame[label_column].value_counts().reindex(LABELS, fill_value=0)
    return {label: int(counts[label]) for label in LABELS}


def _load_split_indices(split_name: str) -> list[int]:
    split_path = SPLITS_DIR / f"phrasebank_{split_name}_indices.csv"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split artifact: {split_path}")
    indices = pd.read_csv(split_path)["source_index"].astype(int).tolist()
    return indices


def ensure_phase1_outputs() -> None:
    expected = [
        PHASE1_SUMMARY_PATH,
        SPLITS_DIR / "phrasebank_train_indices.csv",
        SPLITS_DIR / "phrasebank_validation_indices.csv",
        SPLITS_DIR / "phrasebank_test_indices.csv",
    ]
    if all(path.exists() for path in expected):
        return
    run_phase1()


def load_phrasebank_splits() -> dict[str, pd.DataFrame]:
    ensure_phase1_outputs()
    phrasebank_df = load_phrasebank()
    split_frames: dict[str, pd.DataFrame] = {}
    for split_name in ("train", "validation", "test"):
        indices = _load_split_indices(split_name)
        split_frame = phrasebank_df[phrasebank_df["source_index"].isin(indices)].copy()
        ordered = (
            split_frame.set_index("source_index")
            .loc[indices]
            .reset_index()
        )
        split_frames[split_name] = ordered.reset_index(drop=True)
    return split_frames


def load_project_data() -> dict[str, object]:
    phrasebank_splits = load_phrasebank_splits()
    fiqa_test = load_fiqa_split("test")
    lm_vocabulary = load_lm_restricted_vocabulary()
    return {
        "phrasebank": phrasebank_splits,
        "fiqa_test": fiqa_test,
        "lm_vocabulary": lm_vocabulary,
    }


def load_nosible() -> pd.DataFrame:
    frame = pd.read_parquet(NOSIBLE_PARQUET_PATH).copy()
    frame["text"] = frame["text"].astype(str)
    frame["normalized_text"] = frame["text"].map(normalize_text)
    frame["label"] = frame["label"].astype(str).str.lower()
    frame["label_id"] = frame["label"].map(LABELS.index)
    frame["source_index"] = np.arange(len(frame))
    if not frame["label"].isin(LABELS).all():
        raise ValueError("Unexpected NOSIBLE labels.")
    return frame


def stratified_nosible_split(nosible_df: pd.DataFrame, seed: int = SEED) -> dict[str, pd.DataFrame]:
    train_df, temp_df = train_test_split(
        nosible_df,
        test_size=0.2,
        random_state=seed,
        stratify=nosible_df["label"],
    )
    validation_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=seed,
        stratify=temp_df["label"],
    )
    return {
        "train": train_df.sort_values("source_index").reset_index(drop=True),
        "validation": validation_df.sort_values("source_index").reset_index(drop=True),
        "test": test_df.sort_values("source_index").reset_index(drop=True),
    }


def save_nosible_split_indices(split_frames: dict[str, pd.DataFrame]) -> dict[str, dict[str, object]]:
    saved: dict[str, dict[str, object]] = {}
    for split_name, frame in split_frames.items():
        indices = frame["source_index"].astype(int).tolist()
        split_path = SPLITS_DIR / f"nosible_{split_name}_indices.csv"
        pd.DataFrame({"source_index": indices}).to_csv(split_path, index=False)
        saved[split_name] = {
            "path": str(split_path),
            "size": len(indices),
            "sha256": _split_hash(indices),
        }
    return saved


def ensure_nosible_outputs() -> None:
    expected = [
        NOSIBLE_SUMMARY_PATH,
        SPLITS_DIR / "nosible_train_indices.csv",
        SPLITS_DIR / "nosible_validation_indices.csv",
        SPLITS_DIR / "nosible_test_indices.csv",
    ]
    if all(path.exists() for path in expected):
        return
    run_nosible_phase1()


def load_nosible_splits() -> dict[str, pd.DataFrame]:
    ensure_nosible_outputs()
    nosible_df = load_nosible()
    split_frames: dict[str, pd.DataFrame] = {}
    for split_name in ("train", "validation", "test"):
        split_path = SPLITS_DIR / f"nosible_{split_name}_indices.csv"
        indices = pd.read_csv(split_path)["source_index"].astype(int).tolist()
        split_frame = nosible_df[nosible_df["source_index"].isin(indices)].copy()
        ordered = split_frame.set_index("source_index").loc[indices].reset_index()
        split_frames[split_name] = ordered.reset_index(drop=True)
    return split_frames


def load_nosible_experiment_data() -> dict[str, object]:
    nosible_splits = load_nosible_splits()
    phrasebank = load_phrasebank_splits()
    fiqa_test = load_fiqa_split("test")
    lm_vocabulary = load_lm_restricted_vocabulary()
    return {
        "nosible": nosible_splits,
        "phrasebank": phrasebank,
        "fiqa_test": fiqa_test,
        "lm_vocabulary": lm_vocabulary,
    }


def build_phase1_summary() -> dict[str, object]:
    ensure_output_dirs()
    set_global_seed(SEED)

    phrasebank_df = load_phrasebank()
    split_frames = stratified_phrasebank_split(phrasebank_df, seed=SEED)
    split_artifacts = save_phrasebank_split_indices(split_frames)

    fiqa_frames = load_fiqa_all()
    lm_vocabulary = load_lm_restricted_vocabulary()

    rerun_split_artifacts = {
        split_name: _split_hash(
            split_frame["source_index"].astype(int).tolist()
        )
        for split_name, split_frame in stratified_phrasebank_split(phrasebank_df, seed=SEED).items()
    }

    smoke_checks = {
        "phrasebank_row_count_is_4846": len(phrasebank_df) == 4846,
        "phrasebank_split_total_matches": sum(
            len(split_frame) for split_frame in split_frames.values()
        )
        == len(phrasebank_df),
        "fiqa_labels_are_valid": all(
            frame["label"].isin(LABELS).all() for frame in fiqa_frames.values()
        ),
        "lm_vocabulary_non_empty": bool(lm_vocabulary),
        "lm_vocabulary_is_lowercase": all(token == token.lower() for token in lm_vocabulary),
        "repeated_split_hashes_match": all(
            split_artifacts[split_name]["sha256"] == rerun_split_artifacts[split_name]
            for split_name in split_frames
        ),
    }

    if not all(smoke_checks.values()):
        failed = [name for name, passed in smoke_checks.items() if not passed]
        raise RuntimeError(f"Phase 1 smoke checks failed: {failed}")

    summary = {
        "seed": SEED,
        "phrasebank": {
            "source": {
                "repo_id": PHRASEBANK_REPO_ID,
                "zip_member": PHRASEBANK_ZIP_MEMBER,
            },
            "row_count": int(len(phrasebank_df)),
            "label_counts": _label_counts(phrasebank_df, "label"),
            "splits": {
                split_name: {
                    "size": int(len(split_frame)),
                    "label_counts": _label_counts(split_frame, "label"),
                    "artifact": split_artifacts[split_name],
                }
                for split_name, split_frame in split_frames.items()
            },
        },
        "fiqa": {
            "source_dir": str(FIQA_DIR),
            "thresholds": {
                "negative_lte": FIQA_NEGATIVE_THRESHOLD,
                "positive_gte": FIQA_POSITIVE_THRESHOLD,
            },
            "splits": {
                split_name: {
                    "size": int(len(frame)),
                    "mapped_label_counts": _label_counts(frame, "label"),
                }
                for split_name, frame in fiqa_frames.items()
            },
            "official_test_size": int(len(fiqa_frames["test"])),
            "official_test_mapped_label_counts": _label_counts(fiqa_frames["test"], "label"),
        },
        "loughran_mcdonald": {
            "csv_path": str(LM_CSV_PATH),
            "category_columns": LM_CATEGORY_COLUMNS,
            "restricted_vocabulary_size": int(len(lm_vocabulary)),
        },
        "smoke_checks": smoke_checks,
    }
    return summary


def run_phase1() -> dict[str, object]:
    summary = build_phase1_summary()
    summary_path = SUMMARIES_DIR / "phase1_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_nosible_summary() -> dict[str, object]:
    ensure_output_dirs()
    set_global_seed(SEED)

    nosible_df = load_nosible()
    split_frames = stratified_nosible_split(nosible_df, seed=SEED)
    split_artifacts = save_nosible_split_indices(split_frames)
    lm_vocabulary = load_lm_restricted_vocabulary()

    rerun_split_artifacts = {
        split_name: _split_hash(
            split_frame["source_index"].astype(int).tolist()
        )
        for split_name, split_frame in stratified_nosible_split(nosible_df, seed=SEED).items()
    }

    smoke_checks = {
        "nosible_row_count_is_100000": len(nosible_df) == 100000,
        "nosible_split_total_matches": sum(len(split_frame) for split_frame in split_frames.values()) == len(nosible_df),
        "nosible_labels_are_valid": bool(nosible_df["label"].isin(LABELS).all()),
        "lm_vocabulary_non_empty": bool(lm_vocabulary),
        "repeated_split_hashes_match": all(
            split_artifacts[split_name]["sha256"] == rerun_split_artifacts[split_name]
            for split_name in split_frames
        ),
    }

    if not all(smoke_checks.values()):
        failed = [name for name, passed in smoke_checks.items() if not passed]
        raise RuntimeError(f"NOSIBLE smoke checks failed: {failed}")

    summary = {
        "seed": SEED,
        "nosible": {
            "source_path": str(NOSIBLE_PARQUET_PATH),
            "row_count": int(len(nosible_df)),
            "label_counts": _label_counts(nosible_df, "label"),
            "splits": {
                split_name: {
                    "size": int(len(split_frame)),
                    "label_counts": _label_counts(split_frame, "label"),
                    "artifact": split_artifacts[split_name],
                }
                for split_name, split_frame in split_frames.items()
            },
        },
        "loughran_mcdonald": {
            "csv_path": str(LM_CSV_PATH),
            "category_columns": LM_CATEGORY_COLUMNS,
            "restricted_vocabulary_size": int(len(lm_vocabulary)),
        },
        "smoke_checks": smoke_checks,
    }
    return summary


def run_nosible_phase1() -> dict[str, object]:
    summary = build_nosible_summary()
    NOSIBLE_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
