from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from datasets import DatasetDict, load_dataset
from sklearn.model_selection import train_test_split

from .config import (
    NUMCLAIM_CACHE_DIR,
    NUMCLAIM_DATASET_ID,
    SEED,
    STAGE1_DATA_SUMMARY_PATH,
    STAGE1_LABEL_TO_ID,
    STAGE1_TEST_SPLIT_PATH,
    STAGE1_TRAIN_SPLIT_PATH,
    STAGE1_VALIDATION_SIZE,
    STAGE1_VALIDATION_SPLIT_PATH,
    ensure_project_dirs,
    relative_to_root,
)
from .evaluation import write_json
from .text_utils import normalize_text


def _split_hash(example_ids: list[str]) -> str:
    payload = json.dumps(example_ids, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_numclaim_dataset() -> DatasetDict:
    ensure_project_dirs()
    return load_dataset(NUMCLAIM_DATASET_ID, cache_dir=str(NUMCLAIM_CACHE_DIR))


def _dataset_split_to_frame(dataset_split, source_split: str) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "source_split": source_split,
            "source_index": range(len(dataset_split)),
            "text": dataset_split["context"],
            "label_name": dataset_split["response"],
        }
    )
    frame["label_name"] = frame["label_name"].astype(str).str.upper()
    frame["label_id"] = frame["label_name"].map(STAGE1_LABEL_TO_ID)
    frame["normalized_text"] = frame["text"].map(normalize_text)
    frame["example_id"] = frame.apply(
        lambda row: f"{row['source_split']}-{int(row['source_index'])}",
        axis=1,
    )
    if frame["label_id"].isna().any():
        raise ValueError("Unexpected labels found in NumClaim.")
    frame["label_id"] = frame["label_id"].astype(int)
    return frame


def build_stage1_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset = load_numclaim_dataset()
    train_df = _dataset_split_to_frame(dataset["train"], "train")
    test_df = _dataset_split_to_frame(dataset["test"], "test")
    return train_df, test_df


def create_stage1_splits() -> dict[str, pd.DataFrame]:
    official_train_df, official_test_df = build_stage1_frames()
    train_indices, validation_indices = train_test_split(
        official_train_df["source_index"].tolist(),
        test_size=STAGE1_VALIDATION_SIZE,
        random_state=SEED,
        stratify=official_train_df["label_id"],
    )

    train_df = (
        official_train_df[official_train_df["source_index"].isin(train_indices)]
        .sort_values("source_index")
        .reset_index(drop=True)
    )
    validation_df = (
        official_train_df[official_train_df["source_index"].isin(validation_indices)]
        .sort_values("source_index")
        .reset_index(drop=True)
    )
    test_df = official_test_df.sort_values("source_index").reset_index(drop=True)

    return {"train": train_df, "validation": validation_df, "test": test_df}


def save_stage1_split_indices(split_frames: dict[str, pd.DataFrame]) -> dict[str, dict[str, object]]:
    artifact_paths = {
        "train": STAGE1_TRAIN_SPLIT_PATH,
        "validation": STAGE1_VALIDATION_SPLIT_PATH,
        "test": STAGE1_TEST_SPLIT_PATH,
    }
    saved: dict[str, dict[str, object]] = {}
    for split_name, frame in split_frames.items():
        split_path = artifact_paths[split_name]
        payload = frame[["example_id", "source_split", "source_index"]].copy()
        payload.to_csv(split_path, index=False)
        saved[split_name] = {
            "path": relative_to_root(split_path),
            "size": int(len(payload)),
            "sha256": _split_hash(payload["example_id"].tolist()),
        }
    return saved


def _load_split_csv(split_path: Path) -> pd.DataFrame:
    if not split_path.exists():
        raise FileNotFoundError(f"Missing Stage 1 split artifact: {split_path}")
    frame = pd.read_csv(split_path)
    frame["source_index"] = frame["source_index"].astype(int)
    return frame


def ensure_stage1_data_outputs() -> None:
    expected_paths = [
        STAGE1_DATA_SUMMARY_PATH,
        STAGE1_TRAIN_SPLIT_PATH,
        STAGE1_VALIDATION_SPLIT_PATH,
        STAGE1_TEST_SPLIT_PATH,
    ]
    if all(path.exists() for path in expected_paths):
        return
    run_stage1_data()


def load_stage1_splits() -> dict[str, pd.DataFrame]:
    ensure_stage1_data_outputs()
    official_train_df, official_test_df = build_stage1_frames()
    source_frames = {"train": official_train_df, "test": official_test_df}
    split_csvs = {
        "train": _load_split_csv(STAGE1_TRAIN_SPLIT_PATH),
        "validation": _load_split_csv(STAGE1_VALIDATION_SPLIT_PATH),
        "test": _load_split_csv(STAGE1_TEST_SPLIT_PATH),
    }

    split_frames: dict[str, pd.DataFrame] = {}
    for split_name, split_ids in split_csvs.items():
        source_split = split_ids["source_split"].iloc[0]
        source_frame = source_frames[source_split].set_index("example_id")
        ordered = source_frame.loc[split_ids["example_id"].tolist()].reset_index()
        split_frames[split_name] = ordered.reset_index(drop=True)
    return split_frames


def build_stage1_data_summary() -> dict[str, object]:
    ensure_project_dirs()
    official_train_df, official_test_df = build_stage1_frames()
    split_frames = create_stage1_splits()
    split_artifacts = save_stage1_split_indices(split_frames)

    rerun_hashes = {
        split_name: _split_hash(frame["example_id"].tolist())
        for split_name, frame in create_stage1_splits().items()
    }
    smoke_checks = {
        "official_split_sizes_match": len(official_train_df) == 2144 and len(official_test_df) == 537,
        "derived_train_validation_sum_matches": len(split_frames["train"]) + len(split_frames["validation"]) == len(official_train_df),
        "labels_are_binary": all(frame["label_id"].isin([0, 1]).all() for frame in split_frames.values()),
        "split_hashes_stable": all(
            split_artifacts[split_name]["sha256"] == rerun_hashes[split_name]
            for split_name in split_frames
        ),
    }
    if not all(smoke_checks.values()):
        failed = [name for name, passed in smoke_checks.items() if not passed]
        raise RuntimeError(f"Stage 1 data smoke checks failed: {failed}")

    summary = {
        "dataset_id": NUMCLAIM_DATASET_ID,
        "official_splits": {
            "train": int(len(official_train_df)),
            "test": int(len(official_test_df)),
        },
        "label_mapping": STAGE1_LABEL_TO_ID,
        "validation_rule": {
            "source": "official_train_only",
            "type": "stratified",
            "train_fraction": 1.0 - STAGE1_VALIDATION_SIZE,
            "validation_fraction": STAGE1_VALIDATION_SIZE,
            "seed": SEED,
        },
        "splits": {
            split_name: {
                "size": int(len(frame)),
                "label_counts": {
                    label_name: int((frame["label_name"] == label_name).sum())
                    for label_name in STAGE1_LABEL_TO_ID
                },
                "artifact": split_artifacts[split_name],
            }
            for split_name, frame in split_frames.items()
        },
        "smoke_checks": smoke_checks,
    }
    return summary


def run_stage1_data() -> dict[str, object]:
    summary = build_stage1_data_summary()
    write_json(STAGE1_DATA_SUMMARY_PATH, summary)
    return summary
