from __future__ import annotations

import hashlib
import json
import lzma
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

from .config import (
    FINANCIAL_NEWS_DATASET_ID,
    FINANCIAL_NEWS_RAW_DIR,
    INTERMEDIATE_DIR,
    SEED,
    STAGE2_DATA_SUMMARY_PATH,
    STAGE2_FILTERED_DATA_PATH,
    STAGE2_TEST_RATIO,
    STAGE2_TEST_SPLIT_PATH,
    STAGE2_TRAIN_RATIO,
    STAGE2_TRAIN_SPLIT_PATH,
    STAGE2_VALIDATION_RATIO,
    STAGE2_VALIDATION_SPLIT_PATH,
    ensure_project_dirs,
)
from .evaluation import write_json
from .text_utils import normalize_text


def download_financial_news_files() -> Path:
    ensure_project_dirs()
    snapshot_download(
        repo_id=FINANCIAL_NEWS_DATASET_ID,
        repo_type="dataset",
        local_dir=str(FINANCIAL_NEWS_RAW_DIR),
        allow_patterns=["*.xz", "README.md", ".gitattributes"],
    )
    return FINANCIAL_NEWS_RAW_DIR


def _numeric_price(article: dict[str, object], ticker: str, prefix: str) -> float | None:
    key = f"{prefix}_{ticker}"
    value = article.get(key)
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(price) or price <= 0:
        return None
    return price


def _extract_single_ticker(raw_value: object) -> str | None:
    if not isinstance(raw_value, list):
        return None
    tickers = []
    for item in raw_value:
        ticker = str(item).strip().upper()
        if ticker:
            tickers.append(ticker)
    unique_tickers = sorted(set(tickers))
    if len(unique_tickers) != 1:
        return None
    return unique_tickers[0]


def _event_id(base_identifier: str, ticker: str, publish_time: pd.Timestamp) -> str:
    payload = f"{base_identifier}|{ticker}|{publish_time.isoformat()}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _load_year_file(path: Path) -> list[dict[str, object]]:
    with lzma.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _build_filtered_stage2_frame() -> tuple[pd.DataFrame, dict[str, int]]:
    raw_dir = download_financial_news_files()
    counters = Counter()
    records: list[dict[str, object]] = []

    for file_path in sorted(raw_dir.glob("*.json.xz")):
        for article in _load_year_file(file_path):
            counters["raw_rows"] += 1
            if str(article.get("language", "")).lower() != "en":
                counters["dropped_non_english"] += 1
                continue

            title = str(article.get("title", "")).strip()
            if not title:
                counters["dropped_missing_title"] += 1
                continue

            publish_time = pd.to_datetime(article.get("date_publish"), errors="coerce")
            if pd.isna(publish_time):
                counters["dropped_invalid_date"] += 1
                continue

            ticker = _extract_single_ticker(article.get("mentioned_companies"))
            if ticker is None:
                counters["dropped_ticker_mapping"] += 1
                continue

            prev_day_price = _numeric_price(article, ticker, "prev_day_price")
            curr_day_price = _numeric_price(article, ticker, "curr_day_price")
            next_day_price = _numeric_price(article, ticker, "next_day_price")
            if prev_day_price is None or curr_day_price is None or next_day_price is None:
                counters["dropped_missing_prices"] += 1
                continue

            if next_day_price == curr_day_price:
                counters["dropped_zero_change"] += 1
                continue

            direction_label = int(next_day_price > curr_day_price)
            same_day_return = (curr_day_price - prev_day_price) / prev_day_price
            base_identifier = str(article.get("filename") or article.get("url") or title)

            records.append(
                {
                    "event_id": _event_id(base_identifier, ticker, publish_time),
                    "title": title,
                    "normalized_title": normalize_text(title),
                    "publication_timestamp": publish_time,
                    "publication_date": publish_time.date().isoformat(),
                    "ticker": ticker,
                    "prev_day_price": prev_day_price,
                    "curr_day_price": curr_day_price,
                    "next_day_price": next_day_price,
                    "same_day_return": same_day_return,
                    "label_id": direction_label,
                    "label_name": "up" if direction_label == 1 else "down",
                }
            )
            counters["kept_rows"] += 1

    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise RuntimeError("Stage 2 filtering produced no usable rows.")
    frame = frame.sort_values(["publication_timestamp", "event_id"]).reset_index(drop=True)
    return frame, dict(counters)


def _build_chronological_splits(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    grouped = (
        frame.groupby("publication_date", sort=True)
        .size()
        .reset_index(name="row_count")
        .sort_values("publication_date")
        .reset_index(drop=True)
    )
    total_rows = int(grouped["row_count"].sum())
    grouped["cumulative_rows"] = grouped["row_count"].cumsum()

    train_target = total_rows * STAGE2_TRAIN_RATIO
    validation_target = total_rows * (STAGE2_TRAIN_RATIO + STAGE2_VALIDATION_RATIO)

    train_cutoff_date = grouped[grouped["cumulative_rows"] >= train_target]["publication_date"].iloc[0]
    validation_cutoff_date = grouped[grouped["cumulative_rows"] >= validation_target]["publication_date"].iloc[0]

    train_mask = frame["publication_date"] <= train_cutoff_date
    validation_mask = (frame["publication_date"] > train_cutoff_date) & (frame["publication_date"] <= validation_cutoff_date)
    test_mask = frame["publication_date"] > validation_cutoff_date

    split_frames = {
        "train": frame[train_mask].reset_index(drop=True),
        "validation": frame[validation_mask].reset_index(drop=True),
        "test": frame[test_mask].reset_index(drop=True),
    }
    if any(split.empty for split in split_frames.values()):
        raise RuntimeError("Chronological Stage 2 split produced an empty split.")
    return split_frames


def _save_stage2_split_indices(split_frames: dict[str, pd.DataFrame]) -> None:
    split_paths = {
        "train": STAGE2_TRAIN_SPLIT_PATH,
        "validation": STAGE2_VALIDATION_SPLIT_PATH,
        "test": STAGE2_TEST_SPLIT_PATH,
    }
    for split_name, frame in split_frames.items():
        frame[["event_id", "publication_date"]].to_csv(split_paths[split_name], index=False)


def ensure_stage2_data_outputs() -> None:
    expected_paths = [
        STAGE2_DATA_SUMMARY_PATH,
        STAGE2_FILTERED_DATA_PATH,
        STAGE2_TRAIN_SPLIT_PATH,
        STAGE2_VALIDATION_SPLIT_PATH,
        STAGE2_TEST_SPLIT_PATH,
    ]
    if all(path.exists() for path in expected_paths):
        return
    run_stage2_data()


def load_stage2_dataset() -> pd.DataFrame:
    ensure_stage2_data_outputs()
    frame = pd.read_parquet(STAGE2_FILTERED_DATA_PATH)
    frame["publication_timestamp"] = pd.to_datetime(frame["publication_timestamp"])
    return frame


def load_stage2_splits() -> dict[str, pd.DataFrame]:
    frame = load_stage2_dataset()
    split_paths = {
        "train": STAGE2_TRAIN_SPLIT_PATH,
        "validation": STAGE2_VALIDATION_SPLIT_PATH,
        "test": STAGE2_TEST_SPLIT_PATH,
    }
    source = frame.set_index("event_id")
    split_frames: dict[str, pd.DataFrame] = {}
    for split_name, split_path in split_paths.items():
        split_ids = pd.read_csv(split_path)["event_id"].tolist()
        split_frames[split_name] = source.loc[split_ids].reset_index()
    return split_frames


def build_stage2_data_summary() -> dict[str, object]:
    ensure_project_dirs()
    frame, counters = _build_filtered_stage2_frame()
    split_frames = _build_chronological_splits(frame)
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(STAGE2_FILTERED_DATA_PATH, index=False)
    _save_stage2_split_indices(split_frames)

    date_ranges = {
        split_name: {
            "min_date": str(split_frame["publication_date"].min()),
            "max_date": str(split_frame["publication_date"].max()),
        }
        for split_name, split_frame in split_frames.items()
    }

    smoke_checks = {
        "filtered_dataset_non_empty": len(frame) > 0,
        "no_zero_change_rows": bool((frame["next_day_price"] != frame["curr_day_price"]).all()),
        "single_ticker_rows_only": bool(frame["ticker"].astype(str).str.len().gt(0).all()),
        "chronology_is_strict": date_ranges["train"]["max_date"] < date_ranges["validation"]["min_date"]
        and date_ranges["validation"]["max_date"] < date_ranges["test"]["min_date"],
        "split_total_matches": sum(len(split_frame) for split_frame in split_frames.values()) == len(frame),
    }
    if not all(smoke_checks.values()):
        failed = [name for name, passed in smoke_checks.items() if not passed]
        raise RuntimeError(f"Stage 2 data smoke checks failed: {failed}")

    summary = {
        "dataset_id": FINANCIAL_NEWS_DATASET_ID,
        "raw_file_count": len(list(FINANCIAL_NEWS_RAW_DIR.glob("*.json.xz"))),
        "filter_counts": counters,
        "final_row_count": int(len(frame)),
        "label_counts": {
            "down": int((frame["label_id"] == 0).sum()),
            "up": int((frame["label_id"] == 1).sum()),
        },
        "split_rule": {
            "type": "chronological_by_date_blocks",
            "train_ratio": STAGE2_TRAIN_RATIO,
            "validation_ratio": STAGE2_VALIDATION_RATIO,
            "test_ratio": STAGE2_TEST_RATIO,
            "seed": SEED,
        },
        "splits": {
            split_name: {
                "size": int(len(split_frame)),
                "label_counts": {
                    "down": int((split_frame["label_id"] == 0).sum()),
                    "up": int((split_frame["label_id"] == 1).sum()),
                },
                "date_range": date_ranges[split_name],
            }
            for split_name, split_frame in split_frames.items()
        },
        "smoke_checks": smoke_checks,
    }
    return summary


def run_stage2_data() -> dict[str, object]:
    summary = build_stage2_data_summary()
    write_json(STAGE2_DATA_SUMMARY_PATH, summary)
    return summary
