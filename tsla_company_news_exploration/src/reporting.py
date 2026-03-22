from __future__ import annotations

import json
from pathlib import Path

from .config import (
    AMPLITUDE_SUMMARY_PATH,
    COLLECTION_SUMMARY_PATH,
    DATASET_SUMMARY_PATH,
    MATERIALITY_SUMMARY_PATH,
    STAGE2_DIRECTION_SUMMARY_PATH,
    STAGE3_SENTIMENT_SUMMARY_PATH,
)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def format_collection_summary(summary: dict[str, object]) -> str:
    return "\n".join(
        [
            "Collection",
            f"  Price rows: {summary['price_history_rows']}",
            f"  Price range: {summary['price_date_range']['min_date']}..{summary['price_date_range']['max_date']}",
            f"  Headline rows: {summary['headline_rows']}",
            f"  Headline range: {summary['headline_date_range']['min_date']}..{summary['headline_date_range']['max_date']}",
        ]
    )


def format_dataset_summary(summary: dict[str, object]) -> str:
    return "\n".join(
        [
            "Daily Dataset",
            f"  Rows: {summary['row_count']}",
            f"  Date range: {summary['trade_date_range']['min_date']}..{summary['trade_date_range']['max_date']}",
            f"  Headline days: {summary['headline_days']}",
            f"  Empty-headline days: {summary['empty_headline_days']}",
        ]
    )


def format_materiality_summary(summary: dict[str, object]) -> str:
    return "\n".join(
        [
            "Stage 1 Materiality",
            f"  Headline rows scored: {summary['headline_rows']}",
            f"  Daily rows with claim signal: {summary['daily_rows_with_claim_signal']}",
            f"  Mean claim probability: {summary['headline_claim_prob_mean']:.4f}",
        ]
    )


def format_stage2_summary(summary: dict[str, object]) -> str:
    lines = ["Stage 2 Direction"]
    for model_name in ["market_only", "full_text", "claim_aware"]:
        metrics = summary["models"][model_name]
        lines.append(
            f"  {model_name}: validation macro_f1={metrics['validation']['macro_f1']:.4f}, "
            f"test macro_f1={metrics['test']['macro_f1']:.4f}"
        )
    return "\n".join(lines)


def format_stage3_summary(summary: dict[str, object]) -> str:
    model = summary["model"]
    return "\n".join(
        [
            "Stage 3 Sentiment",
            f"  claim_sentiment_aware: validation macro_f1={model['validation']['macro_f1']:.4f}, "
            f"test macro_f1={model['test']['macro_f1']:.4f}",
        ]
    )


def format_amplitude_summary(summary: dict[str, object]) -> str:
    if summary.get("skipped"):
        return f"Amplitude\n  Skipped: {summary['reason']}"
    lines = ["Amplitude", f"  Threshold: {summary['threshold']:.6f}"]
    for model_name in ["market_only_amp", "claim_sentiment_structured_amp"]:
        metrics = summary["models"][model_name]
        lines.append(
            f"  {model_name}: validation macro_f1={metrics['validation']['macro_f1']:.4f}, "
            f"test macro_f1={metrics['test']['macro_f1']:.4f}"
        )
    return "\n".join(lines)


def print_saved_results_overview() -> str:
    sections: list[str] = []
    if COLLECTION_SUMMARY_PATH.exists():
        sections.append(format_collection_summary(_read_json(COLLECTION_SUMMARY_PATH)))
    if DATASET_SUMMARY_PATH.exists():
        sections.append(format_dataset_summary(_read_json(DATASET_SUMMARY_PATH)))
    if MATERIALITY_SUMMARY_PATH.exists():
        sections.append(format_materiality_summary(_read_json(MATERIALITY_SUMMARY_PATH)))
    if STAGE2_DIRECTION_SUMMARY_PATH.exists():
        sections.append(format_stage2_summary(_read_json(STAGE2_DIRECTION_SUMMARY_PATH)))
    if STAGE3_SENTIMENT_SUMMARY_PATH.exists():
        sections.append(format_stage3_summary(_read_json(STAGE3_SENTIMENT_SUMMARY_PATH)))
    if AMPLITUDE_SUMMARY_PATH.exists():
        sections.append(format_amplitude_summary(_read_json(AMPLITUDE_SUMMARY_PATH)))
    return "\n\n".join(sections)
