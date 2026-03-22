from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import (
    PROJECT_SUMMARY_JSON,
    STAGE1_BEST_DETECTOR_PATH,
    STAGE1_CLASSICAL_SUMMARY_PATH,
    STAGE1_DATA_SUMMARY_PATH,
    STAGE1_EVALUATION_CSV,
    STAGE1_NEURAL_SUMMARY_PATH,
    STAGE3_AMPLITUDE_SUMMARY_PATH,
    STAGE2_DATA_SUMMARY_PATH,
    STAGE2_EVALUATION_CSV,
    STAGE2_MARKET_SUMMARY_PATH,
    STAGE2_TEXT_SUMMARY_PATH,
)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _format_metric_line(label: str, metrics: dict[str, object]) -> str:
    return (
        f"{label}: accuracy={float(metrics['accuracy']):.4f}, "
        f"macro_f1={float(metrics['macro_f1']):.4f}"
    )


def format_stage1_data_summary(summary: dict[str, object]) -> str:
    splits = summary["splits"]
    lines = [
        "Stage 1 Data",
        f"  Dataset: {summary['dataset_id']}",
        f"  Official splits: train={summary['official_splits']['train']}, test={summary['official_splits']['test']}",
        (
            "  Derived splits: "
            f"train={splits['train']['size']}, "
            f"validation={splits['validation']['size']}, "
            f"test={splits['test']['size']}"
        ),
        f"  Smoke checks: {'PASS' if all(summary['smoke_checks'].values()) else 'FAIL'}",
    ]
    return "\n".join(lines)


def format_stage1_model_summary(summary: dict[str, object]) -> str:
    lines = [
        f"Stage 1 {summary['model']}",
        f"  Validation: accuracy={float(summary['validation']['accuracy']):.4f}, macro_f1={float(summary['validation']['macro_f1']):.4f}",
        f"  Test:       accuracy={float(summary['test']['accuracy']):.4f}, macro_f1={float(summary['test']['macro_f1']):.4f}",
    ]
    if "best_c" in summary:
        lines.insert(1, f"  Best C: {summary['best_c']}")
    if "best_epoch" in summary:
        lines.insert(1, f"  Best epoch: {summary['best_epoch']} on {summary['device']}")
    return "\n".join(lines)


def format_stage1_evaluation_summary(summary: dict[str, object]) -> str:
    lines = ["Stage 1 Evaluation"]
    rows = pd.DataFrame(summary["rows"]).sort_values(["split", "macro_f1", "accuracy"], ascending=[True, False, False])
    for split_name in ["validation", "test"]:
        split_rows = rows[rows["split"] == split_name]
        if split_rows.empty:
            continue
        best = split_rows.iloc[0]
        lines.append(
            f"  Best on {split_name}: {best['model']} "
            f"(accuracy={best['accuracy']:.4f}, macro_f1={best['macro_f1']:.4f})"
        )
    best_detector = summary["best_detector"]
    lines.append(
        f"  Frozen detector for Stage 2: {best_detector['model_name']} "
        f"(validation macro_f1={best_detector['validation_macro_f1']:.4f})"
    )
    return "\n".join(lines)


def format_stage2_data_summary(summary: dict[str, object]) -> str:
    splits = summary["splits"]
    lines = [
        "Stage 2 Data",
        f"  Dataset: {summary['dataset_id']}",
        f"  Final usable rows: {summary['final_row_count']}",
        f"  Split sizes: train={splits['train']['size']}, validation={splits['validation']['size']}, test={splits['test']['size']}",
        (
            "  Date ranges: "
            f"train={splits['train']['date_range']['min_date']}..{splits['train']['date_range']['max_date']}, "
            f"validation={splits['validation']['date_range']['min_date']}..{splits['validation']['date_range']['max_date']}, "
            f"test={splits['test']['date_range']['min_date']}..{splits['test']['date_range']['max_date']}"
        ),
        f"  Smoke checks: {'PASS' if all(summary['smoke_checks'].values()) else 'FAIL'}",
    ]
    return "\n".join(lines)


def format_stage2_model_summary(summary: dict[str, object], title: str) -> str:
    lines = [title]
    if "market_only" in summary:
        ordered = ["market_only", "material_tone_only", "all_text", "claim_aware", "claim_finbert_aware"]
        for name in ordered:
            if name not in summary:
                continue
            metrics = summary[name]
            lines.append(
                f"  {name}: validation macro_f1={float(metrics['validation']['macro_f1']):.4f}, "
                f"test macro_f1={float(metrics['test']['macro_f1']):.4f}"
            )
        if "sentiment_model_id" in summary:
            lines.append(f"  sentiment_model: {summary['sentiment_model_id']}")
        return "\n".join(lines)

    lines.append(f"  Validation: accuracy={float(summary['validation']['accuracy']):.4f}, macro_f1={float(summary['validation']['macro_f1']):.4f}")
    lines.append(f"  Test:       accuracy={float(summary['test']['accuracy']):.4f}, macro_f1={float(summary['test']['macro_f1']):.4f}")
    return "\n".join(lines)


def format_project_summary(summary: dict[str, object]) -> str:
    stage1_rows = pd.DataFrame(summary["stage1_rows"])
    stage2_rows = pd.DataFrame(summary["stage2_rows"])
    stage1_best_test = stage1_rows[stage1_rows["split"] == "test"].sort_values(["macro_f1", "accuracy"], ascending=False).iloc[0]
    stage2_best_validation = stage2_rows[stage2_rows["split"] == "validation"].sort_values(["macro_f1", "accuracy"], ascending=False).iloc[0]
    stage2_best_test = stage2_rows[stage2_rows["split"] == "test"].sort_values(["macro_f1", "accuracy"], ascending=False).iloc[0]
    return "\n".join(
        [
            "Project Summary",
            (
                f"  Stage 1 best detector: {summary['stage1_best_detector']['model_name']} "
                f"(validation macro_f1={summary['stage1_best_detector']['validation_macro_f1']:.4f})"
            ),
            (
                f"  Stage 1 best test model: {stage1_best_test['model']} "
                f"(accuracy={stage1_best_test['accuracy']:.4f}, macro_f1={stage1_best_test['macro_f1']:.4f})"
            ),
            (
                f"  Stage 2 best validation model: {stage2_best_validation['model']} "
                f"(accuracy={stage2_best_validation['accuracy']:.4f}, macro_f1={stage2_best_validation['macro_f1']:.4f})"
            ),
            (
                f"  Stage 2 best test model: {stage2_best_test['model']} "
                f"(accuracy={stage2_best_test['accuracy']:.4f}, macro_f1={stage2_best_test['macro_f1']:.4f})"
            ),
        ]
    )


def format_stage3_amplitude_summary(summary: dict[str, object]) -> str:
    lines = [
        "Stage 3 Amplitude",
        (
            f"  Target: top {(1.0 - float(summary['target']['train_quantile'])) * 100:.0f}% "
            f"absolute next-day moves, threshold={float(summary['target']['threshold']):.6f}"
        ),
    ]
    for model_name in ["market_only_amp", "structured_logreg", "structured_mlp"]:
        if model_name not in summary["models"]:
            continue
        metrics = summary["models"][model_name]
        lines.append(
            f"  {model_name}: validation macro_f1={float(metrics['validation']['macro_f1']):.4f}, "
            f"test macro_f1={float(metrics['test']['macro_f1']):.4f}"
        )
    return "\n".join(lines)


def format_phase_output(phase: str, summary: dict[str, object]) -> str:
    if phase == "stage1_data":
        return format_stage1_data_summary(summary)
    if phase == "stage1_classical":
        return format_stage1_model_summary(summary)
    if phase == "stage1_neural":
        return format_stage1_model_summary(summary)
    if phase == "stage1_evaluate":
        return format_stage1_evaluation_summary(summary)
    if phase == "stage2_data":
        return format_stage2_data_summary(summary)
    if phase == "stage2_models":
        return format_stage2_model_summary(summary, "Stage 2 Model Training")
    if phase == "stage2_evaluate":
        return format_project_summary(summary)
    if phase == "stage3_amplitude":
        return format_stage3_amplitude_summary(summary)
    if phase == "smoke":
        return "\n".join(
            [
                "Smoke Check",
                f"  Status: {summary['status']}",
                f"  Stage 1 validation macro_f1: {summary['stage1_validation_macro_f1']:.4f}",
                f"  Stage 2 final row count: {summary['stage2_final_row_count']}",
                f"  Checks: {'PASS' if all(summary['checks'].values()) else 'FAIL'}",
            ]
        )
    if phase == "all":
        return format_project_summary(summary)
    return json.dumps(summary, indent=2)


def print_saved_results_overview() -> str:
    sections: list[str] = []
    if STAGE1_DATA_SUMMARY_PATH.exists():
        sections.append(format_stage1_data_summary(_read_json(STAGE1_DATA_SUMMARY_PATH)))
    if STAGE1_CLASSICAL_SUMMARY_PATH.exists():
        sections.append(format_stage1_model_summary(_read_json(STAGE1_CLASSICAL_SUMMARY_PATH)))
    if STAGE1_NEURAL_SUMMARY_PATH.exists():
        sections.append(format_stage1_model_summary(_read_json(STAGE1_NEURAL_SUMMARY_PATH)))
    if STAGE1_EVALUATION_CSV.exists() and STAGE1_BEST_DETECTOR_PATH.exists():
        sections.append(format_stage1_evaluation_summary({
            "rows": pd.read_csv(STAGE1_EVALUATION_CSV).to_dict(orient="records"),
            "best_detector": _read_json(STAGE1_BEST_DETECTOR_PATH),
        }))
    if STAGE2_DATA_SUMMARY_PATH.exists():
        sections.append(format_stage2_data_summary(_read_json(STAGE2_DATA_SUMMARY_PATH)))
    if STAGE2_MARKET_SUMMARY_PATH.exists() and STAGE2_TEXT_SUMMARY_PATH.exists():
        sections.append(
            format_stage2_model_summary(
                {
                    "market_only": _read_json(STAGE2_MARKET_SUMMARY_PATH),
                    **_read_json(STAGE2_TEXT_SUMMARY_PATH),
                },
                "Stage 2 Model Training",
            )
        )
    if PROJECT_SUMMARY_JSON.exists():
        sections.append(format_project_summary(_read_json(PROJECT_SUMMARY_JSON)))
    if STAGE3_AMPLITUDE_SUMMARY_PATH.exists():
        sections.append(format_stage3_amplitude_summary(_read_json(STAGE3_AMPLITUDE_SUMMARY_PATH)))
    return "\n\n".join(sections)
