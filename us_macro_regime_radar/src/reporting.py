from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import (
    BUILD_CYCLE_LABELS_SUMMARY_PATH,
    BUILD_MONTHLY_PANEL_SUMMARY_PATH,
    FETCH_MACRO_DATA_SUMMARY_PATH,
    FORECAST_PAYLOAD_JSON,
    HISTORY_PAYLOAD_JSON,
    LATEST_SNAPSHOT_JSON,
    PHASE_FORECAST_METRICS_JSON,
    RECESSION_RISK_METRICS_JSON,
    SMOKE_SUMMARY_PATH,
)
from .utils import load_json


def print_saved_results_overview() -> str:
    sections: list[str] = []

    fetch_summary = load_json(FETCH_MACRO_DATA_SUMMARY_PATH)
    if fetch_summary:
        sections.append(
            "Fetch Macro Data\n"
            f"  Series count: {fetch_summary.get('series_count')}\n"
            f"  API key used: {fetch_summary.get('api_key_used')}"
        )

    panel_summary = load_json(BUILD_MONTHLY_PANEL_SUMMARY_PATH)
    if panel_summary:
        sections.append(
            "Monthly Panel\n"
            f"  Rows: {panel_summary.get('rows')}\n"
            f"  Date range: {panel_summary.get('date_min')} to {panel_summary.get('date_max')}\n"
            f"  Feature count: {panel_summary.get('feature_count')}"
        )

    label_summary = load_json(BUILD_CYCLE_LABELS_SUMMARY_PATH)
    if label_summary:
        sections.append(
            "Cycle Labels\n"
            f"  Phase counts: {label_summary.get('phase_counts')}\n"
            f"  Recession within 6m positive rate: {float(label_summary.get('recession_within_6m_positive_rate')):.4f}"
        )

    phase_metrics = load_json(PHASE_FORECAST_METRICS_JSON)
    if phase_metrics:
        selected = phase_metrics.get("selected_model")
        lines = [f"  Selected model: {selected}"]
        for row in phase_metrics.get("validation_selection", [])[:4]:
            lines.append(
                f"  {row['model']}: validation macro_f1={row['macro_f1']:.4f}, "
                f"balanced_accuracy={row['balanced_accuracy']:.4f}"
            )
        sections.append("Phase Forecasts\n" + "\n".join(lines))

    risk_metrics = load_json(RECESSION_RISK_METRICS_JSON)
    if risk_metrics:
        selected = risk_metrics.get("selected_model")
        lines = [f"  Selected model: {selected}"]
        for row in risk_metrics.get("validation_selection", [])[:3]:
            lines.append(
                f"  {row['model']}: validation pr_auc={row['pr_auc']:.4f}, "
                f"brier_score={row['brier_score']:.4f}"
            )
        calibrated_test = risk_metrics.get("selected_calibrated", {}).get("test", {})
        if calibrated_test:
            lines.append(
                f"  Calibrated test: pr_auc={calibrated_test.get('pr_auc'):.4f}, "
                f"auroc={calibrated_test.get('auroc'):.4f}, "
                f"brier={calibrated_test.get('brier_score'):.4f}"
            )
        sections.append("Recession Risk\n" + "\n".join(lines))

    latest_snapshot = load_json(LATEST_SNAPSHOT_JSON)
    if latest_snapshot:
        sections.append(
            "Latest Snapshot\n"
            f"  Date: {latest_snapshot.get('date')}\n"
            f"  Current phase: {latest_snapshot.get('current_phase')}\n"
            f"  Recession within 6m probability: {float(latest_snapshot.get('recession_within_6m_probability')):.4f}\n"
            f"  Most likely next phase: {latest_snapshot.get('most_likely_next_phase')}"
        )

    forecast_payload = load_json(FORECAST_PAYLOAD_JSON)
    if forecast_payload:
        lines = []
        for row in forecast_payload.get("forecasts", [])[:3]:
            lines.append(
                f"  h={row['horizon_months']}m: {row['most_likely_phase']} "
                f"(confidence={row['confidence']:.4f})"
            )
        sections.append("Forward Path\n" + "\n".join(lines))

    smoke = load_json(SMOKE_SUMMARY_PATH)
    if smoke:
        sections.append(
            "Smoke\n"
            f"  Fetched series: {smoke.get('fetched_series_count')}\n"
            f"  Panel rows: {smoke.get('panel_rows')}\n"
            f"  Risk selected model: {smoke.get('risk_selected_model')}"
        )

    return "\n\n".join(sections) if sections else "No saved results yet."
