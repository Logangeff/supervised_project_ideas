from __future__ import annotations

from .config import SMOKE_SUMMARY_PATH
from .dashboard import build_dashboard_payload
from .macro_pipeline import build_cycle_labels, build_monthly_panel, fetch_macro_data
from .modeling import train_phase_forecasts, train_recession_risk
from .utils import save_json


def run_smoke_check() -> dict[str, object]:
    fetch_summary = fetch_macro_data(smoke=True)
    panel_summary = build_monthly_panel(smoke=True)
    label_summary = build_cycle_labels(smoke=True)
    phase_summary = train_phase_forecasts()
    risk_summary = train_recession_risk()
    dashboard_summary = build_dashboard_payload()
    summary = {
        "fetched_series_count": fetch_summary["series_count"],
        "panel_rows": panel_summary["rows"],
        "label_positive_rate": label_summary["recession_within_6m_positive_rate"],
        "phase_selected_model": phase_summary["selected_model"],
        "risk_selected_model": risk_summary["selected_model"],
        "dashboard_history_rows": dashboard_summary["history_rows"],
    }
    save_json(SMOKE_SUMMARY_PATH, summary)
    return summary
