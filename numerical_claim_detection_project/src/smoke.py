from __future__ import annotations

from .config import STAGE1_CLASSICAL_MODEL_PATH, STAGE1_DATA_SUMMARY_PATH, STAGE2_DATA_SUMMARY_PATH
from .stage1_data import run_stage1_data
from .stage1_models import run_stage1_classical
from .stage2_data import run_stage2_data


def run_smoke_check() -> dict[str, object]:
    stage1_data_summary = run_stage1_data()
    stage1_classical_summary = run_stage1_classical()
    stage2_data_summary = run_stage2_data()

    checks = {
        "stage1_data_summary_exists": STAGE1_DATA_SUMMARY_PATH.exists(),
        "stage1_classical_model_exists": STAGE1_CLASSICAL_MODEL_PATH.exists(),
        "stage2_data_summary_exists": STAGE2_DATA_SUMMARY_PATH.exists(),
        "stage1_data_smoke_checks_pass": all(stage1_data_summary["smoke_checks"].values()),
        "stage2_data_smoke_checks_pass": all(stage2_data_summary["smoke_checks"].values()),
    }
    return {
        "stage1_validation_macro_f1": float(stage1_classical_summary["validation"]["macro_f1"]),
        "stage2_final_row_count": int(stage2_data_summary["final_row_count"]),
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
