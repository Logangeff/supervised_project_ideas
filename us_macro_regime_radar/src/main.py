from __future__ import annotations

import argparse
import json

from .config import ensure_project_dirs
from .dashboard import build_dashboard_payload
from .macro_pipeline import build_cycle_labels, build_monthly_panel, fetch_macro_data
from .modeling import train_phase_forecasts, train_recession_risk
from .reporting import print_saved_results_overview
from .smoke import run_smoke_check


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="US Macro Regime Radar CLI")
    parser.add_argument(
        "--phase",
        required=True,
        choices=[
            "fetch_macro_data",
            "build_monthly_panel",
            "build_cycle_labels",
            "train_phase_forecasts",
            "train_recession_risk",
            "build_dashboard_payload",
            "results",
            "smoke",
            "all",
        ],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_project_dirs()
    try:
        if args.phase == "fetch_macro_data":
            print(fetch_macro_data())
            return
        if args.phase == "build_monthly_panel":
            print(build_monthly_panel())
            return
        if args.phase == "build_cycle_labels":
            print(build_cycle_labels())
            return
        if args.phase == "train_phase_forecasts":
            print(train_phase_forecasts())
            return
        if args.phase == "train_recession_risk":
            print(train_recession_risk())
            return
        if args.phase == "build_dashboard_payload":
            print(build_dashboard_payload())
            return
        if args.phase == "smoke":
            print(json.dumps(run_smoke_check(), indent=2))
            return
        if args.phase == "results":
            print(print_saved_results_overview())
            return
        if args.phase == "all":
            fetch_macro_data()
            build_monthly_panel()
            build_cycle_labels()
            train_phase_forecasts()
            train_recession_risk()
            build_dashboard_payload()
            print(print_saved_results_overview())
            return
    except Exception as exc:  # pragma: no cover - CLI surface
        raise SystemExit(f"Error: {exc}") from exc


if __name__ == "__main__":
    main()
