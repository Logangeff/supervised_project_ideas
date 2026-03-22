from __future__ import annotations

import argparse
import json

from .config import ensure_project_dirs
from .calibrated_surface_extension import (
    build_calibrated_surface,
    extract_calibrated_surface_inputs,
    train_calibrated_surface_extension,
)
from .data_access import WrdsUnavailableError, test_wrds_connection
from .option_pipeline import build_option_features, extract_option_data, train_part2
from .phase2_bucket_analysis import run_phase2_bucket_analysis
from .phase2_decision import run_phase2_decision
from .reporting import print_saved_results_overview
from .smoke import run_smoke_check
from .stock_pipeline import build_stock_panel, extract_stock_data, train_part1
from .surface_extension import build_surface_factors, train_surface_extension
from .text_news_extension import build_text_news_panel, train_text_news_extension


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WRDS + OptionMetrics realized-volatility regime project CLI")
    parser.add_argument(
        "--phase",
        required=True,
        choices=[
            "extract_stock_data",
            "build_stock_panel",
            "train_part1",
            "test_wrds_connection",
            "extract_option_data",
            "build_option_features",
            "train_part2",
            "build_surface_factors",
            "train_surface_extension",
            "extract_calibrated_surface_inputs",
            "build_calibrated_surface",
            "train_calibrated_surface_extension",
            "run_phase2_decision",
            "run_phase2_bucket_analysis",
            "build_text_news_panel",
            "train_text_news_extension",
            "smoke",
            "results",
            "all",
        ],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_project_dirs()
    try:
        if args.phase == "extract_stock_data":
            print(extract_stock_data())
            return
        if args.phase == "build_stock_panel":
            print(build_stock_panel())
            return
        if args.phase == "train_part1":
            print(train_part1())
            return
        if args.phase == "test_wrds_connection":
            print(json.dumps(test_wrds_connection(), indent=2))
            return
        if args.phase == "extract_option_data":
            print(extract_option_data())
            return
        if args.phase == "build_option_features":
            print(build_option_features())
            return
        if args.phase == "train_part2":
            print(train_part2())
            return
        if args.phase == "build_surface_factors":
            print(build_surface_factors())
            return
        if args.phase == "train_surface_extension":
            print(train_surface_extension())
            return
        if args.phase == "extract_calibrated_surface_inputs":
            print(extract_calibrated_surface_inputs())
            return
        if args.phase == "build_calibrated_surface":
            print(build_calibrated_surface())
            return
        if args.phase == "train_calibrated_surface_extension":
            print(train_calibrated_surface_extension())
            return
        if args.phase == "run_phase2_decision":
            print(run_phase2_decision())
            return
        if args.phase == "run_phase2_bucket_analysis":
            print(run_phase2_bucket_analysis())
            return
        if args.phase == "build_text_news_panel":
            print(build_text_news_panel())
            return
        if args.phase == "train_text_news_extension":
            print(train_text_news_extension())
            return
        if args.phase == "smoke":
            print(run_smoke_check())
            return
        if args.phase == "results":
            print(print_saved_results_overview())
            return
        if args.phase == "all":
            extract_stock_data()
            build_stock_panel()
            train_part1()
            extract_option_data()
            build_option_features()
            train_part2()
            print(print_saved_results_overview())
    except (RuntimeError, FileNotFoundError, WrdsUnavailableError) as exc:
        raise SystemExit(f"Error: {exc}") from exc


if __name__ == "__main__":
    main()
