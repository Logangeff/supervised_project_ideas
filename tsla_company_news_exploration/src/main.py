from __future__ import annotations

import argparse

from .data_pipeline import build_daily_dataset, run_collect_data
from .models import run_stage1_materiality, run_stage2_direction, run_stage3_amplitude, run_stage3_sentiment
from .reporting import print_saved_results_overview


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TSLA company news exploration CLI")
    parser.add_argument(
        "--phase",
        required=True,
        choices=[
            "collect_data",
            "build_dataset",
            "stage1_materiality",
            "stage2_direction",
            "stage3_sentiment",
            "stage3_amplitude",
            "results",
            "all",
        ],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.phase == "collect_data":
        print(run_collect_data())
        return
    if args.phase == "build_dataset":
        print(build_daily_dataset())
        return
    if args.phase == "stage1_materiality":
        print(run_stage1_materiality())
        return
    if args.phase == "stage2_direction":
        print(run_stage2_direction())
        return
    if args.phase == "stage3_sentiment":
        print(run_stage3_sentiment())
        return
    if args.phase == "stage3_amplitude":
        print(run_stage3_amplitude())
        return
    if args.phase == "results":
        print(print_saved_results_overview())
        return
    if args.phase == "all":
        run_collect_data()
        build_daily_dataset()
        run_stage1_materiality()
        run_stage2_direction()
        run_stage3_sentiment()
        run_stage3_amplitude()
        print(print_saved_results_overview())
        return


if __name__ == "__main__":
    main()
