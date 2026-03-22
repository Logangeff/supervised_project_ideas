from __future__ import annotations

import argparse

from .stage1_data import run_stage1_data
from .stage1_models import run_stage1_classical, run_stage1_evaluate, run_stage1_neural
from .stage2_data import run_stage2_data
from .stage2_models import run_stage2_evaluate, run_stage2_models
from .reporting import format_phase_output, print_saved_results_overview
from .smoke import run_smoke_check


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Numerical claim detection project CLI")
    parser.add_argument(
        "--phase",
        required=True,
        choices=[
            "stage1_data",
            "stage1_classical",
            "stage1_neural",
            "stage1_evaluate",
            "stage2_data",
            "stage2_models",
            "stage2_evaluate",
            "results",
            "smoke",
            "all",
        ],
        help="Project phase to execute.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.phase == "stage1_data":
        summary = run_stage1_data()
        print(format_phase_output("stage1_data", summary))
        return

    if args.phase == "stage1_classical":
        summary = run_stage1_classical()
        print(format_phase_output("stage1_classical", summary))
        return

    if args.phase == "stage1_neural":
        summary = run_stage1_neural()
        print(format_phase_output("stage1_neural", summary))
        return

    if args.phase == "stage1_evaluate":
        summary = run_stage1_evaluate()
        print(format_phase_output("stage1_evaluate", summary))
        return

    if args.phase == "stage2_data":
        summary = run_stage2_data()
        print(format_phase_output("stage2_data", summary))
        return

    if args.phase == "stage2_models":
        summary = run_stage2_models()
        print(format_phase_output("stage2_models", summary))
        return

    if args.phase == "stage2_evaluate":
        summary = run_stage2_evaluate()
        print(format_phase_output("stage2_evaluate", summary))
        return

    if args.phase == "results":
        print(print_saved_results_overview())
        return

    if args.phase == "smoke":
        summary = run_smoke_check()
        print(format_phase_output("smoke", summary))
        return

    if args.phase == "all":
        run_stage1_data()
        run_stage1_classical()
        run_stage1_neural()
        run_stage1_evaluate()
        run_stage2_data()
        run_stage2_models()
        summary = run_stage2_evaluate()
        print(format_phase_output("all", summary))
        return


if __name__ == "__main__":
    main()
