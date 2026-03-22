from __future__ import annotations

import argparse
import json

from .classical_models import run_classical_phase
from .data_pipeline import run_phase1
from .evaluate import run_evaluation_phase
from .neural_model import run_neural_phase
from .nosible_experiment import run_nosible_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Financial sentiment course project CLI")
    parser.add_argument(
        "--phase",
        required=True,
        choices=["phase1", "classical", "neural", "evaluate", "all", "nosible", "raw_finbert", "finbert_finetuned"],
        help="Project phase to execute.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.phase == "phase1":
        summary = run_phase1()
        print(json.dumps(summary, indent=2))
        return

    if args.phase == "classical":
        summary = run_classical_phase()
        print(json.dumps(summary, indent=2))
        return

    if args.phase == "neural":
        summary = run_neural_phase()
        print(json.dumps(summary, indent=2))
        return

    if args.phase == "evaluate":
        summary = run_evaluation_phase()
        print(json.dumps(summary, indent=2))
        return

    if args.phase == "all":
        run_phase1()
        run_classical_phase()
        run_neural_phase()
        summary = run_evaluation_phase()
        print(json.dumps(summary, indent=2))
        return

    if args.phase == "nosible":
        summary = run_nosible_experiment()
        print(json.dumps(summary, indent=2))
        return

    if args.phase == "raw_finbert":
        from .transformer_models import run_raw_finbert_phase

        summary = run_raw_finbert_phase()
        print(json.dumps(summary, indent=2))
        return

    if args.phase == "finbert_finetuned":
        from .transformer_models import run_finbert_finetuned_phase

        summary = run_finbert_finetuned_phase()
        print(json.dumps(summary, indent=2))
        return


if __name__ == "__main__":
    main()
