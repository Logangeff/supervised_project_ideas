from __future__ import annotations

import json

from .classical_models import train_classical_models
from .config import (
    NOSIBLE_CLASSICAL_MODEL_FILENAMES,
    NOSIBLE_CLASSICAL_TRAINING_SUMMARY_PATH,
    NOSIBLE_EVALUATION_SUMMARY_CSV,
    NOSIBLE_EVALUATION_SUMMARY_JSON,
    NOSIBLE_GRU_MODEL_PATH,
    NOSIBLE_GRU_TRAINING_SUMMARY_PATH,
    ensure_output_dirs,
)
from .data_pipeline import load_nosible_experiment_data, run_nosible_phase1, set_global_seed
from .evaluate import evaluate_saved_models
from .neural_model import train_gru_model


def run_nosible_experiment() -> dict[str, object]:
    ensure_output_dirs()
    set_global_seed()
    phase1_summary = run_nosible_phase1()
    experiment_data = load_nosible_experiment_data()
    nosible = experiment_data["nosible"]

    classical_summary = train_classical_models(
        train_frame=nosible["train"],
        validation_frame=nosible["validation"],
        lm_vocabulary=experiment_data["lm_vocabulary"],
        model_filenames=NOSIBLE_CLASSICAL_MODEL_FILENAMES,
        summary_path=NOSIBLE_CLASSICAL_TRAINING_SUMMARY_PATH,
    )

    gru_summary = train_gru_model(
        train_frame=nosible["train"],
        validation_frame=nosible["validation"],
        model_path=NOSIBLE_GRU_MODEL_PATH,
        summary_path=NOSIBLE_GRU_TRAINING_SUMMARY_PATH,
    )

    evaluation_summary = evaluate_saved_models(
        model_filenames=NOSIBLE_CLASSICAL_MODEL_FILENAMES,
        gru_model_path=NOSIBLE_GRU_MODEL_PATH,
        datasets={
            "nosible_validation": nosible["validation"],
            "nosible_test": nosible["test"],
            "phrasebank_test": experiment_data["phrasebank"]["test"],
            "fiqa_test": experiment_data["fiqa_test"],
        },
        output_json=NOSIBLE_EVALUATION_SUMMARY_JSON,
        output_csv=NOSIBLE_EVALUATION_SUMMARY_CSV,
        figure_prefix="nosible_",
    )

    return {
        "nosible_phase1": phase1_summary,
        "nosible_classical": classical_summary,
        "nosible_gru": gru_summary,
        "nosible_evaluation": evaluation_summary,
    }
