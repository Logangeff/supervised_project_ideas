# Numerical Claim Detection Project

This project implements the two-stage numerical claim detection pipeline defined in `plans/PROJECT_DIRECTION.md`.

The implementation rule is:

- `plans/PROJECT_DIRECTION.md` is the build spec
- `plans/study_plan_numerical_claims.pdf` is the teacher-facing summary

## What is in scope

Stage 1:
- load `gtfintechlab/Numclaim`
- preserve the official test split
- derive a fixed validation split from the official train split
- train:
  - TF-IDF + Logistic Regression
  - GRU
- evaluate both and freeze the best Stage 1 detector

Stage 2:
- load `luckycat37/financial-news-dataset`
- filter to usable single-ticker news events
- build a binary next-day direction label
- train:
  - market-only baseline
  - all-text baseline
  - claim-aware baseline using the frozen Stage 1 claim probability

## Environment setup

From the project root:

```bat
cd numerical_claim_detection_project
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Note:
- `requirements.txt` pins `numpy<2` because the GRU path uses PyTorch 2.3.1 and the previous environment produced NumPy 2 runtime warnings.
- If you want a CUDA-enabled PyTorch build for the GRU, install the appropriate Torch wheel for your machine after activating the environment.

## Main commands

Direct Python CLI:

```bat
python -m src.main --phase stage1_data
python -m src.main --phase stage1_classical
python -m src.main --phase stage1_neural
python -m src.main --phase stage1_evaluate
python -m src.main --phase stage2_data
python -m src.main --phase stage2_models
python -m src.main --phase stage2_evaluate
python -m src.main --phase results
python -m src.main --phase smoke
python -m src.main --phase all
```

Convenience batch runner:

```bat
run_project.bat all
run_project.bat results
run_project.bat smoke
```

## What each phase does

`stage1_data`
- downloads and prepares NumClaim
- creates fixed Stage 1 train/validation/test artifacts
- writes `outputs/summaries/stage1_data_summary.json`

`stage1_classical`
- trains TF-IDF + Logistic Regression
- writes the model, metrics, and confusion matrix

`stage1_neural`
- trains the GRU baseline
- writes the checkpoint, metrics, and confusion matrix

`stage1_evaluate`
- compares Stage 1 models
- selects the frozen Stage 1 detector for Stage 2

`stage2_data`
- downloads and filters the financial-news dataset
- creates chronological Stage 2 splits
- writes `outputs/summaries/stage2_data_summary.json`

`stage2_models`
- trains market-only, all-text, and claim-aware Stage 2 models

`stage2_evaluate`
- writes final Stage 2 summary tables
- writes `outputs/metrics/project_summary.json`

`results`
- prints a compact summary of the current saved results

`smoke`
- runs the minimal reproducibility path:
  - `stage1_data`
  - `stage1_classical`
  - `stage2_data`

`all`
- runs the full pipeline end to end

## Main outputs

Important files:

- `outputs/summaries/stage1_data_summary.json`
- `outputs/summaries/stage2_data_summary.json`
- `outputs/metrics/stage1_evaluation_summary.csv`
- `outputs/metrics/stage2_evaluation_summary.csv`
- `outputs/metrics/project_summary.json`
- `outputs/figures/`

## Reproducibility notes

- The seed is fixed to `42`.
- Stage 1 always preserves the official NumClaim test split.
- Stage 2 always uses chronological date-based splits.
- Saved artifact metadata now uses project-relative paths rather than machine-specific absolute paths.
