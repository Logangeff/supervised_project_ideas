# WRDS + OptionMetrics Exploration

This project asks a focused supervised-learning question:

> Do option-implied features improve prediction of future stock-level volatility stress beyond stock-only features?

The main target is whether a stock enters a **high-volatility regime over the next 20 trading days**. The strongest result is not the portfolio overlay branch. It is the finding that **option-aware probabilities are materially better at ranking future stock-level risk**, which makes them useful for cross-sectional screening and risk triage.

## Main Story

- **Target:** future 20-day high-volatility regime
- **Data:** CRSP-style stock panel + OptionMetrics option panel + daily surface descriptors
- **Main model:** fixed-split logistic regression
- **Benchmarks:** XGBoost, 2-year trailing retrain, 5-year trailing retrain
- **Main usefulness result:** Phase 2 bucket analysis shows that higher predicted-risk buckets consistently map to higher future realized volatility and higher high-volatility hit rates

## Headline Result

On the common test panel:

- `Stock Only` is meaningfully weaker
- `Option Only` is much stronger
- `Stock + Option + Surface` is the clean headline result

For the practical Phase 2 ranking use case:

- `Stock + Option + Surface` average daily rank IC is about `0.733`
- top-minus-bottom high-volatility hit-rate spread is about `61.7%`
- XGBoost helps the weak stock-only branch, but does **not** clearly beat the original option-driven result
- 2Y and 5Y trailing retrains are essentially ties on the strongest option-aware signals

## Dashboard

Launch the professor-facing dashboard from the project root:

```bat
cd wrds_optionmetrics_exploration
launch_bucket_dashboard.bat
```

or directly:

```bat
python -m streamlit run app/streamlit_bucket_dashboard.py
```

The dashboard is organized as:

1. `Overview`
2. `Core Result`
3. `Benchmark Comparison`
4. `Single-Stock Drill-Down`
5. `Data Quality`
6. `Appendix`

## Minimal Setup

```bat
cd wrds_optionmetrics_exploration
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## WRDS Access

The extraction phases use the Python `wrds` client if credentials are available.

Environment variables:

```bat
set WRDS_USERNAME=your_username
set WRDS_PASSWORD=your_password
```

Or a local ignored file:

```text
config/wrds_credentials.json
```

starting from:

```text
config/wrds_credentials.template.json
```

If cached parquet files already exist under `data/raw/stock` and `data/raw/options`, the project reuses them and skips extraction.

## Main Phases

```bat
python -m src.main --phase extract_stock_data
python -m src.main --phase build_stock_panel
python -m src.main --phase extract_option_data
python -m src.main --phase build_option_features
python -m src.main --phase build_surface_factors
python -m src.main --phase train_surface_extension
python -m src.main --phase train_calibrated_surface_extension
python -m src.main --phase run_phase2_bucket_analysis
python -m src.main --phase results
python -m src.main --phase all
```

## Important Saved Outputs

- `outputs/metrics/surface_extension_metrics.csv`
- `outputs/metrics/surface_extension_predictions.csv`
- `outputs/metrics/phase2_bucket_analysis_metrics.csv`
- `outputs/metrics/phase2_bucket_analysis_diagnostics.csv`
- `outputs/summaries/extract_stock_data_summary.json`
- `outputs/summaries/build_stock_panel_summary.json`
- `outputs/summaries/extract_option_data_summary.json`
- `outputs/summaries/build_option_features_summary.json`
- `outputs/summaries/build_surface_factors_summary.json`

## Scope Notes

- The fixed chronological train / validation / test split remains the **headline evaluation design**.
- XGBoost and trailing retrains are kept as **robustness benchmarks**, not replacements.
- The calibrated-surface branch remains in the repo as a secondary extension.
- The portfolio overlay and text/news branches remain in the repo, but they are **not** the default presentation path because they were not the strongest results.
