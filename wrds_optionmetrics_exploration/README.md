# WRDS + OptionMetrics Exploration

This project implements a 2-part supervised pipeline for **realized-volatility regime classification**:

- **Part 1:** stock-only panel and baselines
- **Part 2:** OptionMetrics feature increment and complete-case comparison
- **Extension:** daily implied-volatility surface descriptors built from the cached option data
- **Calibrated extension:** forward-based, OTM-only 5-beta daily surface calibration inspired by the earlier `tp2` project, but implemented with the current caching and panel logic
- **Phase 2 decision layer:** probability-scaled risk overlay that converts the high-volatility probabilities into next-day exposure scaling
- **Phase 2 bucket analysis:** cross-sectional bucket sorting that tests whether higher predicted risk maps to higher future realized volatility and hit rates
- **Text/news extension:** a restricted 2020-2023 news-covered subpanel that tests whether Yahoo Finance article intensity, sentiment, and claim-like materiality add value beyond stock + option + surface features

## Setup

```bat
cd wrds_optionmetrics_exploration
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## WRDS access

The extraction phases use the Python `wrds` client if it is installed and credentials are available.

You can provide credentials with environment variables:

```bat
set WRDS_USERNAME=your_username
set WRDS_PASSWORD=your_password
```

Or place them in a local file that is ignored by git:

```text
config/wrds_credentials.json
```

with this shape:

```json
{
  "username": "your_wrds_username",
  "password": "your_wrds_password"
}
```

Start from:

```text
config/wrds_credentials.template.json
```

If raw parquet files already exist under `data/raw/stock` or `data/raw/options`, the pipeline will reuse them and skip WRDS extraction.

## Main phases

```bat
python -m src.main --phase extract_stock_data
python -m src.main --phase build_stock_panel
python -m src.main --phase train_part1
python -m src.main --phase test_wrds_connection
python -m src.main --phase extract_option_data
python -m src.main --phase build_option_features
python -m src.main --phase train_part2
python -m src.main --phase build_surface_factors
python -m src.main --phase train_surface_extension
python -m src.main --phase extract_calibrated_surface_inputs
python -m src.main --phase build_calibrated_surface
python -m src.main --phase train_calibrated_surface_extension
python -m src.main --phase run_phase2_decision
python -m src.main --phase run_phase2_bucket_analysis
python -m src.main --phase build_text_news_panel
python -m src.main --phase train_text_news_extension
python -m src.main --phase smoke
python -m src.main --phase results
python -m src.main --phase all
```

## Outputs

- `outputs/summaries/extract_stock_data_summary.json`
- `outputs/summaries/build_stock_panel_summary.json`
- `outputs/metrics/part1_metrics.json`
- `outputs/summaries/extract_option_data_summary.json`
- `outputs/summaries/build_option_features_summary.json`
- `outputs/metrics/part2_metrics.json`
- `outputs/summaries/build_surface_factors_summary.json`
- `outputs/metrics/surface_extension_metrics.json`
- `outputs/summaries/extract_calibrated_surface_inputs_summary.json`
- `outputs/summaries/build_calibrated_surface_summary.json`
- `outputs/metrics/calibrated_surface_extension_metrics.json`
- `outputs/metrics/phase2_decision_metrics.json`
- `outputs/metrics/phase2_decision_daily_returns.csv`
- `outputs/metrics/phase2_bucket_analysis_metrics.json`
- `outputs/metrics/phase2_bucket_analysis_diagnostics.csv`
- `outputs/summaries/build_text_news_panel_summary.json`
- `outputs/metrics/text_news_extension_metrics.json`
- `outputs/metrics/text_news_extension_predictions.csv`
- `outputs/summaries/smoke_summary.json`

## Bucket Dashboard

The project includes a read-only Streamlit dashboard for the successful Phase 2 bucket-analysis branch.

```bat
cd wrds_optionmetrics_exploration
launch_bucket_dashboard.bat
```

or

```bat
python -m streamlit run app/streamlit_bucket_dashboard.py
```

## Notes

- The universe file is in `data/universe/current_sp100_like_tickers.csv`.
- The project uses a static current large-cap universe on purpose.
- The WRDS schema can differ by subscription and local access. If a table name differs from the defaults, update `src/config.py`.
- The surface extension reuses the existing cached yearly option parquet files; it does not repull WRDS data if the raw files are already present.
- The calibrated-surface extension is separate from the finished v1 result. It adds forward-based moneyness, OTM filtering, and daily 5-beta calibration while keeping the original Part 1 / Part 2 outputs unchanged.
- The Phase 2 decision layer reuses saved prediction files and the cached stock panel. It does not repull WRDS data.
- The Phase 2 bucket analysis also reuses only saved prediction files and the cached stock panel. It is meant to test sorting usefulness even when the portfolio overlays are not compelling.
- The text/news extension reuses the local Yahoo Finance article archive already present in the repository. It does not repull WRDS or scrape fresh news.
