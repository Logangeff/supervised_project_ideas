# US Macro Regime Radar

This project builds a **U.S.-first monthly macro regime engine** with two linked outputs:

- a **current 4-phase business-cycle view**
- a **supervised transition-risk layer** centered on **NBER recession within the next 6 months**

It is a public-data project. The core data pipeline uses FRED-compatible public series and caches one raw file per series locally.

## Setup

```bat
cd us_macro_regime_radar
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## FRED access

You can provide a FRED API key with:

```bat
set FRED_API_KEY=your_key_here
```

or with a local file:

```text
config/fred_api_key.json
```

starting from:

```text
config/fred_api_key.template.json
```

If no key is available, the project falls back to the public FRED CSV endpoint for non-vintage pulls.

## Main phases

```bat
python -m src.main --phase fetch_macro_data
python -m src.main --phase build_monthly_panel
python -m src.main --phase build_cycle_labels
python -m src.main --phase train_phase_forecasts
python -m src.main --phase train_recession_risk
python -m src.main --phase build_dashboard_payload
python -m src.main --phase smoke
python -m src.main --phase results
python -m src.main --phase all
```

## Dashboard

The Streamlit app reads saved payloads only. It does not retrain models.

```bat
python -m streamlit run app/streamlit_app.py
```

or

```bat
run_project.bat dashboard
```

## Key outputs

- `outputs/summaries/fetch_macro_data_summary.json`
- `outputs/summaries/build_monthly_panel_summary.json`
- `outputs/summaries/build_cycle_labels_summary.json`
- `outputs/metrics/phase_forecast_metrics.json`
- `outputs/metrics/recession_risk_metrics.json`
- `outputs/metrics/phase_forecast_predictions.csv`
- `outputs/metrics/recession_risk_predictions.csv`
- `outputs/dashboard/history_payload.json`
- `outputs/dashboard/latest_snapshot.json`
- `outputs/dashboard/forecast_payload.json`

## Notes

- The enabled series catalog is in `data/catalog/us_series_catalog.csv`.
- v1 uses revised public data plus explicit release-lag discipline.
- Full ALFRED vintage reconstruction is intentionally out of scope for v1.
- The 4-phase cycle view is deterministic and separate from the official binary recession-risk target.
