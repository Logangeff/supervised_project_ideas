# Fixed Income Term-Structure Project

This project builds a public-data, U.S.-first fixed-income term-structure workflow around the course material in `Teaching_notes_60201_W2026.pdf`.

The project is intentionally staged:

1. construct Treasury-based par, zero, discount, and forward curves
2. smooth the curve with Nelson-Siegel-Svensson
3. compare one-factor models: Vasicek, CIR, and Hull-White 1F
4. evaluate fit, stability, and pricing usefulness with simple bond and par-swap outputs

## v1 scope

Included:
- public U.S. Treasury data from FRED
- monthly end-of-period snapshots
- bootstrapped zero / discount / forward curves
- Nelson-Siegel-Svensson benchmark curve
- Vasicek, CIR, Hull-White 1F comparison
- single-curve simplified pricing outputs

Excluded from v1:
- OIS discounting and multi-curve swap pricing
- HJM
- cap / swaption calibration
- Kalman filtering
- Diebold-Li forecasting
- credit-risk extensions

## Project structure

- `docs/course_material/`: copied course notes and exercises
- `docs/project_direction.md`: design memo and rationale
- `src/`: pipeline implementation
- `data/raw/`: downloaded Treasury rate data
- `data/processed/`: processed curve and model outputs
- `outputs/figures/`: charts
- `outputs/metrics/`: evaluation tables
- `outputs/summaries/`: JSON run summaries
- `results_overview.ipynb`: professor-facing summary notebook

## Run

From the project root:

```bat
run_project.bat
```

Or directly:

```bat
python -m src.main --phase all
```

## Dashboard

The project includes a professor-facing Streamlit dashboard that makes the benchmark logic explicit:

- Hull-White 1F is shown as the exact-fit anchored benchmark
- CIR and Vasicek are compared as the fair equilibrium-model contest
- pricing and swap outputs are shown by maturity

Launch it with:

```bat
launch_dashboard.bat
```

Useful phases:
- `fetch_public_rates`
- `build_bootstrap_curves`
- `fit_nss_curve`
- `fit_short_rate_models`
- `evaluate_models`
- `build_pricing_outputs`
- `results`
- `smoke`
- `all`

## Main outputs

- `outputs/summaries/fetch_public_rates_summary.json`
- `outputs/summaries/build_bootstrap_curves_summary.json`
- `outputs/summaries/fit_nss_curve_summary.json`
- `outputs/summaries/fit_short_rate_models_summary.json`
- `outputs/metrics/model_fit_metrics.csv`
- `outputs/metrics/model_pricing_metrics.csv`
- `outputs/metrics/parameter_stability_metrics.csv`
- `outputs/metrics/swap_rate_comparison.csv`

## Interpretation

The key comparison in v1 is:

- `Vasicek` and `CIR` as parsimonious equilibrium-model comparators
- `Hull-White 1F` as an arbitrage-free current-curve anchored benchmark

Hull-White should not be interpreted as a fair free-parameter fit competitor on current-curve fit error. Its role is different: it is the exact-fit anchored reference.
