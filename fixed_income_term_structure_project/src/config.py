from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT_DIR / "docs"
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
FIGURES_DIR = ROOT_DIR / "outputs" / "figures"
METRICS_DIR = ROOT_DIR / "outputs" / "metrics"
SUMMARIES_DIR = ROOT_DIR / "outputs" / "summaries"

FRED_BASE_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
FED_NOMINAL_CSV_URL = "https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv"

TREASURY_SERIES = [
    {"series_id": "DGS1MO", "label": "1M", "maturity_years": 1.0 / 12.0},
    {"series_id": "DGS3MO", "label": "3M", "maturity_years": 3.0 / 12.0},
    {"series_id": "DGS6MO", "label": "6M", "maturity_years": 6.0 / 12.0},
    {"series_id": "DGS1", "label": "1Y", "maturity_years": 1.0},
    {"series_id": "DGS2", "label": "2Y", "maturity_years": 2.0},
    {"series_id": "DGS3", "label": "3Y", "maturity_years": 3.0},
    {"series_id": "DGS5", "label": "5Y", "maturity_years": 5.0},
    {"series_id": "DGS7", "label": "7Y", "maturity_years": 7.0},
    {"series_id": "DGS10", "label": "10Y", "maturity_years": 10.0},
    {"series_id": "DGS20", "label": "20Y", "maturity_years": 20.0},
    {"series_id": "DGS30", "label": "30Y", "maturity_years": 30.0},
]

OBSERVED_MATURITIES = [item["maturity_years"] for item in TREASURY_SERIES]
LONG_END_MATURITIES = [maturity for maturity in OBSERVED_MATURITIES if maturity >= 1.0]
BOOTSTRAP_GRID = [round(step * 0.5, 6) for step in range(1, 61)]
MODEL_EVAL_GRID = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0]
PAR_SWAP_MATURITIES = [2.0, 5.0, 10.0, 30.0]
BOND_PRICING_MATURITIES = [2.0, 5.0, 10.0, 30.0]
FED_BENCHMARK_MATURITIES = list(range(1, 31))

DEFAULT_START_DATE = "2010-01-01"
SMOKE_START_DATE = "2022-01-01"

HULL_WHITE_DEFAULT_A = 0.10
HULL_WHITE_DEFAULT_SIGMA = 0.01
