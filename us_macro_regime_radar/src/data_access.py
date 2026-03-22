from __future__ import annotations

import os
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from .config import FRED_API_KEY_JSON, RAW_FETCH_START_DATE, RAW_FRED_DIR, SERIES_CATALOG_CSV
from .utils import safe_numeric


FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


class FredUnavailableError(RuntimeError):
    """Raised when FRED access fails."""


def load_fred_api_key() -> str | None:
    env_value = os.environ.get("FRED_API_KEY", "").strip()
    if env_value:
        return env_value
    if FRED_API_KEY_JSON.exists():
        payload = pd.read_json(FRED_API_KEY_JSON, typ="series")
        api_key = str(payload.get("api_key", "")).strip()
        return api_key or None
    return None


def load_series_catalog() -> pd.DataFrame:
    catalog = pd.read_csv(SERIES_CATALOG_CSV)
    catalog["enabled"] = catalog["enabled"].astype(int)
    catalog["publication_lag_months"] = catalog["publication_lag_months"].astype(int)
    return catalog[catalog["enabled"] == 1].reset_index(drop=True)


def raw_series_path(series_id: str) -> Path:
    return RAW_FRED_DIR / f"{series_id}.parquet"


def fetch_series(series_id: str, api_key: str | None = None, start_date: str = RAW_FETCH_START_DATE) -> tuple[pd.DataFrame, str]:
    if api_key:
        response = requests.get(
            FRED_OBSERVATIONS_URL,
            params={
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "observation_start": start_date,
                "sort_order": "asc",
            },
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        observations = pd.DataFrame(payload.get("observations", []))
        if observations.empty:
            raise FredUnavailableError(f"No observations returned for series {series_id}.")
        frame = observations.rename(columns={"date": "date", "value": "value"}).copy()
        frame["series_id"] = series_id
        frame["date"] = pd.to_datetime(frame["date"])
        frame["value"] = safe_numeric(frame["value"])
        frame["realtime_start"] = pd.to_datetime(frame["realtime_start"], errors="coerce")
        frame["realtime_end"] = pd.to_datetime(frame["realtime_end"], errors="coerce")
        return frame[["series_id", "date", "value", "realtime_start", "realtime_end"]], "api_json"

    response = requests.get(FRED_CSV_URL, params={"id": series_id}, timeout=60)
    response.raise_for_status()
    frame = pd.read_csv(StringIO(response.text))
    date_candidates = [column for column in frame.columns if column.lower() in {"date", "observation_date"}]
    if not date_candidates:
        raise FredUnavailableError(f"Unexpected CSV date column returned for series {series_id}: {list(frame.columns)}")
    date_column = date_candidates[0]
    value_column = [column for column in frame.columns if column != date_column]
    if not value_column:
        raise FredUnavailableError(f"Unexpected CSV format returned for series {series_id}.")
    frame = frame.rename(columns={date_column: "date", value_column[0]: "value"}).copy()
    frame["series_id"] = series_id
    frame["date"] = pd.to_datetime(frame["date"])
    frame["value"] = safe_numeric(frame["value"])
    frame["realtime_start"] = pd.NaT
    frame["realtime_end"] = pd.NaT
    frame = frame[frame["date"] >= pd.Timestamp(start_date)].reset_index(drop=True)
    return frame[["series_id", "date", "value", "realtime_start", "realtime_end"]], "csv_fallback"


def save_raw_series(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def load_raw_series(series_id: str) -> pd.DataFrame:
    path = raw_series_path(series_id)
    if not path.exists():
        raise FileNotFoundError(f"Missing raw series cache: {path}")
    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["date"])
    if "realtime_start" in frame.columns:
        frame["realtime_start"] = pd.to_datetime(frame["realtime_start"], errors="coerce")
    if "realtime_end" in frame.columns:
        frame["realtime_end"] = pd.to_datetime(frame["realtime_end"], errors="coerce")
    frame["value"] = safe_numeric(frame["value"])
    return frame.sort_values("date").reset_index(drop=True)
