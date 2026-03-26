from __future__ import annotations

from io import StringIO

import pandas as pd
import requests

from .config import DEFAULT_START_DATE, FED_NOMINAL_CSV_URL, FRED_BASE_CSV_URL, TREASURY_SERIES


def fetch_fred_series(series_id: str) -> pd.DataFrame:
    response = requests.get(FRED_BASE_CSV_URL.format(series_id=series_id), timeout=60)
    response.raise_for_status()
    frame = pd.read_csv(StringIO(response.text))
    frame.columns = ["date", series_id]
    frame["date"] = pd.to_datetime(frame["date"])
    frame[series_id] = pd.to_numeric(frame[series_id], errors="coerce")
    return frame


def fetch_public_rates(start_date: str = DEFAULT_START_DATE) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for item in TREASURY_SERIES:
        frame = fetch_fred_series(item["series_id"])
        merged = frame if merged is None else merged.merge(frame, on="date", how="outer")
    assert merged is not None
    merged = merged.sort_values("date").reset_index(drop=True)
    merged = merged[merged["date"] >= pd.Timestamp(start_date)].copy()
    return merged


def fetch_fed_nominal_curve(start_date: str = DEFAULT_START_DATE) -> pd.DataFrame:
    response = requests.get(FED_NOMINAL_CSV_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    response.raise_for_status()
    text = response.text
    lines = text.splitlines()
    header_index = next(index for index, line in enumerate(lines) if line.startswith("Date,"))
    frame = pd.read_csv(StringIO("\n".join(lines[header_index:])))
    frame = frame.replace({"NA": pd.NA, "ND": pd.NA})
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.rename(columns={"Date": "date"})
    for column in frame.columns:
        if column != "date":
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame[frame["date"] >= pd.Timestamp(start_date)].copy()
    frame = frame.sort_values("date").reset_index(drop=True)
    return frame


def build_monthly_snapshots(daily_rates: pd.DataFrame) -> pd.DataFrame:
    frame = daily_rates.copy()
    value_columns = [column for column in frame.columns if column != "date"]
    frame = frame.dropna(subset=value_columns, how="all").copy()
    frame["month"] = frame["date"].dt.to_period("M")
    monthly = frame.sort_values("date").groupby("month", as_index=False).tail(1).drop(columns=["month"])
    return monthly.reset_index(drop=True)
