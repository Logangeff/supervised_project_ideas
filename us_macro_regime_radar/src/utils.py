from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def save_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def month_end(date_like: pd.Series | Iterable[pd.Timestamp] | pd.DatetimeIndex) -> pd.Series:
    dates = pd.to_datetime(date_like)
    if isinstance(dates, pd.Series):
        return dates.dt.to_period("M").dt.to_timestamp("M")
    if isinstance(dates, pd.DatetimeIndex):
        return pd.Series(dates.to_period("M").to_timestamp("M"), index=dates)
    return pd.Series(pd.to_datetime(dates).to_period("M").to_timestamp("M"))


def safe_numeric(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    return values.replace([np.inf, -np.inf], np.nan)


def zscore_with_reference(series: pd.Series, reference_mask: pd.Series) -> pd.Series:
    reference = series[reference_mask].dropna()
    mean_value = float(reference.mean()) if not reference.empty else 0.0
    std_value = float(reference.std(ddof=0)) if not reference.empty else 0.0
    if std_value <= 0:
        return pd.Series(np.nan, index=series.index, dtype=float)
    return (series - mean_value) / std_value


def latest_complete_month(as_of: pd.Timestamp | None = None) -> pd.Timestamp:
    current = pd.Timestamp.now().normalize() if as_of is None else pd.Timestamp(as_of).normalize()
    month_start = current.to_period("M").to_timestamp()
    return (month_start - pd.offsets.Day(1)).normalize()


def expanding_window_split(date_series: pd.Series, validation_end: str, warmup_end: str) -> pd.Series:
    dates = pd.to_datetime(date_series)
    split = pd.Series(index=dates.index, dtype="object")
    split.loc[dates <= pd.Timestamp(warmup_end)] = "train"
    split.loc[(dates > pd.Timestamp(warmup_end)) & (dates <= pd.Timestamp(validation_end))] = "validation"
    split.loc[dates > pd.Timestamp(validation_end)] = "test"
    return split


def annualized_volatility(monthly_returns: pd.Series) -> float:
    valid = monthly_returns.dropna()
    if valid.empty:
        return float("nan")
    return float(valid.std(ddof=0) * np.sqrt(12))


def annualized_return(monthly_returns: pd.Series) -> float:
    valid = monthly_returns.dropna()
    if valid.empty:
        return float("nan")
    compounded = float((1.0 + valid).prod())
    periods = valid.shape[0]
    if periods == 0 or compounded <= 0:
        return float("nan")
    return float(compounded ** (12.0 / periods) - 1.0)
