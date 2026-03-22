from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import (
    BUILD_CYCLE_LABELS_SUMMARY_PATH,
    BUILD_MONTHLY_PANEL_SUMMARY_PATH,
    CATALOG_CACHE_PATH,
    CYCLE_LABEL_PANEL_PATH,
    FETCH_MACRO_DATA_SUMMARY_PATH,
    MONTHLY_PANEL_PATH,
    PHASE_HORIZONS,
    RAW_FETCH_START_DATE,
    VALIDATION_END,
    WARMUP_TRAIN_END,
)
from .data_access import FredUnavailableError, fetch_series, load_fred_api_key, load_raw_series, load_series_catalog, raw_series_path, save_raw_series
from .utils import expanding_window_split, latest_complete_month, month_end, save_json, zscore_with_reference


SMOKE_SERIES_IDS = {
    "USRECDM",
    "PAYEMS",
    "UNRATE",
    "ICSA",
    "INDPRO",
    "W875RX1",
    "CMRMTSPL",
    "GS10",
    "TB3MS",
    "SP500",
    "VIXCLS",
    "CFNAI",
}


@dataclass
class SeriesSpec:
    series_id: str
    display_name: str
    native_frequency: str
    monthly_aggregation: str
    publication_lag_months: int
    transform_family: str
    role: str


def _load_specs(smoke: bool = False) -> list[SeriesSpec]:
    catalog = load_series_catalog()
    if smoke:
        catalog = catalog[catalog["series_id"].isin(SMOKE_SERIES_IDS)].reset_index(drop=True)
    catalog.to_parquet(CATALOG_CACHE_PATH, index=False)
    return [
        SeriesSpec(
            series_id=row["series_id"],
            display_name=row["display_name"],
            native_frequency=row["native_frequency"],
            monthly_aggregation=row["monthly_aggregation"],
            publication_lag_months=int(row["publication_lag_months"]),
            transform_family=row["transform_family"],
            role=row["role"],
        )
        for _, row in catalog.iterrows()
    ]


def fetch_macro_data(smoke: bool = False) -> dict[str, object]:
    specs = _load_specs(smoke=smoke)
    api_key = load_fred_api_key()
    fetch_rows: list[dict[str, object]] = []
    for spec in specs:
        path = raw_series_path(spec.series_id)
        if path.exists():
            frame = load_raw_series(spec.series_id)
            fetch_rows.append(
                {
                    "series_id": spec.series_id,
                    "source": "cache",
                    "rows": int(frame.shape[0]),
                    "min_date": frame["date"].min().strftime("%Y-%m-%d"),
                    "max_date": frame["date"].max().strftime("%Y-%m-%d"),
                    "missing_share": float(frame["value"].isna().mean()),
                }
            )
            continue
        frame, source = fetch_series(spec.series_id, api_key=api_key, start_date=RAW_FETCH_START_DATE)
        save_raw_series(frame, path)
        fetch_rows.append(
            {
                "series_id": spec.series_id,
                "source": source,
                "rows": int(frame.shape[0]),
                "min_date": frame["date"].min().strftime("%Y-%m-%d"),
                "max_date": frame["date"].max().strftime("%Y-%m-%d"),
                "missing_share": float(frame["value"].isna().mean()),
            }
        )
    summary = {
        "series_count": len(specs),
        "api_key_used": bool(api_key),
        "raw_fetch_start_date": RAW_FETCH_START_DATE,
        "rows": fetch_rows,
    }
    if not smoke:
        save_json(FETCH_MACRO_DATA_SUMMARY_PATH, summary)
    return summary


def _aggregate_to_monthly(raw_frame: pd.DataFrame, spec: SeriesSpec) -> pd.DataFrame:
    frame = raw_frame[["date", "value"]].dropna().copy()
    if frame.empty:
        return pd.DataFrame(columns=["date", spec.series_id])
    frame["month"] = month_end(frame["date"])
    if spec.native_frequency.lower() == "monthly":
        frame = frame.sort_values("date").drop_duplicates("month", keep="last")
        monthly = frame[["month", "value"]].rename(columns={"month": "date", "value": spec.series_id})
        return monthly.reset_index(drop=True)
    if spec.monthly_aggregation == "mean":
        monthly = frame.groupby("month", as_index=False)["value"].mean()
    elif spec.monthly_aggregation == "last":
        monthly = frame.sort_values("date").groupby("month", as_index=False).tail(1)[["month", "value"]]
    else:
        raise RuntimeError(f"Unsupported monthly aggregation rule: {spec.monthly_aggregation}")
    monthly = monthly.rename(columns={"month": "date", "value": spec.series_id}).reset_index(drop=True)
    return monthly


def _transform_series(monthly_values: pd.Series, family: str) -> pd.Series:
    if family == "level":
        return monthly_values.astype(float)
    if family == "log_level":
        return np.log(monthly_values.where(monthly_values > 0))
    if family == "log_diff_12m":
        log_values = np.log(monthly_values.where(monthly_values > 0))
        return log_values - log_values.shift(12)
    raise RuntimeError(f"Unsupported transform family: {family}")


def _build_market_features(raw_sp500: pd.DataFrame) -> pd.DataFrame:
    frame = raw_sp500[["date", "value"]].dropna().copy()
    frame = frame.sort_values("date")
    frame["daily_ret"] = np.log(frame["value"] / frame["value"].shift(1))
    frame["rolling_rv_3m"] = frame["daily_ret"].rolling(63, min_periods=21).std(ddof=0) * np.sqrt(252)
    rolling_max = frame["value"].rolling(126, min_periods=21).max()
    frame["drawdown_6m"] = frame["value"] / rolling_max - 1.0
    frame["month"] = month_end(frame["date"])
    monthly_close = frame.sort_values("date").groupby("month", as_index=False).tail(1)
    monthly_close["sp500_ret_1m"] = np.log(monthly_close["value"] / monthly_close["value"].shift(1))
    output = monthly_close[["month", "sp500_ret_1m", "rolling_rv_3m", "drawdown_6m"]].rename(columns={"month": "date"})
    return output.reset_index(drop=True)


def build_monthly_panel(smoke: bool = False) -> dict[str, object]:
    specs = _load_specs(smoke=smoke)
    latest_month = latest_complete_month()
    monthly_tables: list[pd.DataFrame] = []
    raw_lookup: dict[str, pd.DataFrame] = {}
    feature_columns: list[str] = []
    for spec in specs:
        raw_frame = load_raw_series(spec.series_id)
        raw_lookup[spec.series_id] = raw_frame.copy()
        monthly = _aggregate_to_monthly(raw_frame, spec)
        monthly_tables.append(monthly)
    monthly_panel = monthly_tables[0]
    for table in monthly_tables[1:]:
        monthly_panel = monthly_panel.merge(table, on="date", how="outer")
    monthly_panel = monthly_panel.sort_values("date").reset_index(drop=True)
    monthly_panel = monthly_panel[monthly_panel["date"] <= latest_month].reset_index(drop=True)

    feature_frame = monthly_panel[["date"]].copy()
    for spec in specs:
        raw_col = spec.series_id
        transformed = _transform_series(monthly_panel[raw_col], spec.transform_family)
        current = transformed.shift(spec.publication_lag_months)
        change_3 = transformed.diff(3).shift(spec.publication_lag_months)
        change_6 = transformed.diff(6).shift(spec.publication_lag_months)
        feature_frame[f"{raw_col}_current"] = current
        feature_frame[f"{raw_col}_chg3"] = change_3
        feature_frame[f"{raw_col}_chg6"] = change_6
        feature_columns.extend([f"{raw_col}_current", f"{raw_col}_chg3", f"{raw_col}_chg6"])

    if "SP500" in raw_lookup:
        market_features = _build_market_features(raw_lookup["SP500"])
        feature_frame = feature_frame.merge(market_features, on="date", how="left")
        feature_columns.extend(["sp500_ret_1m", "rolling_rv_3m", "drawdown_6m"])

    if {"GS10_current", "TB3MS_current"}.issubset(feature_frame.columns):
        feature_frame["term_spread"] = feature_frame["GS10_current"] - feature_frame["TB3MS_current"]
        feature_columns.append("term_spread")
    if {"BAA_current", "AAA_current"}.issubset(feature_frame.columns):
        feature_frame["baa_aaa_spread"] = feature_frame["BAA_current"] - feature_frame["AAA_current"]
        feature_columns.append("baa_aaa_spread")

    feature_frame["split"] = expanding_window_split(feature_frame["date"], validation_end=VALIDATION_END, warmup_end=WARMUP_TRAIN_END)
    feature_frame.to_parquet(MONTHLY_PANEL_PATH, index=False)

    summary = {
        "rows": int(feature_frame.shape[0]),
        "date_min": feature_frame["date"].min().strftime("%Y-%m-%d"),
        "date_max": feature_frame["date"].max().strftime("%Y-%m-%d"),
        "feature_count": len(feature_columns),
        "feature_missing_share_top20": (
            {
                key: float(value)
                for key, value in feature_frame[feature_columns].isna().mean().sort_values(ascending=False).head(20).to_dict().items()
            }
        ),
        "split_counts": feature_frame["split"].value_counts(dropna=False).to_dict(),
    }
    if not smoke:
        save_json(BUILD_MONTHLY_PANEL_SUMMARY_PATH, summary)
    return summary


def _classify_phase(level_score: float, momentum_score: float) -> str | None:
    if pd.isna(level_score) or pd.isna(momentum_score):
        return None
    if level_score >= 0 and momentum_score >= 0:
        return "Expansion"
    if level_score >= 0 and momentum_score < 0:
        return "Slowdown"
    if level_score < 0 and momentum_score < 0:
        return "Contraction"
    return "Recovery"


def build_cycle_labels(smoke: bool = False) -> dict[str, object]:
    panel = pd.read_parquet(MONTHLY_PANEL_PATH)
    panel["date"] = pd.to_datetime(panel["date"])
    reference_mask = (panel["date"] >= pd.Timestamp("1973-01-31")) & (panel["date"] <= pd.Timestamp(WARMUP_TRAIN_END))
    component_map = {
        "PAYEMS_current": "payems_z",
        "INDPRO_current": "indpro_z",
        "W875RX1_current": "income_z",
        "CMRMTSPL_current": "sales_z",
    }
    for source_col, target_col in component_map.items():
        panel[target_col] = zscore_with_reference(panel[source_col], reference_mask)
    panel["unrate_component"] = zscore_with_reference(-panel["UNRATE_chg3"], reference_mask)
    component_cols = ["payems_z", "indpro_z", "income_z", "sales_z", "unrate_component"]
    panel["activity_score"] = panel[component_cols].mean(axis=1, skipna=True)
    panel["level_score"] = panel["activity_score"].rolling(3, min_periods=3).mean()
    panel["momentum_score"] = panel["level_score"] - panel["level_score"].shift(3)
    panel["current_phase"] = [
        _classify_phase(level_score, momentum_score)
        for level_score, momentum_score in zip(panel["level_score"], panel["momentum_score"])
    ]
    panel["current_phase_int"] = panel["current_phase"].map(
        {"Expansion": 0, "Slowdown": 1, "Contraction": 2, "Recovery": 3}
    )
    for horizon in PHASE_HORIZONS:
        panel[f"phase_t_plus_{horizon}m"] = panel["current_phase"].shift(-horizon)
        panel[f"phase_t_plus_{horizon}m_int"] = panel[f"phase_t_plus_{horizon}m"].map(
            {"Expansion": 0, "Slowdown": 1, "Contraction": 2, "Recovery": 3}
        )

    recession_forward = []
    for idx in range(panel.shape[0]):
        window = panel["USRECDM_current"].iloc[idx + 1 : idx + 7]
        recession_forward.append(int((window == 1).any()) if not window.empty else np.nan)
    panel["recession_within_6m"] = recession_forward
    panel["recession_start"] = ((panel["USRECDM_current"] == 1) & (panel["USRECDM_current"].shift(1).fillna(0) == 0)).astype(int)
    panel.to_parquet(CYCLE_LABEL_PANEL_PATH, index=False)

    summary = {
        "rows": int(panel.shape[0]),
        "date_min": panel["date"].min().strftime("%Y-%m-%d"),
        "date_max": panel["date"].max().strftime("%Y-%m-%d"),
        "phase_counts": {
            ("Missing" if pd.isna(key) else str(key)): int(value)
            for key, value in panel["current_phase"].value_counts(dropna=False).to_dict().items()
        },
        "recession_within_6m_positive_rate": float(pd.Series(panel["recession_within_6m"]).dropna().mean()),
        "reference_window": {"start": "1973-01-31", "end": WARMUP_TRAIN_END},
    }
    if not smoke:
        save_json(BUILD_CYCLE_LABELS_SUMMARY_PATH, summary)
    return summary
