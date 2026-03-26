from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    CORE_SIGNAL_WEIGHTS,
    EARLY_SIGNAL_WEIGHTS,
    EXPERIMENTAL_OVERLAY_WEIGHTS,
    SMOOTHING_SPAN,
    WINSOR_LOWER,
    WINSOR_UPPER,
)


CORE_SIGNAL_COLUMNS = list(CORE_SIGNAL_WEIGHTS)
EXPERIMENTAL_SIGNAL_COLUMNS = list(EXPERIMENTAL_OVERLAY_WEIGHTS)


def _winsorize(series: pd.Series) -> pd.Series:
    valid = series.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=series.index)
    lower = series.quantile(WINSOR_LOWER)
    upper = series.quantile(WINSOR_UPPER)
    return series.clip(lower=lower, upper=upper)


def _zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def _score_single_date(frame: pd.DataFrame) -> pd.DataFrame:
    scored = frame.copy()
    for column in set(CORE_SIGNAL_COLUMNS).union(EXPERIMENTAL_SIGNAL_COLUMNS):
        winsorized = _winsorize(scored[column].replace([np.inf, -np.inf], np.nan))
        fill_value = winsorized.dropna().median() if winsorized.notna().any() else 0.0
        scored[f"{column}_z"] = _zscore(winsorized.fillna(fill_value))

    log_dollar_volume = pd.Series(np.nan, index=scored.index, dtype=float)
    positive_liquidity = scored["avg_dollar_volume_60"] > 0
    log_dollar_volume.loc[positive_liquidity] = np.log(scored.loc[positive_liquidity, "avg_dollar_volume_60"])
    liquidity_score = _zscore(log_dollar_volume.fillna(0.0))
    raw_core = 0.0
    for column, weight in CORE_SIGNAL_WEIGHTS.items():
        raw_core += weight * scored[f"{column}_z"]
    scored["raw_momentum_score"] = raw_core
    scored["raw_experimental_overlay"] = 0.0
    for column, weight in EXPERIMENTAL_OVERLAY_WEIGHTS.items():
        scored["raw_experimental_overlay"] += weight * scored[f"{column}_z"]
    scored["raw_experimental_score"] = scored["raw_momentum_score"] + scored["raw_experimental_overlay"]
    scored["trend_multiplier"] = np.where(scored["trend_filter"] == 1, 1.0, 0.5)
    scored["core_score_pre_smooth"] = scored["raw_momentum_score"] * scored["trend_multiplier"] + 0.05 * liquidity_score
    scored["experimental_score_pre_smooth"] = scored["raw_experimental_score"] * scored["trend_multiplier"] + 0.05 * liquidity_score
    return scored


def _early_score_single_date(frame: pd.DataFrame) -> pd.DataFrame:
    scored = frame.copy()
    for column in EARLY_SIGNAL_WEIGHTS:
        series = _winsorize(scored[column].replace([np.inf, -np.inf], np.nan)).fillna(0.0)
        scored[f"{column}_early_z"] = _zscore(series)
    early = 0.0
    for column, weight in EARLY_SIGNAL_WEIGHTS.items():
        early += weight * scored[f"{column}_early_z"]
    scored["early_score_pre_smooth"] = early
    return scored


def compute_score_panel(feature_panel: pd.DataFrame) -> pd.DataFrame:
    base = (
        feature_panel.groupby("date", group_keys=False, sort=True)
        .apply(_score_single_date)
        .reset_index(drop=True)
    )
    base = base.sort_values(["ticker", "date"]).copy()
    base["core_score"] = (
        base.groupby("ticker", sort=False)["core_score_pre_smooth"]
        .transform(lambda series: series.ewm(span=SMOOTHING_SPAN, adjust=False).mean())
    )
    base["experimental_score"] = (
        base.groupby("ticker", sort=False)["experimental_score_pre_smooth"]
        .transform(lambda series: series.ewm(span=SMOOTHING_SPAN, adjust=False).mean())
    )
    base["core_rank_pct"] = base.groupby("date", sort=False)["core_score"].rank(method="first", pct=True)
    base["core_rank"] = base.groupby("date", sort=False)["core_score"].rank(method="first", ascending=False).astype(int)
    base["experimental_rank_pct"] = base.groupby("date", sort=False)["experimental_score"].rank(method="first", pct=True)
    base["experimental_rank"] = (
        base.groupby("date", sort=False)["experimental_score"].rank(method="first", ascending=False).astype(int)
    )
    base["rank_delta_20"] = base.groupby("ticker", sort=False)["core_rank_pct"].diff(20)

    early = (
        base.groupby("date", group_keys=False, sort=True)
        .apply(_early_score_single_date)
        .reset_index(drop=True)
    )
    early = early.sort_values(["ticker", "date"]).copy()
    early["early_score"] = (
        early.groupby("ticker", sort=False)["early_score_pre_smooth"]
        .transform(lambda series: series.ewm(span=SMOOTHING_SPAN, adjust=False).mean())
    )
    early["early_rank_pct"] = early.groupby("date", sort=False)["early_score"].rank(method="first", pct=True)
    early["early_rank"] = early.groupby("date", sort=False)["early_score"].rank(method="first", ascending=False).astype(int)
    return early.sort_values(["date", "core_rank"]).reset_index(drop=True)
