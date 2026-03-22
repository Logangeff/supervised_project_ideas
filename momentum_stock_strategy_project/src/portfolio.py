from __future__ import annotations

from math import floor

import numpy as np
import pandas as pd

from .config import (
    ENTRY_PERCENTILE,
    EXIT_PERCENTILE,
    MAX_NAME_WEIGHT,
    MAX_SECTOR_WEIGHT,
    MIN_ACTIVE_NAMES,
    MIN_NAME_WEIGHT,
    STOP_LOSS_ATR_MULTIPLIER,
    STOP_LOSS_FLOOR,
    TARGET_POSITIONS,
)


def _apply_sector_cap(frame: pd.DataFrame, target_positions: int = TARGET_POSITIONS) -> pd.DataFrame:
    sector_limit = max(1, floor(MAX_SECTOR_WEIGHT * target_positions))
    kept: list[pd.Series] = []
    sector_counts: dict[str, int] = {}
    for _, row in frame.iterrows():
        sector = row["sector"]
        if sector_counts.get(sector, 0) >= sector_limit:
            continue
        kept.append(row)
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
        if len(kept) >= target_positions:
            break
    return pd.DataFrame(kept)


def _clip_and_normalize(weights: pd.Series) -> pd.Series:
    if weights.empty:
        return weights
    clipped = weights.clip(lower=MIN_NAME_WEIGHT, upper=MAX_NAME_WEIGHT)
    clipped = clipped / clipped.sum()
    clipped = clipped.clip(lower=MIN_NAME_WEIGHT, upper=MAX_NAME_WEIGHT)
    return clipped / clipped.sum()


def select_target_portfolio(
    score_frame: pd.DataFrame,
    signal_date: pd.Timestamp,
    score_column: str = "core_score",
    rank_pct_column: str = "core_rank_pct",
) -> pd.DataFrame:
    daily = score_frame[score_frame["date"] == signal_date].copy()
    daily = daily[(daily[rank_pct_column] >= ENTRY_PERCENTILE) & (daily["trend_filter"] == 1)].copy()
    daily = daily[(daily["prox_52w_high"] >= 0.95) | (daily["breakout_60"] == 1)].copy()
    daily = daily.sort_values([score_column, rank_pct_column], ascending=[False, False])
    daily = _apply_sector_cap(daily)

    if len(daily) < MIN_ACTIVE_NAMES:
        fallback = score_frame[score_frame["date"] == signal_date].copy()
        fallback = fallback[fallback["trend_filter"] == 1].sort_values(score_column, ascending=False).head(TARGET_POSITIONS * 2)
        daily = _apply_sector_cap(fallback)

    if daily.empty:
        return daily

    signal_strength = (daily[rank_pct_column] - EXIT_PERCENTILE).clip(lower=0.01)
    inv_vol = 1.0 / daily["realized_vol_20"].clip(lower=0.10)
    raw_weights = signal_strength * inv_vol
    daily["target_weight"] = _clip_and_normalize(raw_weights / raw_weights.sum()).values
    daily["stop_distance"] = np.maximum(STOP_LOSS_FLOOR, STOP_LOSS_ATR_MULTIPLIER * daily["atr_20"].fillna(0.0))
    return daily.reset_index(drop=True)
