from __future__ import annotations

from math import floor

import numpy as np
import pandas as pd
from scipy.optimize import minimize

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


def _prefilter_sector_diversity(frame: pd.DataFrame, target_positions: int = TARGET_POSITIONS) -> pd.DataFrame:
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


def _target_gross_exposure(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    sector_capacity = (
        frame.groupby("sector", sort=False)["ticker"]
        .count()
        .map(lambda count: min(MAX_SECTOR_WEIGHT, count * MAX_NAME_WEIGHT))
        .sum()
    )
    name_capacity = len(frame) * MAX_NAME_WEIGHT
    return float(min(1.0, name_capacity, sector_capacity))


def _optimize_weights(frame: pd.DataFrame, raw_weights: pd.Series) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)

    gross_target = _target_gross_exposure(frame)
    if gross_target <= 0:
        return pd.Series(0.0, index=frame.index, dtype=float)

    normalized = raw_weights / raw_weights.sum()
    target = normalized * gross_target
    sectors = frame["sector"].fillna("Unknown").tolist()
    lower_bound = MIN_NAME_WEIGHT if gross_target >= len(frame) * MIN_NAME_WEIGHT else 0.0
    bounds = [(lower_bound, MAX_NAME_WEIGHT) for _ in range(len(frame))]

    constraints = [{"type": "eq", "fun": lambda weights, gross_target=gross_target: np.sum(weights) - gross_target}]
    unique_sectors = sorted(set(sectors))
    for sector in unique_sectors:
        sector_idx = [idx for idx, value in enumerate(sectors) if value == sector]
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda weights, sector_idx=sector_idx: MAX_SECTOR_WEIGHT - np.sum(weights[sector_idx]),
            }
        )

    solution = minimize(
        fun=lambda weights: float(np.sum((weights - target.values) ** 2)),
        x0=target.clip(lower=lower_bound, upper=MAX_NAME_WEIGHT).values,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-10},
    )
    if solution.success:
        optimized = np.clip(solution.x, lower_bound, MAX_NAME_WEIGHT)
        if gross_target > 0:
            scale = gross_target / optimized.sum() if optimized.sum() else 0.0
            optimized = optimized * scale
        return pd.Series(optimized, index=frame.index, dtype=float)

    fallback = target.clip(lower=lower_bound, upper=MAX_NAME_WEIGHT)
    for sector in unique_sectors:
        sector_mask = frame["sector"].fillna("Unknown").eq(sector)
        sector_total = float(fallback[sector_mask].sum())
        if sector_total > MAX_SECTOR_WEIGHT:
            fallback.loc[sector_mask] *= MAX_SECTOR_WEIGHT / sector_total
    fallback_total = float(fallback.sum())
    if fallback_total > 0:
        fallback *= gross_target / fallback_total
    return fallback.astype(float)


def select_target_portfolio(
    score_frame: pd.DataFrame,
    signal_date: pd.Timestamp,
    score_column: str = "core_score",
    rank_pct_column: str = "core_rank_pct",
) -> pd.DataFrame:
    daily = score_frame[score_frame["date"] == signal_date].copy()
    daily = daily[(daily[rank_pct_column] >= ENTRY_PERCENTILE) & (daily["trend_filter"] == 1)].copy()
    daily["breakout_preference"] = (
        ((daily["prox_52w_high"] >= 0.95).fillna(False)) | ((daily["breakout_60"] == 1).fillna(False))
    ).astype(int)
    daily = daily.sort_values(
        ["breakout_preference", score_column, rank_pct_column, "prox_52w_high"],
        ascending=[False, False, False, False],
    )
    daily = _prefilter_sector_diversity(daily)

    if len(daily) < MIN_ACTIVE_NAMES:
        fallback = score_frame[score_frame["date"] == signal_date].copy()
        fallback = fallback[fallback["trend_filter"] == 1].copy()
        fallback["breakout_preference"] = (
            ((fallback["prox_52w_high"] >= 0.95).fillna(False)) | ((fallback["breakout_60"] == 1).fillna(False))
        ).astype(int)
        fallback = fallback.sort_values(
            ["breakout_preference", score_column, rank_pct_column, "prox_52w_high"],
            ascending=[False, False, False, False],
        ).head(TARGET_POSITIONS * 2)
        daily = _prefilter_sector_diversity(fallback)

    if daily.empty:
        return daily

    signal_strength = (daily[rank_pct_column] - EXIT_PERCENTILE).clip(lower=0.01)
    inv_vol = 1.0 / daily["realized_vol_20"].clip(lower=0.10)
    raw_weights = signal_strength * inv_vol
    daily["target_weight"] = _optimize_weights(daily, raw_weights).values
    daily["stop_distance"] = np.maximum(STOP_LOSS_FLOOR, STOP_LOSS_ATR_MULTIPLIER * daily["atr_20"].fillna(0.0))
    return daily.reset_index(drop=True)
