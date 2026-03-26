from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MIN_HISTORY_DAYS, MIN_MEDIAN_DOLLAR_VOLUME, MIN_PRICE, SECTOR_ETF_MAP


def build_point_in_time_universe(
    universe_source: pd.DataFrame,
    prices: pd.DataFrame,
    target_count: int | None = None,
) -> pd.DataFrame:
    static_cols = [column for column in ["ticker", "company_name", "sector", "siccd", "index_source"] if column in universe_source.columns]
    static = universe_source[static_cols].drop_duplicates("ticker").copy()
    stocks = prices[prices["ticker"].isin(static["ticker"])].copy()
    if stocks.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "company_name",
                "sector",
                "sector_etf",
                "latest_adj_close",
                "median_dollar_volume_60",
                "history_days",
                "eligible",
                "universe_rank",
            ]
        )

    stocks["date"] = pd.to_datetime(stocks["date"])
    stocks = stocks.sort_values(["ticker", "date"]).copy()
    stocks["dollar_volume"] = stocks["close"] * stocks["volume"]
    stocks["median_dollar_volume_60"] = (
        stocks.groupby("ticker", sort=False)["dollar_volume"].transform(lambda series: series.rolling(60, min_periods=40).median())
    )
    stocks["history_days"] = stocks.groupby("ticker", sort=False).cumcount() + 1
    stocks["latest_adj_close"] = stocks["close"]

    membership = stocks.merge(static, on="ticker", how="left")
    for column in ("company_name", "sector"):
        if column not in membership.columns:
            membership[column] = np.nan
    membership["eligible"] = (
        membership["latest_adj_close"].fillna(0.0) >= MIN_PRICE
    ) & (
        membership["median_dollar_volume_60"].fillna(0.0) >= MIN_MEDIAN_DOLLAR_VOLUME
    ) & (
        membership["history_days"].fillna(0).astype(int) >= MIN_HISTORY_DAYS
    )
    membership = membership[membership["eligible"]].copy()
    if membership.empty:
        return membership

    membership["universe_rank"] = (
        membership.groupby("date", sort=False)["median_dollar_volume_60"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    if target_count is not None:
        membership = membership[membership["universe_rank"] <= int(target_count)].copy()

    membership["sector"] = membership["sector"].fillna("Unknown")
    membership["sector_etf"] = membership["sector"].map(SECTOR_ETF_MAP).fillna("SPY")
    membership = membership[
        [
            "date",
            "ticker",
            "company_name",
            "sector",
            "sector_etf",
            "latest_adj_close",
            "median_dollar_volume_60",
            "history_days",
            "eligible",
            "universe_rank",
        ]
    ].sort_values(["date", "universe_rank", "ticker"])
    return membership.reset_index(drop=True)


def build_latest_universe(universe_membership: pd.DataFrame) -> pd.DataFrame:
    if universe_membership.empty:
        return universe_membership
    latest_date = pd.to_datetime(universe_membership["date"]).max()
    latest = universe_membership[universe_membership["date"] == latest_date].copy()
    latest = latest.sort_values(["universe_rank", "ticker"]).reset_index(drop=True)
    latest["universe_rank"] = np.arange(1, len(latest) + 1)
    return latest
