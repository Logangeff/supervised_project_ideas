from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MIN_HISTORY_DAYS, MIN_MEDIAN_DOLLAR_VOLUME, MIN_PRICE, SECTOR_ETF_MAP, UNIVERSE_TARGET_COUNT


def build_latest_universe(universe_source: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    stocks = prices[prices["ticker"].isin(universe_source["ticker"])].copy()
    stocks["dollar_volume"] = stocks["adj_close"] * stocks["volume"]
    stocks = stocks.sort_values(["ticker", "date"])

    summaries: list[dict[str, object]] = []
    for ticker, frame in stocks.groupby("ticker", sort=True):
        last_60 = frame.tail(60)
        last_row = frame.iloc[-1]
        summaries.append(
            {
                "ticker": ticker,
                "latest_date": last_row["date"],
                "latest_adj_close": float(last_row["adj_close"]),
                "median_dollar_volume_60": float(last_60["dollar_volume"].median()),
                "history_days": int(frame["date"].nunique()),
            }
        )

    stats = pd.DataFrame.from_records(summaries)
    base = universe_source.drop(
        columns=[
            "latest_date",
            "latest_adj_close",
            "median_dollar_volume_60",
            "history_days",
            "eligible",
            "universe_rank",
            "sector_etf",
        ],
        errors="ignore",
    )
    merged = base.merge(stats, on="ticker", how="left")
    merged["eligible"] = (
        merged["latest_adj_close"].fillna(0.0) >= MIN_PRICE
    ) & (
        merged["median_dollar_volume_60"].fillna(0.0) >= MIN_MEDIAN_DOLLAR_VOLUME
    ) & (
        merged["history_days"].fillna(0).astype(int) >= MIN_HISTORY_DAYS
    )
    eligible = merged[merged["eligible"]].copy()
    eligible = eligible.sort_values("median_dollar_volume_60", ascending=False).head(UNIVERSE_TARGET_COUNT)
    eligible["universe_rank"] = np.arange(1, len(eligible) + 1)
    eligible["sector_etf"] = eligible["sector"].map(SECTOR_ETF_MAP).fillna("SPY")
    return eligible.reset_index(drop=True)
