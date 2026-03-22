from __future__ import annotations

from collections.abc import Iterable
from io import StringIO

import pandas as pd
import requests
import yfinance as yf

from .config import DATA_PROVIDER, S_AND_P_400_URL, S_AND_P_500_URL


def sanitize_ticker(ticker: str) -> str:
    return str(ticker).strip().upper().replace(".", "-")


def _normalized_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _pick_column(frame: pd.DataFrame, candidates: list[str]) -> str:
    normalized = {_normalized_name(column): column for column in frame.columns}
    for candidate in candidates:
        key = _normalized_name(candidate)
        if key in normalized:
            return normalized[key]
    raise KeyError(f"Could not find any of {candidates} in columns {list(frame.columns)}")


def fetch_wikipedia_constituents(url: str, index_name: str) -> pd.DataFrame:
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    response.raise_for_status()
    frame = pd.read_html(StringIO(response.text))[0].copy()
    ticker_col = _pick_column(frame, ["symbol", "ticker symbol", "ticker"])
    name_col = _pick_column(frame, ["security", "company", "company name"])
    sector_col = _pick_column(frame, ["gics sector", "sector"])
    result = frame[[ticker_col, name_col, sector_col]].copy()
    result.columns = ["ticker", "company_name", "sector"]
    result["ticker"] = result["ticker"].map(sanitize_ticker)
    result["company_name"] = result["company_name"].astype(str).str.strip()
    result["sector"] = result["sector"].astype(str).str.strip()
    result["index_source"] = index_name
    return result.drop_duplicates("ticker").reset_index(drop=True)


def build_free_universe_source() -> pd.DataFrame:
    sp500 = fetch_wikipedia_constituents(S_AND_P_500_URL, "sp500")
    sp400 = fetch_wikipedia_constituents(S_AND_P_400_URL, "sp400")
    combined = pd.concat([sp500, sp400], ignore_index=True)
    return combined.drop_duplicates("ticker", keep="first").reset_index(drop=True)


def get_universe_source() -> pd.DataFrame:
    if DATA_PROVIDER != "free":
        raise NotImplementedError(
            "Only the free-data provider is implemented in code right now. "
            "The architecture is ready for a cheap-paid provider upgrade, but the live implementation path is free-only."
        )
    return build_free_universe_source()


def build_download_ticker_list(
    universe_source: pd.DataFrame,
    benchmark_tickers: list[str],
    sector_etfs: list[str],
    watchlist: list[str],
) -> list[str]:
    tickers = set(universe_source["ticker"].astype(str))
    tickers.update(benchmark_tickers)
    tickers.update(sector_etfs)
    tickers.update(watchlist)
    return sorted(sanitize_ticker(ticker) for ticker in tickers if str(ticker).strip())


def download_price_history(tickers: Iterable[str], start: str, end: str | None = None) -> pd.DataFrame:
    ticker_list = sorted(set(sanitize_ticker(ticker) for ticker in tickers))
    raw = yf.download(
        tickers=ticker_list,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )
    if raw.empty:
        raise ValueError("No price data returned from yfinance.")

    if isinstance(raw.columns, pd.MultiIndex):
        long = raw.stack(level=1, future_stack=True).rename_axis(index=["date", "ticker"]).reset_index()
    else:
        long = raw.reset_index()
        long["ticker"] = ticker_list[0]

    rename_map = {
        "Adj Close": "adj_close",
        "Close": "close",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Volume": "volume",
    }
    long = long.rename(columns=rename_map)
    long["ticker"] = long["ticker"].map(sanitize_ticker)
    long["date"] = pd.to_datetime(long["date"])
    long = long.dropna(subset=["close"]).copy()

    if "adj_close" not in long.columns:
        long["adj_close"] = long["close"]

    adj_factor = (long["adj_close"] / long["close"]).replace([float("inf"), float("-inf")], 1.0).fillna(1.0)
    long["adj_open"] = long["open"] * adj_factor
    long["adj_high"] = long["high"] * adj_factor
    long["adj_low"] = long["low"] * adj_factor
    long["volume"] = pd.to_numeric(long["volume"], errors="coerce").fillna(0.0)

    return long[
        [
            "date",
            "ticker",
            "open",
            "high",
            "low",
            "close",
            "adj_open",
            "adj_high",
            "adj_low",
            "adj_close",
            "volume",
        ]
    ].sort_values(["ticker", "date"]).reset_index(drop=True)
