from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Iterable

import pandas as pd

from .config import (
    BENCHMARK_TICKERS,
    FALLBACK_WRDS_CREDENTIALS_JSON,
    MIN_HISTORY_DAYS,
    MIN_MEDIAN_DOLLAR_VOLUME,
    MIN_PRICE,
    RECENT_MARKET_CACHE_CSV,
    RECENT_MARKET_CACHE_PARQUET,
    RECENT_SYMBOL_ALIASES,
    SECTOR_ETF_MAP,
    UNIVERSE_TARGET_COUNT,
    USER_WATCHLIST_PATH,
    WATCHLIST,
    WRDS_CREDENTIALS_JSON,
    WRDS_STOCK_CACHE_CSV,
    WRDS_STOCK_CACHE_PARQUET,
    WRDS_UNIVERSE_SOURCE_CSV,
    WRDS_UNIVERSE_SOURCE_PARQUET,
)
from .data_provider import download_price_history, sanitize_ticker


class WrdsUnavailableError(RuntimeError):
    pass


@dataclass
class WrdsRefreshArtifacts:
    universe_source: pd.DataFrame
    price_panel: pd.DataFrame
    summary: dict[str, object]


def _load_wrds_credentials() -> dict[str, str]:
    for candidate in (WRDS_CREDENTIALS_JSON, FALLBACK_WRDS_CREDENTIALS_JSON):
        if candidate.exists():
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            username = str(payload.get("username", "")).strip()
            password = str(payload.get("password", "")).strip()
            if username:
                return {"username": username, "password": password}
    username = str(os.getenv("WRDS_USERNAME", "")).strip()
    password = str(os.getenv("WRDS_PASSWORD", "")).strip()
    return {"username": username, "password": password}


def connect_wrds():
    try:
        import wrds
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise WrdsUnavailableError("The 'wrds' package is not installed.") from exc

    credentials = _load_wrds_credentials()
    username = credentials["username"]
    password = credentials["password"]
    try:
        if username and password:
            return wrds.Connection(wrds_username=username, wrds_password=password)
        if username:
            return wrds.Connection(wrds_username=username)
        return wrds.Connection()
    except Exception as exc:  # pragma: no cover - depends on WRDS auth state
        raise WrdsUnavailableError("Could not establish a WRDS connection for the momentum project.") from exc


def _chunked(values: Iterable[int], size: int = 250) -> list[list[int]]:
    items = list(values)
    return [items[idx: idx + size] for idx in range(0, len(items), size)]


def _sic_to_sector(siccd: object) -> str:
    if pd.isna(siccd):
        return "Unknown"
    sic = int(float(siccd))
    if sic == 9999:
        return "Unknown"
    if 100 <= sic <= 999:
        return "Energy"
    if 1000 <= sic <= 1499:
        return "Materials"
    if 1500 <= sic <= 1999:
        return "Industrials"
    if 2000 <= sic <= 2399:
        return "Consumer Discretionary"
    if 2830 <= sic <= 2836:
        return "Health Care"
    if 2400 <= sic <= 2999:
        return "Consumer Staples"
    if 3000 <= sic <= 3569:
        return "Industrials"
    if 3570 <= sic <= 3692:
        return "Information Technology"
    if 3710 <= sic <= 3716:
        return "Consumer Discretionary"
    if 3693 <= sic <= 3999:
        return "Industrials"
    if 4000 <= sic <= 4799:
        return "Industrials"
    if 4800 <= sic <= 4899:
        return "Communication Services"
    if 4900 <= sic <= 4949:
        return "Utilities"
    if 4950 <= sic <= 4999:
        return "Industrials"
    if 5000 <= sic <= 5999:
        return "Consumer Discretionary"
    if 6000 <= sic <= 6499:
        return "Financials"
    if 6500 <= sic <= 6799:
        return "Real Estate"
    if 7000 <= sic <= 7299:
        return "Industrials"
    if 7300 <= sic <= 7399:
        return "Information Technology"
    if 7400 <= sic <= 7999:
        return "Consumer Discretionary"
    if 8000 <= sic <= 8999:
        return "Health Care"
    if 9000 <= sic <= 9999:
        return "Unknown"
    return "Unknown"


def _wrds_max_date(connection) -> pd.Timestamp:
    frame = connection.raw_sql("select max(date) as max_date from crsp.dsf")
    return pd.to_datetime(frame.iloc[0]["max_date"])


def _load_user_watchlist() -> list[str]:
    if USER_WATCHLIST_PATH.exists():
        frame = pd.read_csv(USER_WATCHLIST_PATH)
        if "ticker" in frame.columns:
            tickers = [sanitize_ticker(value) for value in frame["ticker"].dropna().tolist()]
            tickers = [ticker for ticker in tickers if ticker]
            if tickers:
                return sorted(set(tickers))
    return sorted(set(WATCHLIST))


def _read_cached_wrds_prices() -> pd.DataFrame:
    if WRDS_STOCK_CACHE_PARQUET.exists():
        return pd.read_parquet(WRDS_STOCK_CACHE_PARQUET)
    if WRDS_STOCK_CACHE_CSV.exists():
        frame = pd.read_csv(WRDS_STOCK_CACHE_CSV, parse_dates=["date"])
        return frame
    return pd.DataFrame(
        columns=["date", "ticker", "permno", "open", "high", "low", "close", "adj_open", "adj_high", "adj_low", "adj_close", "volume"]
    )


def _persist_wrds_cache(universe_source: pd.DataFrame, price_cache: pd.DataFrame) -> None:
    universe_source.to_csv(WRDS_UNIVERSE_SOURCE_CSV, index=False)
    universe_source.to_parquet(WRDS_UNIVERSE_SOURCE_PARQUET, index=False)
    price_cache.to_csv(WRDS_STOCK_CACHE_CSV, index=False)
    price_cache.to_parquet(WRDS_STOCK_CACHE_PARQUET, index=False)


def _read_recent_market_cache() -> pd.DataFrame:
    if RECENT_MARKET_CACHE_PARQUET.exists():
        return pd.read_parquet(RECENT_MARKET_CACHE_PARQUET)
    if RECENT_MARKET_CACHE_CSV.exists():
        return pd.read_csv(RECENT_MARKET_CACHE_CSV, parse_dates=["date"])
    return pd.DataFrame(
        columns=["date", "ticker", "open", "high", "low", "close", "adj_open", "adj_high", "adj_low", "adj_close", "volume", "source"]
    )


def _persist_recent_market_cache(cache: pd.DataFrame) -> None:
    cache.to_csv(RECENT_MARKET_CACHE_CSV, index=False)
    cache.to_parquet(RECENT_MARKET_CACHE_PARQUET, index=False)


def _fetch_active_common_names(connection, as_of_date: pd.Timestamp) -> pd.DataFrame:
    sql = f"""
        select distinct
            permno,
            permco,
            ticker,
            comnam,
            siccd,
            shrcd,
            exchcd,
            namedt,
            nameenddt
        from crsp.stocknames
        where '{as_of_date.date()}' between namedt and nameenddt
          and shrcd in (10, 11)
          and exchcd in (1, 2, 3)
          and ticker is not null
    """
    names = connection.raw_sql(sql)
    names["ticker"] = names["ticker"].map(sanitize_ticker)
    names = names[names["ticker"] != ""].drop_duplicates("ticker", keep="first").reset_index(drop=True)
    names["company_name"] = names["comnam"].astype(str).str.strip()
    names["sector"] = names["siccd"].map(_sic_to_sector)
    names["index_source"] = "wrds_crsp"
    return names[["permno", "permco", "ticker", "company_name", "sector", "siccd", "index_source"]]


def _fetch_watchlist_names(connection, as_of_date: pd.Timestamp, tickers: list[str]) -> pd.DataFrame:
    ticker_sql = ", ".join(f"'{ticker}'" for ticker in tickers)
    sql = f"""
        select distinct
            permno,
            permco,
            ticker,
            comnam,
            siccd,
            shrcd,
            exchcd,
            namedt,
            nameenddt
        from crsp.stocknames
        where '{as_of_date.date()}' between namedt and nameenddt
          and shrcd in (10, 11)
          and exchcd in (1, 2, 3)
          and upper(ticker) in ({ticker_sql})
    """
    names = connection.raw_sql(sql)
    names["ticker"] = names["ticker"].map(sanitize_ticker)
    names["company_name"] = names["comnam"].astype(str).str.strip()
    names["sector"] = names["siccd"].map(_sic_to_sector)
    names["index_source"] = "wrds_watchlist"
    names = names.drop_duplicates("ticker", keep="first").reset_index(drop=True)
    return names[["permno", "permco", "ticker", "company_name", "sector", "siccd", "index_source"]]


def _fetch_recent_liquidity(connection, as_of_date: pd.Timestamp) -> pd.DataFrame:
    recent_start = as_of_date - timedelta(days=130)
    sql = f"""
        with active_names as (
            select distinct permno
            from crsp.stocknames
            where '{as_of_date.date()}' between namedt and nameenddt
              and shrcd in (10, 11)
              and exchcd in (1, 2, 3)
        )
        select
            d.date,
            d.permno,
            abs(d.prc) / nullif(d.cfacpr, 0) as adj_close,
            d.vol
        from crsp.dsf d
        inner join active_names a
            on d.permno = a.permno
        where d.date between '{recent_start.date()}' and '{as_of_date.date()}'
    """
    recent = connection.raw_sql(sql)
    recent["date"] = pd.to_datetime(recent["date"])
    recent["adj_close"] = pd.to_numeric(recent["adj_close"], errors="coerce")
    recent["vol"] = pd.to_numeric(recent["vol"], errors="coerce").fillna(0.0)
    recent["dollar_volume"] = recent["adj_close"] * recent["vol"]
    return recent


def _build_top_liquid_universe(connection, as_of_date: pd.Timestamp, limit: int | None) -> pd.DataFrame:
    names = _fetch_active_common_names(connection, as_of_date)
    recent = _fetch_recent_liquidity(connection, as_of_date)
    stats = (
        recent.groupby("permno", sort=False)
        .agg(
            latest_date=("date", "max"),
            latest_adj_close=("adj_close", "last"),
            median_dollar_volume_60=("dollar_volume", "median"),
            history_days=("date", "nunique"),
        )
        .reset_index()
    )
    universe = names.merge(stats, on="permno", how="inner")
    universe = universe[
        (universe["latest_adj_close"] >= MIN_PRICE)
        & (universe["median_dollar_volume_60"] >= MIN_MEDIAN_DOLLAR_VOLUME)
        & (universe["history_days"] >= 40)
    ].copy()
    universe = universe.sort_values("median_dollar_volume_60", ascending=False)
    target_count = limit if limit is not None else UNIVERSE_TARGET_COUNT
    return universe.head(target_count).reset_index(drop=True)


def _build_watchlist_universe(connection, as_of_date: pd.Timestamp) -> pd.DataFrame:
    tickers = _load_user_watchlist()
    return _fetch_watchlist_names(connection, as_of_date, tickers)


def _resolve_wrds_universe(
    connection,
    end: str,
    limit: int | None = None,
    universe_mode: str = "watchlist",
) -> tuple[pd.DataFrame, pd.Timestamp]:
    max_date = _wrds_max_date(connection)
    requested_end = pd.to_datetime(end)
    as_of_date = min(requested_end, max_date)
    if universe_mode == "watchlist":
        return _build_watchlist_universe(connection, as_of_date), as_of_date
    return _build_top_liquid_universe(connection, as_of_date, limit=limit), as_of_date


def _fetch_wrds_history_chunk(connection, permnos: list[int], start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    permno_csv = ", ".join(str(value) for value in permnos)
    sql = f"""
        select
            d.date,
            d.permno,
            abs(d.openprc) as open,
            abs(d.askhi) as high,
            abs(d.bidlo) as low,
            abs(d.prc) as close,
            abs(d.openprc) / nullif(d.cfacpr, 0) as adj_open,
            abs(d.askhi) / nullif(d.cfacpr, 0) as adj_high,
            abs(d.bidlo) / nullif(d.cfacpr, 0) as adj_low,
            abs(d.prc) / nullif(d.cfacpr, 0) as adj_close,
            d.vol as volume
        from crsp.dsf d
        where d.permno in ({permno_csv})
          and d.date between '{start_date.date()}' and '{end_date.date()}'
    """
    frame = connection.raw_sql(sql)
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["date"])
    frame["permno"] = frame["permno"].astype(int)
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0)
    for column in ("open", "high", "low", "close", "adj_open", "adj_high", "adj_low", "adj_close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _update_wrds_price_cache(
    connection,
    universe_source: pd.DataFrame,
    start: str,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, int]:
    cache = _read_cached_wrds_prices()
    if not cache.empty:
        cache["date"] = pd.to_datetime(cache["date"])
        cache["permno"] = cache["permno"].astype(int)

    requested = universe_source[["permno", "ticker"]].drop_duplicates("permno").copy()
    existing_last_dates = (
        cache.groupby("permno", sort=False)["date"].max().reset_index().rename(columns={"date": "last_cached_date"})
        if not cache.empty
        else pd.DataFrame(columns=["permno", "last_cached_date"])
    )
    fetch_plan = requested.merge(existing_last_dates, on="permno", how="left")
    fetch_plan["fetch_start"] = fetch_plan["last_cached_date"].apply(
        lambda value: pd.to_datetime(start) if pd.isna(value) else (pd.to_datetime(value) + timedelta(days=1))
    )
    fetch_plan = fetch_plan[fetch_plan["fetch_start"] <= end].copy()

    new_rows: list[pd.DataFrame] = []
    for fetch_start, group in fetch_plan.groupby("fetch_start", sort=True):
        permnos = sorted(int(value) for value in group["permno"].tolist())
        for chunk in _chunked(permnos, size=250):
            frame = _fetch_wrds_history_chunk(connection, chunk, start_date=pd.to_datetime(fetch_start), end_date=end)
            if not frame.empty:
                new_rows.append(frame)

    appended_rows = 0
    if new_rows:
        fresh = pd.concat(new_rows, ignore_index=True)
        appended_rows = int(len(fresh))
        cache = fresh if cache.empty else pd.concat([cache, fresh], ignore_index=True)

    cache = cache.drop_duplicates(subset=["permno", "date"], keep="last").sort_values(["permno", "date"]).reset_index(drop=True)
    cache["ticker"] = cache["permno"].map(universe_source.set_index("permno")["ticker"].to_dict()).fillna(cache.get("ticker"))
    _persist_wrds_cache(universe_source, cache)
    return cache, appended_rows


def _align_recent_prices_to_wrds(
    recent_frame: pd.DataFrame,
    wrds_cache: pd.DataFrame,
    wrds_end: pd.Timestamp,
) -> pd.DataFrame:
    if recent_frame.empty:
        return recent_frame
    adjusted = recent_frame.copy()
    overlap_recent = adjusted[adjusted["date"] <= wrds_end].copy()
    overlap_wrds = wrds_cache[wrds_cache["date"] <= wrds_end].copy()
    if overlap_recent.empty or overlap_wrds.empty:
        adjusted["source"] = "yfinance_recent"
        return adjusted

    recent_anchor = overlap_recent.sort_values("date").groupby("ticker", sort=False).tail(1)[["ticker", "adj_close"]]
    wrds_anchor = overlap_wrds.sort_values("date").groupby("ticker", sort=False).tail(1)[["ticker", "adj_close"]]
    anchor = wrds_anchor.merge(recent_anchor, on="ticker", how="inner", suffixes=("_wrds", "_recent"))
    anchor["scale"] = anchor["adj_close_wrds"] / anchor["adj_close_recent"].replace(0.0, pd.NA)
    scale_map = anchor.set_index("ticker")["scale"].replace([pd.NA], 1.0).fillna(1.0).to_dict()

    for column in ("open", "high", "low", "close", "adj_open", "adj_high", "adj_low", "adj_close"):
        adjusted[column] = adjusted[column] * adjusted["ticker"].map(scale_map).fillna(1.0)
    adjusted["source"] = "yfinance_recent"
    return adjusted


def _download_recent_prices_resilient(
    tickers: list[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> tuple[pd.DataFrame, list[str]]:
    if not tickers:
        return pd.DataFrame(), []
    try:
        frame = download_price_history(
            tickers,
            start=str(start_date.date()),
            end=str((end_date + timedelta(days=1)).date()),
        )
        return frame, []
    except Exception:
        frames: list[pd.DataFrame] = []
        failed: list[str] = []
        for ticker in tickers:
            try:
                single = download_price_history(
                    [ticker],
                    start=str(start_date.date()),
                    end=str((end_date + timedelta(days=1)).date()),
                )
                if single.empty:
                    failed.append(ticker)
                    continue
                frames.append(single)
            except Exception:
                failed.append(ticker)
        if not frames:
            return pd.DataFrame(), failed
        return pd.concat(frames, ignore_index=True), failed


def _download_recent_prices_with_aliases(
    tickers: list[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> tuple[pd.DataFrame, list[str]]:
    initial, failed = _download_recent_prices_resilient(tickers, start_date, end_date)
    frames: list[pd.DataFrame] = [initial] if not initial.empty else []
    resolved: list[str] = []
    unresolved: list[str] = []

    for ticker in failed:
        aliases = RECENT_SYMBOL_ALIASES.get(ticker, [])
        alias_success = False
        for alias in aliases:
            alias_frame, alias_failed = _download_recent_prices_resilient([alias], start_date, end_date)
            if alias_frame.empty or alias_failed:
                continue
            alias_frame = alias_frame.copy()
            alias_frame["ticker"] = ticker
            frames.append(alias_frame)
            resolved.append(ticker)
            alias_success = True
            break
        if not alias_success:
            unresolved.append(ticker)

    if not frames:
        return pd.DataFrame(), unresolved
    combined = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["ticker", "date"], keep="last")
    return combined.sort_values(["ticker", "date"]).reset_index(drop=True), sorted(set(unresolved))


def _update_recent_stock_cache(
    selected_tickers: list[str],
    wrds_cache: pd.DataFrame,
    wrds_end: pd.Timestamp,
    requested_end: pd.Timestamp,
) -> tuple[pd.DataFrame, int, list[str]]:
    if requested_end <= wrds_end:
        cache = _read_recent_market_cache()
        if not cache.empty:
            cache["date"] = pd.to_datetime(cache["date"])
        return cache, 0, []

    cache = _read_recent_market_cache()
    if not cache.empty:
        cache["date"] = pd.to_datetime(cache["date"])

    existing_last_dates = (
        cache.groupby("ticker", sort=False)["date"].max().reset_index().rename(columns={"date": "last_cached_date"})
        if not cache.empty
        else pd.DataFrame(columns=["ticker", "last_cached_date"])
    )
    request = pd.DataFrame({"ticker": sorted(set(selected_tickers))})
    fetch_plan = request.merge(existing_last_dates, on="ticker", how="left")
    overlap_start = wrds_end - timedelta(days=20)
    fetch_plan["fetch_start"] = fetch_plan["last_cached_date"].apply(
        lambda value: overlap_start if pd.isna(value) else (pd.to_datetime(value) + timedelta(days=1))
    )
    fetch_plan = fetch_plan[fetch_plan["fetch_start"] <= requested_end].copy()

    new_rows: list[pd.DataFrame] = []
    failed_tickers: list[str] = []
    for fetch_start, group in fetch_plan.groupby("fetch_start", sort=True):
        tickers = group["ticker"].tolist()
        fresh, failed = _download_recent_prices_with_aliases(
            tickers=tickers,
            start_date=pd.to_datetime(fetch_start),
            end_date=requested_end,
        )
        failed_tickers.extend(failed)
        if fresh.empty:
            continue
        new_rows.append(_align_recent_prices_to_wrds(fresh, wrds_cache, wrds_end))

    appended_rows = 0
    if new_rows:
        fresh_all = pd.concat(new_rows, ignore_index=True)
        appended_rows = int(len(fresh_all))
        cache = fresh_all if cache.empty else pd.concat([cache, fresh_all], ignore_index=True)

    if not cache.empty:
        cache = cache.drop_duplicates(subset=["ticker", "date"], keep="last").sort_values(["ticker", "date"]).reset_index(drop=True)
        _persist_recent_market_cache(cache)
    return cache, appended_rows, sorted(set(failed_tickers))


def refresh_data_from_wrds(
    start: str,
    end: str,
    limit: int | None = None,
    universe_mode: str = "watchlist",
) -> WrdsRefreshArtifacts:
    connection = connect_wrds()
    try:
        universe_source, actual_end = _resolve_wrds_universe(connection, end=end, limit=limit, universe_mode=universe_mode)
        if universe_source.empty:
            raise ValueError("The WRDS universe query returned no stocks for the current configuration.")
        wrds_cache, wrds_appended_rows = _update_wrds_price_cache(connection, universe_source, start=start, end=actual_end)
    finally:
        connection.close()

    selected_permnos = set(universe_source["permno"].astype(int))
    wrds_stock_prices = wrds_cache[
        (wrds_cache["permno"].isin(selected_permnos))
        & (wrds_cache["date"] >= pd.to_datetime(start))
        & (wrds_cache["date"] <= actual_end)
    ].copy()
    wrds_stock_prices["ticker"] = wrds_stock_prices["permno"].map(universe_source.set_index("permno")["ticker"].to_dict())

    requested_end = pd.to_datetime(end)
    recent_cache, recent_appended_rows, recent_failed_tickers = _update_recent_stock_cache(
        selected_tickers=universe_source["ticker"].astype(str).tolist(),
        wrds_cache=wrds_stock_prices,
        wrds_end=actual_end,
        requested_end=requested_end,
    )
    recent_stock_prices = recent_cache[
        (recent_cache["ticker"].isin(universe_source["ticker"]))
        & (recent_cache["date"] > actual_end)
        & (recent_cache["date"] >= pd.to_datetime(start))
        & (recent_cache["date"] <= requested_end)
    ].copy()
    stock_prices = pd.concat(
        [
            wrds_stock_prices.drop(columns=["permno"], errors="ignore"),
            recent_stock_prices.drop(columns=["source"], errors="ignore"),
        ],
        ignore_index=True,
    ).sort_values(["ticker", "date"]).reset_index(drop=True)

    sector_etfs = sorted(set(SECTOR_ETF_MAP.values()))
    benchmark_end = max(actual_end, requested_end)
    benchmark_prices = download_price_history(
        list(BENCHMARK_TICKERS) + sector_etfs,
        start=start,
        end=str((benchmark_end + timedelta(days=1)).date()),
    )

    price_panel = pd.concat(
        [
            stock_prices.drop(columns=["permno"], errors="ignore"),
            benchmark_prices,
        ],
        ignore_index=True,
    ).sort_values(["ticker", "date"]).reset_index(drop=True)

    summary = {
        "provider": "wrds",
        "universe_mode": universe_mode,
        "requested_end": end,
        "actual_wrds_end": str(actual_end.date()),
        "latest_combined_date": str(pd.to_datetime(price_panel["date"]).max().date()),
        "universe_source_rows": int(len(universe_source)),
        "stock_tickers_from_wrds": int(stock_prices["ticker"].nunique()),
        "downloaded_tickers": int(price_panel["ticker"].nunique()),
        "price_rows": int(len(price_panel)),
        "wrds_cache_rows": int(len(wrds_cache)),
        "wrds_rows_appended": int(wrds_appended_rows),
        "recent_extension_rows": int(len(recent_stock_prices)),
        "recent_rows_appended": int(recent_appended_rows),
        "recent_failed_tickers": recent_failed_tickers,
        "wrds_cache_csv": str(WRDS_STOCK_CACHE_CSV),
        "wrds_universe_csv": str(WRDS_UNIVERSE_SOURCE_CSV),
        "recent_extension_cache_csv": str(RECENT_MARKET_CACHE_CSV),
    }
    return WrdsRefreshArtifacts(universe_source=universe_source, price_panel=price_panel, summary=summary)
