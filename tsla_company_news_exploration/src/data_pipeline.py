from __future__ import annotations

import json
import time
import xml.etree.ElementTree as ET
from calendar import monthrange
from datetime import date, timedelta
from email.utils import parsedate_to_datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from .config import (
    COLLECTION_SUMMARY_PATH,
    COMPANY_NAME,
    DAILY_DATASET_PARQUET,
    DATASET_SUMMARY_PATH,
    GOOGLE_NEWS_RSS_URL,
    HISTORY_RANGE,
    NEWS_LOCALE_PARAMS,
    NEWS_MONTH_SLEEP_SECONDS,
    NEWS_QUERY,
    PRICE_HISTORY_CSV,
    PRICE_HISTORY_PARQUET,
    PRICE_INTERVAL,
    RAW_NEWS_CSV,
    RAW_NEWS_DIR,
    RAW_NEWS_PARQUET,
    RAW_PRICE_JSON,
    TIMEZONE_NAME,
    TICKER,
    YAHOO_PRICE_URL,
    ensure_project_dirs,
)
from .evaluation import write_json
from .text_utils import normalize_text, title_mentions_tesla


USER_AGENT = {"User-Agent": "Mozilla/5.0"}


def _collect_price_history() -> pd.DataFrame:
    if PRICE_HISTORY_PARQUET.exists():
        frame = pd.read_parquet(PRICE_HISTORY_PARQUET)
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        return frame
    response = requests.get(
        YAHOO_PRICE_URL,
        params={
            "range": HISTORY_RANGE,
            "interval": PRICE_INTERVAL,
            "includePrePost": "false",
            "events": "div,splits,capitalGains",
        },
        timeout=30,
        headers=USER_AGENT,
    )
    response.raise_for_status()
    payload = response.json()
    RAW_PRICE_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    result = payload["chart"]["result"][0]
    timestamps = pd.to_datetime(result["timestamp"], unit="s", utc=True).tz_convert(TIMEZONE_NAME)
    quotes = pd.DataFrame(result["indicators"]["quote"][0])
    quotes["trade_date"] = timestamps.tz_localize(None).normalize()
    if "adjclose" in result["indicators"]:
        quotes["adjclose"] = result["indicators"]["adjclose"][0]["adjclose"]
    else:
        quotes["adjclose"] = quotes["close"]
    quotes["ticker"] = TICKER
    quotes = quotes[["trade_date", "ticker", "open", "high", "low", "close", "adjclose", "volume"]]
    quotes = quotes.dropna(subset=["trade_date", "adjclose"]).sort_values("trade_date").reset_index(drop=True)
    quotes.to_parquet(PRICE_HISTORY_PARQUET, index=False)
    quotes.to_csv(PRICE_HISTORY_CSV, index=False)
    return quotes


def _month_windows(start_date: date, end_date: date) -> list[tuple[date, date]]:
    windows: list[tuple[date, date]] = []
    current = date(start_date.year, start_date.month, 1)
    while current <= end_date:
        month_end = date(current.year, current.month, monthrange(current.year, current.month)[1])
        next_month = month_end + timedelta(days=1)
        windows.append((current, min(next_month, end_date + timedelta(days=1))))
        current = next_month
    return windows


def _collect_news_for_window(window_start: date, window_end: date) -> list[dict[str, object]]:
    params = {
        "q": f"{NEWS_QUERY} after:{window_start.isoformat()} before:{window_end.isoformat()}",
        **NEWS_LOCALE_PARAMS,
    }
    response = requests.get(GOOGLE_NEWS_RSS_URL, params=params, timeout=30, headers=USER_AGENT)
    response.raise_for_status()
    raw_xml_path = RAW_NEWS_DIR / f"{window_start.isoformat()}_{window_end.isoformat()}.xml"
    raw_xml_path.write_text(response.text, encoding="utf-8")

    root = ET.fromstring(response.text)
    rows: list[dict[str, object]] = []
    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub_date = item.findtext("pubDate")
        publisher = item.findtext("./source")
        if not title or not pub_date or not title_mentions_tesla(title):
            continue
        published_utc = parsedate_to_datetime(pub_date)
        if published_utc.tzinfo is None:
            published_utc = published_utc.replace(tzinfo=ZoneInfo("UTC"))
        published_local = published_utc.astimezone(ZoneInfo(TIMEZONE_NAME))
        rows.append(
            {
                "title": title,
                "normalized_title": normalize_text(title),
                "link": link,
                "publisher": publisher,
                "published_at_utc": published_utc.isoformat(),
                "published_at_local": published_local.isoformat(),
                "published_date_local": published_local.date().isoformat(),
            }
        )
    return rows


def _collect_headlines(price_history: pd.DataFrame) -> pd.DataFrame:
    if RAW_NEWS_PARQUET.exists():
        frame = pd.read_parquet(RAW_NEWS_PARQUET)
        frame["published_at_local"] = pd.to_datetime(frame["published_at_local"], utc=True).dt.tz_convert(TIMEZONE_NAME)
        return frame
    min_date = price_history["trade_date"].min().date()
    max_date = price_history["trade_date"].max().date()
    rows: list[dict[str, object]] = []
    for window_start, window_end in _month_windows(min_date, max_date):
        rows.extend(_collect_news_for_window(window_start, window_end))
        time.sleep(NEWS_MONTH_SLEEP_SECONDS)

    headlines = pd.DataFrame(rows)
    if headlines.empty:
        raise RuntimeError("No TSLA headlines were collected from Google News RSS.")
    headlines["published_at_local"] = pd.to_datetime(headlines["published_at_local"], utc=True).dt.tz_convert(TIMEZONE_NAME)
    headlines["published_minute"] = headlines["published_at_local"].dt.floor("min")
    headlines = (
        headlines.sort_values(["published_at_local", "title"])
        .drop_duplicates(subset=["normalized_title", "published_minute"])
        .reset_index(drop=True)
    )
    headlines["headline_id"] = [f"headline_{idx:06d}" for idx in range(len(headlines))]
    headlines.to_parquet(RAW_NEWS_PARQUET, index=False)
    headlines.to_csv(RAW_NEWS_CSV, index=False)
    return headlines


def run_collect_data() -> dict[str, object]:
    ensure_project_dirs()
    price_history = _collect_price_history()
    headlines = _collect_headlines(price_history)
    summary = {
        "ticker": TICKER,
        "company_name": COMPANY_NAME,
        "price_history_rows": int(len(price_history)),
        "price_date_range": {
            "min_date": price_history["trade_date"].min().date().isoformat(),
            "max_date": price_history["trade_date"].max().date().isoformat(),
        },
        "headline_rows": int(len(headlines)),
        "headline_date_range": {
            "min_date": headlines["published_at_local"].min().date().isoformat(),
            "max_date": headlines["published_at_local"].max().date().isoformat(),
        },
        "headline_publishers": int(headlines["publisher"].fillna("").nunique()),
    }
    write_json(COLLECTION_SUMMARY_PATH, summary)
    return summary


def _load_collected_prices() -> pd.DataFrame:
    if not PRICE_HISTORY_PARQUET.exists():
        run_collect_data()
    frame = pd.read_parquet(PRICE_HISTORY_PARQUET)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    return frame


def _load_collected_headlines() -> pd.DataFrame:
    if not RAW_NEWS_PARQUET.exists():
        run_collect_data()
    frame = pd.read_parquet(RAW_NEWS_PARQUET)
    frame["published_at_local"] = pd.to_datetime(frame["published_at_local"], utc=True).dt.tz_convert(TIMEZONE_NAME)
    return frame


def _assign_headlines_to_trading_days(headlines: pd.DataFrame, trade_dates: pd.Series) -> pd.DataFrame:
    sorted_trade_dates = pd.Series(pd.to_datetime(trade_dates).sort_values().unique())

    def assign_trade_date(published_at_local: pd.Timestamp):
        candidate_date = published_at_local.tz_localize(None).normalize()
        if published_at_local.hour >= 16:
            candidate_date = candidate_date + pd.Timedelta(days=1)
        idx = sorted_trade_dates.searchsorted(candidate_date)
        if idx >= len(sorted_trade_dates):
            return pd.NaT
        return sorted_trade_dates.iloc[int(idx)]

    assigned = headlines.copy()
    assigned["assigned_trade_date"] = assigned["published_at_local"].map(assign_trade_date)
    return assigned.dropna(subset=["assigned_trade_date"]).reset_index(drop=True)


def build_daily_dataset() -> dict[str, object]:
    ensure_project_dirs()
    prices = _load_collected_prices()
    headlines = _load_collected_headlines()
    assigned = _assign_headlines_to_trading_days(headlines, prices["trade_date"])

    price_frame = prices.sort_values("trade_date").reset_index(drop=True).copy()
    price_frame["close_t"] = price_frame["adjclose"].astype(float)
    price_frame["close_t_plus_1"] = price_frame["close_t"].shift(-1)
    price_frame["next_day_return"] = (price_frame["close_t_plus_1"] - price_frame["close_t"]) / price_frame["close_t"]
    price_frame["ret_1d"] = price_frame["close_t"].pct_change(1)
    price_frame["ret_5d"] = price_frame["close_t"].pct_change(5)
    price_frame["vol_5d"] = price_frame["ret_1d"].rolling(5).std()

    grouped = assigned.sort_values(["assigned_trade_date", "published_at_local"]).groupby("assigned_trade_date")
    headline_daily = grouped.agg(
        headline_count=("headline_id", "count"),
        daily_text=("title", lambda values: " || ".join(map(str, values))),
    ).reset_index().rename(columns={"assigned_trade_date": "trade_date"})

    dataset = price_frame.merge(headline_daily, on="trade_date", how="left")
    dataset["headline_count"] = dataset["headline_count"].fillna(0).astype(int)
    dataset["daily_text"] = dataset["daily_text"].fillna("")
    dataset["ret_1d"] = dataset["ret_1d"].fillna(0.0)
    dataset["ret_5d"] = dataset["ret_5d"].fillna(0.0)
    dataset["vol_5d"] = dataset["vol_5d"].fillna(0.0)
    dataset = dataset.dropna(subset=["close_t_plus_1"]).reset_index(drop=True)
    dataset = dataset[dataset["close_t_plus_1"] != dataset["close_t"]].reset_index(drop=True)
    dataset["direction_label"] = (dataset["close_t_plus_1"] > dataset["close_t"]).astype(int)
    dataset.to_parquet(DAILY_DATASET_PARQUET, index=False)

    summary = {
        "row_count": int(len(dataset)),
        "trade_date_range": {
            "min_date": dataset["trade_date"].min().date().isoformat(),
            "max_date": dataset["trade_date"].max().date().isoformat(),
        },
        "headline_days": int((dataset["headline_count"] > 0).sum()),
        "empty_headline_days": int((dataset["headline_count"] == 0).sum()),
        "direction_label_counts": {
            "down": int((dataset["direction_label"] == 0).sum()),
            "up": int((dataset["direction_label"] == 1).sum()),
        },
    }
    write_json(DATASET_SUMMARY_PATH, summary)
    return summary


def load_daily_dataset() -> pd.DataFrame:
    if not DAILY_DATASET_PARQUET.exists():
        build_daily_dataset()
    frame = pd.read_parquet(DAILY_DATASET_PARQUET)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    return frame


def compute_chronological_splits(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    frame = frame.sort_values("trade_date").reset_index(drop=True)
    total = len(frame)
    train_end = int(total * 0.70)
    validation_end = int(total * 0.85)
    splits = {
        "train": frame.iloc[:train_end].reset_index(drop=True),
        "validation": frame.iloc[train_end:validation_end].reset_index(drop=True),
        "test": frame.iloc[validation_end:].reset_index(drop=True),
    }
    if any(split.empty for split in splits.values()):
        raise RuntimeError("Chronological split produced an empty split.")
    return splits
