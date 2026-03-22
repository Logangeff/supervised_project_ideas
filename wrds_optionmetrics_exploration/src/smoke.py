from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    BUILD_OPTION_FEATURES_SUMMARY_PATH,
    BUILD_STOCK_PANEL_SUMMARY_PATH,
    COMPLETE_CASE_PANEL_PATH,
    EXTRACT_OPTION_SUMMARY_PATH,
    EXTRACT_STOCK_SUMMARY_PATH,
    OPTION_FEATURE_PANEL_PATH,
    OPTION_SECURITY_LINK_PATH,
    PART1_METRICS_CSV,
    PART1_METRICS_JSON,
    PART1_PERSISTENCE_FIGURE,
    PART1_PREDICTIONS_CSV,
    PART1_STOCK_LOGREG_FIGURE,
    PART1_STOCK_MODEL_PATH,
    PART2_COMBINED_LOGREG_FIGURE,
    PART2_COMBINED_MODEL_PATH,
    PART2_METRICS_CSV,
    PART2_METRICS_JSON,
    PART2_OPTION_LOGREG_FIGURE,
    PART2_OPTION_MODEL_PATH,
    PART2_PREDICTIONS_CSV,
    PART2_STOCK_LOGREG_FIGURE,
    PART2_STOCK_MODEL_PATH,
    RAW_OPTION_PRICES_PATH,
    RAW_OPTION_SECURITIES_PATH,
    RAW_STOCK_DAILY_PATH,
    RAW_STOCK_NAMES_PATH,
    SEED,
    SMOKE_SUMMARY_PATH,
    STOCK_PANEL_PATH,
    STOCK_IDENTIFIER_MAP_PATH,
    ensure_project_dirs,
)
from .option_pipeline import build_option_features, extract_option_data, train_part2
from .stock_pipeline import build_stock_panel, extract_stock_data, train_part1
from .utils import write_json


@dataclass
class BackupEntry:
    original: Path
    backup: Path


def _protected_paths() -> list[Path]:
    option_yearly_paths = sorted(RAW_OPTION_SECURITIES_PATH.parent.glob("optionm_option_prices_*.parquet"))
    return [
        RAW_STOCK_NAMES_PATH,
        RAW_STOCK_DAILY_PATH,
        STOCK_IDENTIFIER_MAP_PATH,
        RAW_OPTION_SECURITIES_PATH,
        RAW_OPTION_PRICES_PATH,
        OPTION_SECURITY_LINK_PATH,
        BUILD_STOCK_PANEL_SUMMARY_PATH,
        EXTRACT_STOCK_SUMMARY_PATH,
        STOCK_PANEL_PATH,
        PART1_METRICS_JSON,
        PART1_METRICS_CSV,
        PART1_PREDICTIONS_CSV,
        PART1_STOCK_MODEL_PATH,
        PART1_PERSISTENCE_FIGURE,
        PART1_STOCK_LOGREG_FIGURE,
        EXTRACT_OPTION_SUMMARY_PATH,
        BUILD_OPTION_FEATURES_SUMMARY_PATH,
        OPTION_FEATURE_PANEL_PATH,
        COMPLETE_CASE_PANEL_PATH,
        PART2_METRICS_JSON,
        PART2_METRICS_CSV,
        PART2_PREDICTIONS_CSV,
        PART2_STOCK_MODEL_PATH,
        PART2_OPTION_MODEL_PATH,
        PART2_COMBINED_MODEL_PATH,
        PART2_STOCK_LOGREG_FIGURE,
        PART2_OPTION_LOGREG_FIGURE,
        PART2_COMBINED_LOGREG_FIGURE,
    ] + option_yearly_paths


def _backup_existing_files() -> list[BackupEntry]:
    entries: list[BackupEntry] = []
    for path in _protected_paths():
        if path.exists():
            backup = path.with_name(path.name + ".smoke_backup")
            if backup.exists():
                if backup.is_file():
                    backup.unlink()
                else:
                    shutil.rmtree(backup)
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(backup))
            entries.append(BackupEntry(original=path, backup=backup))
    return entries


def _restore_backups(entries: list[BackupEntry]) -> None:
    for entry in _protected_paths():
        if entry.exists() and entry.is_file() and entry.suffix != ".smoke_backup":
            entry.unlink()
    for backup in entries:
        if backup.backup.exists():
            backup.original.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(backup.backup), str(backup.original))


def _write_synthetic_stock_raw() -> None:
    universe_rows = [
        {"universe_ticker": "AAPL", "company_name": "Apple Inc.", "sector": "Information Technology", "permno": 10001, "crsp_ticker": "AAPL", "comnam": "Apple Inc.", "ncusip8": "03783310"},
        {"universe_ticker": "MSFT", "company_name": "Microsoft", "sector": "Information Technology", "permno": 10002, "crsp_ticker": "MSFT", "comnam": "Microsoft Corp.", "ncusip8": "59491810"},
        {"universe_ticker": "JPM", "company_name": "JPMorgan Chase", "sector": "Financials", "permno": 10003, "crsp_ticker": "JPM", "comnam": "JPMorgan Chase", "ncusip8": "46625H10"},
        {"universe_ticker": "XOM", "company_name": "Exxon Mobil", "sector": "Energy", "permno": 10004, "crsp_ticker": "XOM", "comnam": "Exxon Mobil Corp.", "ncusip8": "30231G10"},
    ]
    stock_names = [
        {"permno": row["permno"], "ticker": row["crsp_ticker"], "comnam": row["comnam"], "ncusip": row["ncusip8"], "namedt": "2010-01-01", "nameenddt": "2099-12-31", "shrcd": 10, "exchcd": 1}
        for row in universe_rows
    ]
    pd.DataFrame(stock_names).to_parquet(RAW_STOCK_NAMES_PATH, index=False)
    pd.DataFrame(universe_rows).to_parquet(STOCK_IDENTIFIER_MAP_PATH, index=False)

    dates = pd.bdate_range("2016-01-01", "2024-12-31")
    n_dates = len(dates)
    stock_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(SEED)
    base_prices = {10001: 110.0, 10002: 90.0, 10003: 75.0, 10004: 60.0}
    firm_offset = {10001: 0.2, 10002: -0.1, 10003: 0.0, 10004: 0.35}

    for permno, start_price in base_prices.items():
        noise = rng.normal(0.0, 0.45, size=n_dates)
        latent = np.zeros(n_dates)
        for idx in range(1, n_dates):
            latent[idx] = 0.97 * latent[idx - 1] + noise[idx]
        future_signal = np.roll(latent, -8)
        sigma = 0.009 + 0.005 * np.maximum(latent, 0.0) + 0.008 * np.maximum(future_signal, 0.0)
        sigma = np.clip(sigma + firm_offset[permno] * 0.001, 0.008, 0.08)

        price = start_price
        for idx, trade_date in enumerate(dates):
            ret = rng.normal(0.0002, sigma[idx])
            price = max(5.0, price * (1.0 + ret))
            volume = int(abs(rng.normal(7_000_000 + permno, 900_000)))
            shrout = 4_000_000 + (permno - 10000) * 500_000
            stock_rows.append(
                {
                    "permno": permno,
                    "trade_date": trade_date,
                    "ret": ret,
                    "prc": price,
                    "vol": volume,
                    "shrout": shrout,
                }
            )
    pd.DataFrame(stock_rows).to_parquet(RAW_STOCK_DAILY_PATH, index=False)


def _write_synthetic_option_raw() -> None:
    identifier_map = pd.read_parquet(STOCK_IDENTIFIER_MAP_PATH)
    stock_daily = pd.read_parquet(RAW_STOCK_DAILY_PATH).sort_values(["permno", "trade_date"]).reset_index(drop=True)
    stock_daily["trade_date"] = pd.to_datetime(stock_daily["trade_date"])

    securities = []
    security_link = []
    option_rows = []
    rng = np.random.default_rng(SEED + 1)

    for idx, row in identifier_map.reset_index(drop=True).iterrows():
        secid = 20001 + idx
        securities.append(
            {
                "secid": secid,
                "ticker": row["crsp_ticker"],
                "issuer": row["company_name"],
                "cusip": row["ncusip8"],
            }
        )
        security_link.append(
            {
                "secid": secid,
                "permno": int(row["permno"]),
                "universe_ticker": row["universe_ticker"],
                "match_method": "cusip8",
            }
        )

        local = stock_daily[stock_daily["permno"] == row["permno"]].copy()
        local["future_sigma_proxy"] = local["ret"].rolling(20).std(ddof=1).shift(-5).fillna(local["ret"].rolling(20).std(ddof=1))
        local["future_sigma_proxy"] = local["future_sigma_proxy"].bfill().fillna(local["ret"].std())
        base_level = 0.18 + 0.02 * idx
        for _, obs in local.iterrows():
            trade_date = pd.Timestamp(obs["trade_date"])
            signal = float(np.clip(base_level + 4.5 * float(obs["future_sigma_proxy"]) + rng.normal(0.0, 0.005), 0.10, 0.90))
            short_ex = trade_date + pd.Timedelta(days=30)
            long_ex = trade_date + pd.Timedelta(days=60)
            option_rows.extend(
                [
                    {"secid": secid, "trade_date": trade_date, "exdate": short_ex, "cp_flag": "C", "delta": 0.48, "impl_volatility": signal + 0.005, "open_interest": 1200, "volume": 220, "best_bid": 1.20, "best_offer": 1.40},
                    {"secid": secid, "trade_date": trade_date, "exdate": short_ex, "cp_flag": "P", "delta": -0.52, "impl_volatility": signal + 0.008, "open_interest": 1350, "volume": 240, "best_bid": 1.30, "best_offer": 1.50},
                    {"secid": secid, "trade_date": trade_date, "exdate": short_ex, "cp_flag": "P", "delta": -0.25, "impl_volatility": signal + 0.030, "open_interest": 920, "volume": 170, "best_bid": 0.90, "best_offer": 1.08},
                    {"secid": secid, "trade_date": trade_date, "exdate": short_ex, "cp_flag": "P", "delta": -0.20, "impl_volatility": signal + 0.026, "open_interest": 860, "volume": 150, "best_bid": 0.84, "best_offer": 1.02},
                    {"secid": secid, "trade_date": trade_date, "exdate": long_ex, "cp_flag": "C", "delta": 0.49, "impl_volatility": signal - 0.010, "open_interest": 1020, "volume": 140, "best_bid": 1.35, "best_offer": 1.55},
                    {"secid": secid, "trade_date": trade_date, "exdate": long_ex, "cp_flag": "P", "delta": -0.51, "impl_volatility": signal - 0.012, "open_interest": 1080, "volume": 145, "best_bid": 1.38, "best_offer": 1.58},
                ]
            )

    pd.DataFrame(securities).to_parquet(RAW_OPTION_SECURITIES_PATH, index=False)
    pd.DataFrame(security_link).to_parquet(OPTION_SECURITY_LINK_PATH, index=False)
    pd.DataFrame(option_rows).to_parquet(RAW_OPTION_PRICES_PATH, index=False)


def run_smoke_check() -> dict[str, object]:
    ensure_project_dirs()
    backups = _backup_existing_files()
    try:
        _write_synthetic_stock_raw()
        _write_synthetic_option_raw()

        stock_extract = extract_stock_data()
        stock_panel = build_stock_panel()
        part1 = train_part1()
        option_extract = extract_option_data()
        option_build = build_option_features()
        part2 = train_part2()

        summary = {
            "smoke_mode": True,
            "stock_extract_source": stock_extract["source"],
            "option_extract_source": option_extract["source"],
            "stock_panel_rows": stock_panel["panel_rows"],
            "complete_case_rows": option_build["complete_case_rows"],
            "part1_test_macro_f1": float(part1["models"]["stock_only_logreg"]["test"]["macro_f1"]),
            "part1_test_pr_auc": float(part1["models"]["stock_only_logreg"]["test"]["pr_auc"]),
            "part2_test_macro_f1": {
                model_name: float(model_summary["test"]["macro_f1"])
                for model_name, model_summary in part2["models"].items()
            },
            "part2_test_pr_auc": {
                model_name: float(model_summary["test"]["pr_auc"])
                for model_name, model_summary in part2["models"].items()
            },
        }
        write_json(SMOKE_SUMMARY_PATH, summary)
        return summary
    finally:
        _restore_backups(backups)
