from __future__ import annotations

import argparse
import json
from datetime import datetime
from datetime import date

import pandas as pd

from .backtest import run_backtest
from .config import (
    BACKTEST_SUMMARY_PATH,
    BENCHMARK_TICKERS,
    CURRENT_EXPERIMENTAL_HOLDINGS_PATH,
    CURRENT_HOLDINGS_PATH,
    DATA_PROVIDER,
    EXPERIMENTAL_BACKTEST_SUMMARY_PATH,
    EXPERIMENTAL_PORTFOLIO_HISTORY_PATH,
    EXPERIMENTAL_TRADE_LOG_PATH,
    FEATURE_PANEL_PATH,
    LATEST_PORTFOLIO_PATH,
    LATEST_EXPERIMENTAL_PORTFOLIO_PATH,
    PIPELINE_SUMMARY_PATH,
    PRICE_PANEL_PATH,
    PROCESSED_DIR,
    SCORE_PANEL_PATH,
    PORTFOLIO_HISTORY_PATH,
    PROCESSED_UNIVERSE_SOURCE_PATH,
    TRADE_LOG_PATH,
    RUN_MANIFEST_PATH,
    START_DATE,
    UNIVERSE_MODE,
    UNIVERSE_MEMBERSHIP_PATH,
    UNIVERSE_SOURCE_PATH,
    WATCHLIST,
    ensure_output_dirs,
    save_json,
)
from .data_provider import build_download_ticker_list, download_price_history, get_universe_source
from .features import compute_feature_panel
from .reporting import write_reports
from .ranking import compute_score_panel
from .universe import build_latest_universe, build_point_in_time_universe
from .wrds_provider import refresh_data_from_wrds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Momentum stock strategy and screener pipeline")
    parser.add_argument(
        "--phase",
        required=True,
        choices=["refresh_data", "build_features", "score", "backtest", "report", "all", "smoke"],
        help="Pipeline phase to run.",
    )
    parser.add_argument("--start", default=START_DATE, help="History start date in YYYY-MM-DD.")
    parser.add_argument("--end", default=str(date.today()), help="History end date in YYYY-MM-DD.")
    parser.add_argument("--limit", type=int, default=None, help="Optional ticker limit for smoke or quick tests.")
    parser.add_argument(
        "--universe-mode",
        default=UNIVERSE_MODE,
        choices=["watchlist", "top_liquid"],
        help="Universe selection mode for WRDS runs.",
    )
    return parser.parse_args()


def _load_universe_source() -> pd.DataFrame:
    if UNIVERSE_SOURCE_PATH.exists():
        return pd.read_parquet(UNIVERSE_SOURCE_PATH)
    frame = get_universe_source()
    frame.to_parquet(UNIVERSE_SOURCE_PATH, index=False)
    return frame


def _path_snapshot(path) -> dict[str, object]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": int(stat.st_size),
        "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
    }


def _write_run_manifest(args: argparse.Namespace, phase: str, summaries: dict[str, object]) -> None:
    save_json(
        RUN_MANIFEST_PATH,
        {
            "generated_at": datetime.now().isoformat(),
            "phase": phase,
            "arguments": {
                "start": args.start,
                "end": args.end,
                "limit": args.limit,
                "universe_mode": args.universe_mode,
            },
            "summaries": summaries,
            "inputs": {
                "universe_source": _path_snapshot(UNIVERSE_SOURCE_PATH),
                "price_panel": _path_snapshot(PRICE_PANEL_PATH),
                "processed_universe_source": _path_snapshot(PROCESSED_UNIVERSE_SOURCE_PATH),
                "universe_membership": _path_snapshot(UNIVERSE_MEMBERSHIP_PATH),
                "feature_panel": _path_snapshot(FEATURE_PANEL_PATH),
                "score_panel": _path_snapshot(SCORE_PANEL_PATH),
                "portfolio_history": _path_snapshot(PORTFOLIO_HISTORY_PATH),
                "trade_log": _path_snapshot(TRADE_LOG_PATH),
                "backtest_summary": _path_snapshot(BACKTEST_SUMMARY_PATH),
                "pipeline_summary": _path_snapshot(PIPELINE_SUMMARY_PATH),
            },
        },
    )


def refresh_data(
    start: str,
    end: str,
    limit: int | None = None,
    universe_mode: str = UNIVERSE_MODE,
) -> dict[str, object]:
    ensure_output_dirs()
    if DATA_PROVIDER == "wrds":
        artifacts = refresh_data_from_wrds(start=start, end=end, limit=limit, universe_mode=universe_mode)
        artifacts.universe_source.to_parquet(UNIVERSE_SOURCE_PATH, index=False)
        artifacts.price_panel.to_parquet(PRICE_PANEL_PATH, index=False)
        return artifacts.summary

    universe_source = get_universe_source()
    from .config import SECTOR_ETF_MAP

    all_tickers = build_download_ticker_list(universe_source, BENCHMARK_TICKERS, list(SECTOR_ETF_MAP.values()), WATCHLIST)
    if limit is not None:
        preferred = WATCHLIST + BENCHMARK_TICKERS + list(SECTOR_ETF_MAP.values())
        limited: list[str] = []
        for ticker in preferred + all_tickers:
            if ticker not in limited:
                limited.append(ticker)
            if len(limited) >= limit:
                break
        all_tickers = limited

    price_panel = download_price_history(all_tickers, start=start, end=end)
    universe_source.to_parquet(UNIVERSE_SOURCE_PATH, index=False)
    price_panel.to_parquet(PRICE_PANEL_PATH, index=False)
    return {
        "universe_source_rows": int(len(universe_source)),
        "downloaded_tickers": int(price_panel["ticker"].nunique()),
        "price_rows": int(len(price_panel)),
        "price_panel_path": str(PRICE_PANEL_PATH),
    }


def build_features(limit: int | None = None) -> dict[str, object]:
    universe_source = _load_universe_source()
    universe_source.to_parquet(PROCESSED_UNIVERSE_SOURCE_PATH, index=False)
    prices = pd.read_parquet(PRICE_PANEL_PATH)
    target_count = min(limit or len(universe_source), len(universe_source))
    universe_membership = build_point_in_time_universe(universe_source, prices, target_count=target_count)
    latest_universe = build_latest_universe(universe_membership)
    features = compute_feature_panel(prices, universe_membership, BENCHMARK_TICKERS)
    universe_membership.to_parquet(UNIVERSE_MEMBERSHIP_PATH, index=False)
    features.to_parquet(FEATURE_PANEL_PATH, index=False)
    average_daily_size = (
        float(universe_membership.groupby("date", sort=False)["ticker"].nunique().mean()) if not universe_membership.empty else 0.0
    )
    return {
        "eligible_universe_size": int(len(latest_universe)),
        "average_daily_universe_size": average_daily_size,
        "feature_rows": int(len(features)),
        "feature_panel_path": str(FEATURE_PANEL_PATH),
    }


def score_universe() -> dict[str, object]:
    features = pd.read_parquet(FEATURE_PANEL_PATH)
    score_panel = compute_score_panel(features)
    score_panel.to_parquet(SCORE_PANEL_PATH, index=False)
    latest_date = pd.to_datetime(score_panel["date"]).max()
    latest = score_panel[score_panel["date"] == latest_date]
    return {
        "score_rows": int(len(score_panel)),
        "latest_signal_date": str(latest_date.date()),
        "latest_universe_size": int(len(latest)),
        "score_panel_path": str(SCORE_PANEL_PATH),
    }


def backtest_phase() -> dict[str, object]:
    score_panel = pd.read_parquet(SCORE_PANEL_PATH)
    artifacts = run_backtest(score_panel)
    experimental_artifacts = run_backtest(
        score_panel,
        score_column="experimental_score",
        rank_pct_column="experimental_rank_pct",
        rank_column="experimental_rank",
    )
    artifacts.history.to_parquet(PORTFOLIO_HISTORY_PATH, index=False)
    artifacts.trades.to_parquet(TRADE_LOG_PATH, index=False)
    experimental_artifacts.history.to_parquet(EXPERIMENTAL_PORTFOLIO_HISTORY_PATH, index=False)
    experimental_artifacts.trades.to_parquet(EXPERIMENTAL_TRADE_LOG_PATH, index=False)
    artifacts.latest_target_portfolio.to_csv(LATEST_PORTFOLIO_PATH, index=False)
    artifacts.latest_holdings.to_csv(CURRENT_HOLDINGS_PATH, index=False)
    experimental_artifacts.latest_target_portfolio.to_csv(LATEST_EXPERIMENTAL_PORTFOLIO_PATH, index=False)
    experimental_artifacts.latest_holdings.to_csv(CURRENT_EXPERIMENTAL_HOLDINGS_PATH, index=False)
    BACKTEST_SUMMARY_PATH.write_text(json.dumps(artifacts.summary, indent=2), encoding="utf-8")
    EXPERIMENTAL_BACKTEST_SUMMARY_PATH.write_text(json.dumps(experimental_artifacts.summary, indent=2), encoding="utf-8")
    return {
        "baseline": artifacts.summary,
        "experimental": experimental_artifacts.summary,
    }


def report_phase() -> dict[str, object]:
    score_panel = pd.read_parquet(SCORE_PANEL_PATH)
    history = pd.read_parquet(PORTFOLIO_HISTORY_PATH)
    experimental_history = pd.read_parquet(EXPERIMENTAL_PORTFOLIO_HISTORY_PATH)
    trades = pd.read_parquet(TRADE_LOG_PATH)
    experimental_trades = pd.read_parquet(EXPERIMENTAL_TRADE_LOG_PATH)
    latest_portfolio = pd.read_csv(LATEST_PORTFOLIO_PATH) if LATEST_PORTFOLIO_PATH.exists() else pd.DataFrame()
    current_holdings = pd.read_csv(CURRENT_HOLDINGS_PATH) if CURRENT_HOLDINGS_PATH.exists() else pd.DataFrame()
    latest_experimental_portfolio = (
        pd.read_csv(LATEST_EXPERIMENTAL_PORTFOLIO_PATH) if LATEST_EXPERIMENTAL_PORTFOLIO_PATH.exists() else pd.DataFrame()
    )
    current_experimental_holdings = (
        pd.read_csv(CURRENT_EXPERIMENTAL_HOLDINGS_PATH) if CURRENT_EXPERIMENTAL_HOLDINGS_PATH.exists() else pd.DataFrame()
    )
    backtest_summary = json.loads(BACKTEST_SUMMARY_PATH.read_text(encoding="utf-8"))
    experimental_backtest_summary = json.loads(EXPERIMENTAL_BACKTEST_SUMMARY_PATH.read_text(encoding="utf-8"))
    summary = write_reports(
        score_panel,
        history,
        experimental_history,
        latest_portfolio,
        latest_experimental_portfolio,
        current_holdings,
        current_experimental_holdings,
        trades,
        experimental_trades,
        backtest_summary,
        experimental_backtest_summary,
    )
    return summary


def main() -> None:
    args = parse_args()

    if args.phase == "refresh_data":
        summary = refresh_data(args.start, args.end, args.limit, args.universe_mode)
        _write_run_manifest(args, args.phase, {"refresh": summary})
        print(json.dumps(summary, indent=2))
        return
    if args.phase == "build_features":
        summary = build_features(args.limit)
        _write_run_manifest(args, args.phase, {"features": summary})
        print(json.dumps(summary, indent=2))
        return
    if args.phase == "score":
        summary = score_universe()
        _write_run_manifest(args, args.phase, {"scores": summary})
        print(json.dumps(summary, indent=2))
        return
    if args.phase == "backtest":
        summary = backtest_phase()
        _write_run_manifest(args, args.phase, {"backtest": summary})
        print(json.dumps(summary, indent=2))
        return
    if args.phase == "report":
        summary = report_phase()
        _write_run_manifest(args, args.phase, {"report": summary})
        print(json.dumps(summary, indent=2))
        return
    if args.phase in {"all", "smoke"}:
        limit = args.limit
        if args.phase == "smoke" and limit is None:
            limit = 80
        refresh_summary = refresh_data(args.start, args.end, limit, args.universe_mode)
        feature_summary = build_features(limit)
        score_summary = score_universe()
        backtest_summary = backtest_phase()
        report_summary = report_phase()
        combined = {
            "refresh": refresh_summary,
            "features": feature_summary,
            "scores": score_summary,
            "backtest": backtest_summary,
            "report": report_summary,
        }
        _write_run_manifest(args, args.phase, combined)
        print(json.dumps(combined, indent=2))
        return


if __name__ == "__main__":
    main()
