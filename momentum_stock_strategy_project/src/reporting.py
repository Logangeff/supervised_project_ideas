from __future__ import annotations

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import (
    BACKTEST_SUMMARY_PATH,
    EQUITY_CURVE_FIGURE,
    EXPERIMENTAL_BACKTEST_SUMMARY_PATH,
    EXPERIMENTAL_SCREENER_SUMMARY_PATH,
    LATEST_EXPERIMENTAL_PORTFOLIO_PATH,
    LATEST_EXPERIMENTAL_RANK_PATH,
    LATEST_CORE_RANK_PATH,
    LATEST_EARLY_RANK_PATH,
    LATEST_PORTFOLIO_PATH,
    LATEST_WATCHLIST_PATH,
    ORDER_RECOMMENDATIONS_PATH,
    PIPELINE_SUMMARY_PATH,
    RANK_SCATTER_FIGURE,
    SCREENER_SUMMARY_PATH,
    SIGNAL_COMPARISON_PATH,
    WATCHLIST,
    WAREHOUSE_PATH,
    save_json,
)


def _latest_rows(score_panel: pd.DataFrame, rank_column: str = "core_rank") -> pd.DataFrame:
    latest_date = pd.to_datetime(score_panel["date"]).max()
    latest = score_panel[score_panel["date"] == latest_date].copy()
    return latest.sort_values(rank_column).reset_index(drop=True)


def compute_screener_metrics(score_panel: pd.DataFrame, score_column: str = "core_score") -> dict[str, float]:
    rows: list[dict[str, float]] = []
    for _, frame in score_panel.groupby("date", sort=True):
        valid = frame.dropna(subset=[score_column]).copy()
        valid_21 = valid.dropna(subset=["forward_return_21"])
        valid_63 = valid.dropna(subset=["forward_return_63"])
        min_required = max(10, min(30, len(valid) // 3 if len(valid) else 0))
        if len(valid_21) < min_required:
            continue
        rank_ic_21 = (
            valid_21[score_column].corr(valid_21["forward_return_21"], method="spearman")
            if valid_21[score_column].nunique() > 1 and valid_21["forward_return_21"].nunique() > 1
            else np.nan
        )
        rank_ic_63 = (
            valid_63[score_column].corr(valid_63["forward_return_63"], method="spearman")
            if len(valid_63) >= min_required and valid_63[score_column].nunique() > 1 and valid_63["forward_return_63"].nunique() > 1
            else np.nan
        )
        bucket_size = max(5, min(20, int(round(len(valid_21) * 0.1))))
        top_bucket = valid_21.nlargest(bucket_size, score_column)
        if valid_21.empty:
            median_bucket = valid_21.iloc[:0]
        else:
            ordered = valid_21.sort_values(score_column, ascending=False).reset_index(drop=True)
            midpoint = len(ordered) // 2
            start = max(0, midpoint - bucket_size // 2)
            stop = min(len(ordered), start + bucket_size)
            median_bucket = ordered.iloc[start:stop]
        top20_63 = valid_63.nlargest(min(bucket_size, len(valid_63)), score_column) if not valid_63.empty else valid_63
        rows.append(
            {
                "rank_ic_21": float(rank_ic_21) if pd.notna(rank_ic_21) else 0.0,
                "rank_ic_63": float(rank_ic_63) if pd.notna(rank_ic_63) else 0.0,
                "top_bucket_size": int(bucket_size),
                "top20_hit_rate_21": float((top_bucket["forward_return_21"] > 0).mean()),
                "top20_hit_rate_63": float((top20_63["forward_return_63"] > 0).mean()) if not top20_63.empty else 0.0,
                "top20_forward_return_21": float(top_bucket["forward_return_21"].mean()),
                "top20_forward_return_63": float(top20_63["forward_return_63"].mean()) if not top20_63.empty else 0.0,
                "top_decile_vs_median_21": float(top_bucket["forward_return_21"].mean() - median_bucket["forward_return_21"].mean()),
                "universe_size": int(len(valid_21)),
            }
        )

    metrics = pd.DataFrame(rows)
    if metrics.empty:
        return {}
    return {
        "evaluation_dates": int(len(metrics)),
        "average_universe_size": float(metrics["universe_size"].mean()),
        "average_top_bucket_size": float(metrics["top_bucket_size"].mean()),
        "average_rank_ic_21": float(metrics["rank_ic_21"].mean()),
        "average_rank_ic_63": float(metrics["rank_ic_63"].mean()),
        "average_top20_hit_rate_21": float(metrics["top20_hit_rate_21"].mean()),
        "average_top20_hit_rate_63": float(metrics["top20_hit_rate_63"].mean()),
        "average_top20_forward_return_21": float(metrics["top20_forward_return_21"].mean()),
        "average_top20_forward_return_63": float(metrics["top20_forward_return_63"].mean()),
        "average_top_decile_vs_median_21": float(metrics["top_decile_vs_median_21"].mean()),
    }


def _plot_equity_curve(history: pd.DataFrame) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(history["date"], history["equity_curve"], color="black", linewidth=1.8)
    axes[0].set_title("Momentum strategy equity curve")
    axes[0].set_ylabel("Equity")
    axes[1].fill_between(history["date"], history["drawdown"], 0.0, color="#C62828", alpha=0.65)
    axes[1].set_title("Drawdown")
    axes[1].set_ylabel("Drawdown")
    figure.tight_layout()
    figure.savefig(EQUITY_CURVE_FIGURE, dpi=160)
    plt.close(figure)


def _plot_latest_rank_scatter(latest: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(9, 6))
    scatter = axis.scatter(
        latest["core_score"],
        latest["forward_return_21"],
        c=latest["core_rank"],
        cmap="viridis_r",
        alpha=0.7,
    )
    axis.set_title("Latest cross-section: score vs 21-day forward return")
    axis.set_xlabel("Core score")
    axis.set_ylabel("Forward 21-day return")
    figure.colorbar(scatter, ax=axis, label="Core rank")
    figure.tight_layout()
    figure.savefig(RANK_SCATTER_FIGURE, dpi=160)
    plt.close(figure)


def _watchlist_actions(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    scored = frame.copy()
    scored["short_term_action"] = "Neutral"
    scored["position_action"] = "Neutral"
    scored["action_note"] = "Mixed setup."

    short_hard_buy = (
        (scored["trend_filter"] == 1)
        & (scored["early_rank_pct"] >= 0.90)
        & (scored["mom_3_1w"] > 0)
        & (scored["sector_rel_strength"] > 0)
    )
    short_buy = (
        (scored["trend_filter"] == 1)
        & (scored["early_rank_pct"] >= 0.75)
        & (scored["mom_3_1w"] > -0.02)
    )
    short_sell = (scored["early_rank_pct"] < 0.35) | (scored["mom_3_1w"] < -0.05)
    short_hard_sell = (scored["trend_filter"] == 0) & ((scored["mom_3_1w"] < -0.08) | (scored["below_sma100_2d"] == 1))

    core_hard_buy = (
        (scored["trend_filter"] == 1)
        & (scored["core_rank_pct"] >= 0.90)
        & (scored["prox_52w_high"] >= 0.97)
        & (scored["core_score"] > 0)
    )
    core_buy = (
        (scored["trend_filter"] == 1)
        & (scored["core_rank_pct"] >= 0.70)
        & (scored["core_score"] > 0)
    )
    core_sell = (
        (scored["core_rank_pct"] < 0.40)
        | (scored["core_score"] < 0)
        | (scored["trend_filter"] == 0)
    )
    core_hard_sell = (
        ((scored["below_sma100_2d"] == 1) & (scored["trend_filter"] == 0))
        | (scored["core_rank_pct"] < 0.20)
    )

    scored.loc[short_buy, "short_term_action"] = "Buy"
    scored.loc[short_hard_buy, "short_term_action"] = "Hard Buy"
    scored.loc[short_sell, "short_term_action"] = "Sell"
    scored.loc[short_hard_sell, "short_term_action"] = "Hard Sell"

    scored.loc[core_buy, "position_action"] = "Buy"
    scored.loc[core_hard_buy, "position_action"] = "Hard Buy"
    scored.loc[core_sell, "position_action"] = "Sell"
    scored.loc[core_hard_sell, "position_action"] = "Hard Sell"

    scored.loc[scored["position_action"] == "Hard Buy", "action_note"] = "Strong trend and top-tier core rank."
    scored.loc[scored["position_action"] == "Buy", "action_note"] = "Trend is intact and the stock ranks well."
    scored.loc[scored["position_action"] == "Sell", "action_note"] = "Momentum quality is fading."
    scored.loc[scored["position_action"] == "Hard Sell", "action_note"] = "Trend break or very weak rank."
    return scored


def _write_duckdb_tables(
    score_panel: pd.DataFrame,
    latest_core: pd.DataFrame,
    latest_experimental: pd.DataFrame,
    latest_early: pd.DataFrame,
    history: pd.DataFrame,
    history_experimental: pd.DataFrame,
    latest_portfolio: pd.DataFrame,
    latest_experimental_portfolio: pd.DataFrame,
) -> None:
    connection = duckdb.connect(str(WAREHOUSE_PATH))
    connection.register("score_panel_df", score_panel)
    connection.register("latest_core_df", latest_core)
    connection.register("latest_experimental_df", latest_experimental)
    connection.register("latest_early_df", latest_early)
    connection.register("history_df", history)
    connection.register("history_experimental_df", history_experimental)
    connection.register("latest_portfolio_df", latest_portfolio)
    connection.register("latest_experimental_portfolio_df", latest_experimental_portfolio)
    connection.execute("CREATE OR REPLACE TABLE score_panel AS SELECT * FROM score_panel_df")
    connection.execute("CREATE OR REPLACE TABLE latest_core_rank AS SELECT * FROM latest_core_df")
    connection.execute("CREATE OR REPLACE TABLE latest_experimental_rank AS SELECT * FROM latest_experimental_df")
    connection.execute("CREATE OR REPLACE TABLE latest_early_rank AS SELECT * FROM latest_early_df")
    connection.execute("CREATE OR REPLACE TABLE portfolio_history AS SELECT * FROM history_df")
    connection.execute("CREATE OR REPLACE TABLE portfolio_history_experimental AS SELECT * FROM history_experimental_df")
    connection.execute("CREATE OR REPLACE TABLE latest_target_portfolio AS SELECT * FROM latest_portfolio_df")
    connection.execute(
        "CREATE OR REPLACE TABLE latest_experimental_target_portfolio AS SELECT * FROM latest_experimental_portfolio_df"
    )
    connection.close()


def _signal_comparison_summary(
    baseline_backtest: dict[str, object],
    experimental_backtest: dict[str, object],
    baseline_screener: dict[str, object],
    experimental_screener: dict[str, object],
) -> dict[str, object]:
    def _delta(metric: str) -> float:
        return float(experimental_backtest.get(metric, 0.0)) - float(baseline_backtest.get(metric, 0.0))

    def _delta_screener(metric: str) -> float:
        return float(experimental_screener.get(metric, 0.0)) - float(baseline_screener.get(metric, 0.0))

    return {
        "baseline_signals": ["mom_12_1", "mom_6_1", "mom_3_1w", "risk_adj_mom", "prox_52w_high", "sector_rel_strength"],
        "experimental_added_signals": ["macd_hist", "adx_14", "rsi_14"],
        "backtest_delta": {
            "cagr": _delta("cagr"),
            "sharpe": _delta("sharpe"),
            "max_drawdown": _delta("max_drawdown"),
            "calmar": _delta("calmar"),
        },
        "screener_delta": {
            "average_rank_ic_21": _delta_screener("average_rank_ic_21"),
            "average_rank_ic_63": _delta_screener("average_rank_ic_63"),
            "average_top20_hit_rate_21": _delta_screener("average_top20_hit_rate_21"),
            "average_top_decile_vs_median_21": _delta_screener("average_top_decile_vs_median_21"),
        },
    }


def write_reports(
    score_panel: pd.DataFrame,
    backtest_history: pd.DataFrame,
    experimental_history: pd.DataFrame,
    latest_portfolio: pd.DataFrame,
    latest_experimental_portfolio: pd.DataFrame,
    trade_log: pd.DataFrame,
    experimental_trade_log: pd.DataFrame,
    backtest_summary: dict[str, object],
    experimental_backtest_summary: dict[str, object],
) -> dict[str, object]:
    latest = _latest_rows(score_panel)
    latest_core = latest.sort_values("core_rank").reset_index(drop=True)
    latest_experimental = _latest_rows(score_panel, rank_column="experimental_rank")
    latest_early = latest.sort_values("early_rank").reset_index(drop=True)
    latest_watchlist = latest[latest["ticker"].isin(WATCHLIST)].sort_values("core_rank").reset_index(drop=True)
    latest_watchlist = _watchlist_actions(latest_watchlist)
    screener_metrics = compute_screener_metrics(score_panel, score_column="core_score")
    experimental_screener_metrics = compute_screener_metrics(score_panel, score_column="experimental_score")
    orders = latest_portfolio[["ticker", "weight", "core_rank", "core_score"]].copy() if not latest_portfolio.empty else pd.DataFrame()
    if not orders.empty:
        orders["action"] = "hold_or_buy"
        orders["comment"] = "Latest target portfolio weight from weekly rebalance engine."
        orders = orders[["action", "ticker", "weight", "core_rank", "core_score", "comment"]]

    latest_core.to_csv(LATEST_CORE_RANK_PATH, index=False)
    latest_experimental.to_csv(LATEST_EXPERIMENTAL_RANK_PATH, index=False)
    latest_early.to_csv(LATEST_EARLY_RANK_PATH, index=False)
    latest_watchlist.to_csv(LATEST_WATCHLIST_PATH, index=False)
    latest_portfolio.to_csv(LATEST_PORTFOLIO_PATH, index=False)
    latest_experimental_portfolio.to_csv(LATEST_EXPERIMENTAL_PORTFOLIO_PATH, index=False)
    orders.to_csv(ORDER_RECOMMENDATIONS_PATH, index=False)

    save_json(BACKTEST_SUMMARY_PATH, backtest_summary)
    save_json(EXPERIMENTAL_BACKTEST_SUMMARY_PATH, experimental_backtest_summary)
    save_json(SCREENER_SUMMARY_PATH, screener_metrics)
    save_json(EXPERIMENTAL_SCREENER_SUMMARY_PATH, experimental_screener_metrics)
    save_json(
        SIGNAL_COMPARISON_PATH,
        _signal_comparison_summary(
            backtest_summary,
            experimental_backtest_summary,
            screener_metrics,
            experimental_screener_metrics,
        ),
    )
    save_json(
        PIPELINE_SUMMARY_PATH,
        {
            "latest_signal_date": str(pd.to_datetime(latest["date"].max()).date()),
            "latest_universe_size": int(latest["ticker"].nunique()),
            "latest_target_positions": int(len(latest_portfolio)),
            "watchlist_hits": int(len(latest_watchlist)),
            "trade_log_rows": int(len(trade_log)),
            "experimental_trade_log_rows": int(len(experimental_trade_log)),
        },
    )

    _plot_equity_curve(backtest_history)
    _plot_latest_rank_scatter(latest)
    _write_duckdb_tables(
        score_panel,
        latest_core,
        latest_experimental,
        latest_early,
        backtest_history,
        experimental_history,
        latest_portfolio,
        latest_experimental_portfolio,
    )
    return {
        "latest_core_rank_path": str(LATEST_CORE_RANK_PATH),
        "latest_experimental_rank_path": str(LATEST_EXPERIMENTAL_RANK_PATH),
        "latest_early_rank_path": str(LATEST_EARLY_RANK_PATH),
        "latest_watchlist_path": str(LATEST_WATCHLIST_PATH),
        "latest_portfolio_path": str(LATEST_PORTFOLIO_PATH),
        "latest_experimental_portfolio_path": str(LATEST_EXPERIMENTAL_PORTFOLIO_PATH),
        "backtest_summary_path": str(BACKTEST_SUMMARY_PATH),
        "experimental_backtest_summary_path": str(EXPERIMENTAL_BACKTEST_SUMMARY_PATH),
        "screener_summary_path": str(SCREENER_SUMMARY_PATH),
        "experimental_screener_summary_path": str(EXPERIMENTAL_SCREENER_SUMMARY_PATH),
    }
