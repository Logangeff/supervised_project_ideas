from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import REGIME_CASH_CUT, REGIME_VOL_THRESHOLD, REBALANCE_WEEKDAY, TRANSACTION_COST_BPS
from .portfolio import select_target_portfolio


@dataclass
class BacktestArtifacts:
    history: pd.DataFrame
    trades: pd.DataFrame
    latest_portfolio: pd.DataFrame
    summary: dict[str, float | int | str]


def _build_price_maps(feature_panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    closes = feature_panel.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    opens = feature_panel.pivot(index="date", columns="ticker", values="adj_open").sort_index()
    return opens, closes


def _candidate_signal_dates(score_panel: pd.DataFrame) -> list[pd.Timestamp]:
    dates = sorted(pd.to_datetime(score_panel["date"].unique()))
    return [date for date in dates if date.weekday() == REBALANCE_WEEKDAY]


def _next_trading_date(all_dates: list[pd.Timestamp], current: pd.Timestamp) -> pd.Timestamp | None:
    idx = all_dates.index(current)
    if idx + 1 >= len(all_dates):
        return None
    return all_dates[idx + 1]


def _annualized_return(equity_curve: pd.Series) -> float:
    if equity_curve.empty or len(equity_curve) < 2:
        return 0.0
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0
    years = len(equity_curve) / 252.0
    if years <= 0:
        return 0.0
    return float((1 + total_return) ** (1 / years) - 1)


def _max_drawdown(equity_curve: pd.Series) -> float:
    rolling_peak = equity_curve.cummax()
    drawdown = equity_curve / rolling_peak - 1.0
    return float(drawdown.min())


def _summarize(history: pd.DataFrame, trades: pd.DataFrame) -> dict[str, float | int | str]:
    returns = history["portfolio_return"].fillna(0.0)
    equity = history["equity_curve"].ffill()
    ann_return = _annualized_return(equity)
    ann_vol = float(returns.std(ddof=0) * np.sqrt(252))
    downside_vol = float(returns[returns < 0].std(ddof=0) * np.sqrt(252)) if (returns < 0).any() else 0.0
    sharpe = ann_return / ann_vol if ann_vol else 0.0
    sortino = ann_return / downside_vol if downside_vol else 0.0
    max_dd = _max_drawdown(equity)
    calmar = ann_return / abs(max_dd) if max_dd else 0.0
    return {
        "start_date": str(history["date"].min().date()),
        "end_date": str(history["date"].max().date()),
        "days": int(len(history)),
        "cagr": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "total_turnover": float(history["turnover"].fillna(0.0).sum()),
        "average_daily_turnover": float(history["turnover"].fillna(0.0).mean()),
        "trade_count": int(len(trades)),
        "average_position_count": float(history["active_names"].mean()),
        "average_gross_exposure": float(history["gross_exposure"].mean()),
    }


def run_backtest(
    score_panel: pd.DataFrame,
    score_column: str = "core_score",
    rank_pct_column: str = "core_rank_pct",
    rank_column: str = "core_rank",
) -> BacktestArtifacts:
    panel = score_panel.sort_values(["date", "ticker"]).copy()
    panel["date"] = pd.to_datetime(panel["date"])
    all_dates = sorted(pd.to_datetime(panel["date"].unique()))
    opens, closes = _build_price_maps(panel)
    close_returns = closes.pct_change(fill_method=None).fillna(0.0)
    rank_pct_map = panel.pivot(index="date", columns="ticker", values=rank_pct_column).sort_index()
    below_sma_map = panel.pivot(index="date", columns="ticker", values="below_sma100_2d").sort_index()
    spy_frame = (
        panel[panel["ticker"] == "SPY"][["date", "adj_close", "sma_200", "mom_6_1", "realized_vol_20"]]
        .drop_duplicates("date")
        .set_index("date")
        .sort_index()
    )
    panel_by_date = {date: frame.copy() for date, frame in panel.groupby("date", sort=True)}

    scheduled_targets: dict[pd.Timestamp, pd.DataFrame] = {}
    for signal_date in _candidate_signal_dates(panel):
        trade_date = _next_trading_date(all_dates, signal_date)
        if trade_date is None:
            continue
        target = select_target_portfolio(
            panel,
            signal_date,
            score_column=score_column,
            rank_pct_column=rank_pct_column,
        )
        if target.empty:
            continue
        target = target[["ticker", "sector", "target_weight", score_column, rank_column, rank_pct_column, "stop_distance"]].copy()
        target["signal_date"] = signal_date
        target["trade_date"] = trade_date
        scheduled_targets[trade_date] = target

    current_weights: dict[str, float] = {}
    current_stop_levels: dict[str, float] = {}
    trade_rows: list[dict[str, object]] = []
    history_rows: list[dict[str, object]] = []
    equity = 1.0

    for date in all_dates:
        turnover = 0.0
        close_row = closes.loc[date] if date in closes.index else pd.Series(dtype=float)
        open_row = opens.loc[date] if date in opens.index else pd.Series(dtype=float)
        return_row = close_returns.loc[date] if date in close_returns.index else pd.Series(dtype=float)
        rank_pct_row = rank_pct_map.loc[date] if date in rank_pct_map.index else pd.Series(dtype=float)
        below_row = below_sma_map.loc[date] if date in below_sma_map.index else pd.Series(dtype=float)

        if date in scheduled_targets:
            target = scheduled_targets[date]
            regime_cut = 1.0
            if date in spy_frame.index:
                spy = spy_frame.loc[date]
                risk_off = (spy["adj_close"] < spy["sma_200"]) and (spy["mom_6_1"] < 0)
                elevated_vol = float(spy["realized_vol_20"]) > REGIME_VOL_THRESHOLD if pd.notna(spy["realized_vol_20"]) else False
                if risk_off:
                    regime_cut = REGIME_CASH_CUT
                if risk_off and elevated_vol:
                    regime_cut = min(regime_cut, 0.35)
            target_weights = {row["ticker"]: float(row["target_weight"]) * regime_cut for _, row in target.iterrows()}
            keys = sorted(set(current_weights) | set(target_weights))
            turnover = sum(abs(target_weights.get(key, 0.0) - current_weights.get(key, 0.0)) for key in keys)
            stop_lookup = target.set_index("ticker")["stop_distance"].to_dict()
            for ticker, target_weight in target_weights.items():
                open_price = open_row.get(ticker, np.nan)
                close_price = close_row.get(ticker, np.nan)
                entry_price = float(open_price if pd.notna(open_price) else close_price)
                stop_distance = float(stop_lookup[ticker])
                current_stop_levels[ticker] = entry_price * (1.0 - stop_distance)
                trade_rows.append({"date": date, "ticker": ticker, "action": "rebalance", "weight": target_weight})
            for ticker in set(current_weights) - set(target_weights):
                current_stop_levels.pop(ticker, None)
            current_weights = target_weights

        active_names = sum(1 for weight in current_weights.values() if weight > 0)
        daily_return = sum(float(weight) * float(return_row.get(ticker, 0.0)) for ticker, weight in current_weights.items())

        if turnover:
            daily_return -= turnover * (TRANSACTION_COST_BPS / 10_000.0)
        equity *= 1.0 + daily_return

        exit_tickers: list[str] = []
        for ticker in list(current_weights):
            rank_pct = rank_pct_row.get(ticker, np.nan)
            below_flag = below_row.get(ticker, 0)
            close_price = close_row.get(ticker, np.nan)
            if pd.notna(rank_pct) and rank_pct < 0.65:
                exit_tickers.append(ticker)
                continue
            if pd.notna(below_flag) and int(below_flag) == 1:
                exit_tickers.append(ticker)
                continue
            if ticker in current_stop_levels and pd.notna(close_price) and float(close_price) <= current_stop_levels[ticker]:
                exit_tickers.append(ticker)

        if exit_tickers:
            for ticker in exit_tickers:
                trade_rows.append({"date": date, "ticker": ticker, "action": "event_exit", "weight": current_weights[ticker]})
                current_weights.pop(ticker, None)
                current_stop_levels.pop(ticker, None)

        history_rows.append(
            {
                "date": date,
                "portfolio_return": daily_return,
                "equity_curve": equity,
                "turnover": turnover,
                "gross_exposure": float(sum(current_weights.values())),
                "active_names": active_names,
            }
        )

    history = pd.DataFrame(history_rows)
    history["drawdown"] = history["equity_curve"] / history["equity_curve"].cummax() - 1.0
    trades = pd.DataFrame(trade_rows)
    latest_date = history["date"].max()
    latest_cross_section = panel_by_date.get(latest_date, pd.DataFrame())
    latest_portfolio = (
        latest_cross_section[latest_cross_section["ticker"].isin(current_weights)]
        .assign(weight=lambda frame: frame["ticker"].map(current_weights))
        [["date", "ticker", "weight", rank_column, rank_pct_column, score_column]]
        .sort_values("weight", ascending=False)
        .reset_index(drop=True)
    )
    summary = _summarize(history, trades)
    return BacktestArtifacts(history=history, trades=trades, latest_portfolio=latest_portfolio, summary=summary)
