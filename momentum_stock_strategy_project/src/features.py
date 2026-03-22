from __future__ import annotations

import numpy as np
import pandas as pd


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = losses.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    up_move = high.diff().fillna(0.0)
    down_move = (-low.diff()).fillna(0.0)
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=high.index,
        dtype=float,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=high.index,
        dtype=float,
    )
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr.replace(0.0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr.replace(0.0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean().fillna(0.0)


def _compute_ticker_features(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values("date").copy()
    ordered["ret_1d"] = ordered["adj_close"].pct_change(fill_method=None)
    ordered["mom_12_1"] = ordered["adj_close"].shift(21) / ordered["adj_close"].shift(252) - 1.0
    ordered["mom_6_1"] = ordered["adj_close"].shift(21) / ordered["adj_close"].shift(126) - 1.0
    ordered["mom_3_1w"] = ordered["adj_close"].shift(5) / ordered["adj_close"].shift(63) - 1.0
    ordered["mom_63"] = ordered["adj_close"] / ordered["adj_close"].shift(63) - 1.0
    ordered["realized_vol_20"] = ordered["ret_1d"].rolling(20).std() * np.sqrt(252)
    ordered["realized_vol_63"] = ordered["ret_1d"].rolling(63).std() * np.sqrt(252)
    ordered["risk_adj_mom"] = ordered["mom_6_1"] / ordered["realized_vol_63"].replace(0.0, np.nan)
    ordered["rolling_high_252"] = ordered["adj_close"].rolling(252).max()
    ordered["rolling_high_60"] = ordered["adj_close"].rolling(60).max()
    ordered["prox_52w_high"] = ordered["adj_close"] / ordered["rolling_high_252"]
    ordered["breakout_60"] = (ordered["adj_close"] >= ordered["rolling_high_60"].shift(1)).fillna(False)
    ordered["ema_12"] = _ema(ordered["adj_close"], 12)
    ordered["ema_26"] = _ema(ordered["adj_close"], 26)
    ordered["macd_line"] = ordered["ema_12"] - ordered["ema_26"]
    ordered["macd_signal"] = _ema(ordered["macd_line"], 9)
    ordered["macd_hist"] = ordered["macd_line"] - ordered["macd_signal"]
    ordered["rsi_14"] = _rsi(ordered["adj_close"], 14)
    ordered["adx_14"] = _adx(ordered["adj_high"], ordered["adj_low"], ordered["adj_close"], 14)
    ordered["sma_100"] = ordered["adj_close"].rolling(100).mean()
    ordered["sma_200"] = ordered["adj_close"].rolling(200).mean()
    ordered["trend_filter"] = (
        ((ordered["adj_close"] > ordered["sma_100"]) & (ordered["adj_close"] > ordered["sma_200"]))
        .fillna(False)
        .astype(int)
    )
    ordered["avg_dollar_volume_20"] = (ordered["adj_close"] * ordered["volume"]).rolling(20).mean()
    ordered["avg_dollar_volume_60"] = (ordered["adj_close"] * ordered["volume"]).rolling(60).mean()
    ordered["volume_confirmation"] = np.log(ordered["avg_dollar_volume_20"] / ordered["avg_dollar_volume_60"]).clip(-1.5, 1.5)

    prev_close = ordered["adj_close"].shift(1)
    true_range = pd.concat(
        [
            ordered["adj_high"] - ordered["adj_low"],
            (ordered["adj_high"] - prev_close).abs(),
            (ordered["adj_low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    ordered["atr_20"] = true_range.rolling(20).mean() / ordered["adj_close"]
    ordered["below_sma100_2d"] = (
        ((ordered["adj_close"] < ordered["sma_100"]).rolling(2).sum() >= 2)
        .fillna(False)
        .astype(int)
    )
    ordered["forward_return_21"] = ordered["adj_close"].shift(-21) / ordered["adj_close"] - 1.0
    ordered["forward_return_63"] = ordered["adj_close"].shift(-63) / ordered["adj_close"] - 1.0
    return ordered


def compute_feature_panel(prices: pd.DataFrame, universe: pd.DataFrame, benchmark_tickers: list[str]) -> pd.DataFrame:
    relevant_tickers = set(universe["ticker"]).union(benchmark_tickers).union(universe["sector_etf"])
    panel = prices[prices["ticker"].isin(relevant_tickers)].copy()
    feature_panel = (
        panel.groupby("ticker", group_keys=False, sort=True)
        .apply(_compute_ticker_features)
        .reset_index(drop=True)
    )

    sector_returns = feature_panel[feature_panel["ticker"].isin(set(universe["sector_etf"]))][["date", "ticker", "mom_63"]].copy()
    sector_returns = sector_returns.rename(columns={"ticker": "sector_etf", "mom_63": "sector_etf_mom_63"})

    spy_returns = feature_panel[feature_panel["ticker"] == "SPY"][["date", "mom_63", "adj_close", "sma_200", "realized_vol_20"]].copy()
    spy_returns = spy_returns.rename(
        columns={
            "mom_63": "spy_mom_63",
            "adj_close": "spy_close",
            "sma_200": "spy_sma_200",
            "realized_vol_20": "spy_realized_vol_20",
        }
    )

    stock_features = feature_panel[feature_panel["ticker"].isin(set(universe["ticker"]))].copy()
    stock_features = stock_features.merge(universe[["ticker", "sector", "sector_etf", "universe_rank"]], on="ticker", how="left")
    stock_features = stock_features.merge(sector_returns, on=["date", "sector_etf"], how="left")
    stock_features = stock_features.merge(spy_returns, on="date", how="left")
    stock_features["sector_rel_strength"] = stock_features["mom_63"] - stock_features["sector_etf_mom_63"]
    stock_features["risk_off_regime"] = (
        ((stock_features["spy_close"] < stock_features["spy_sma_200"]) & (stock_features["spy_mom_63"] < 0))
        .fillna(False)
        .astype(int)
    )
    return stock_features.sort_values(["date", "ticker"]).reset_index(drop=True)
