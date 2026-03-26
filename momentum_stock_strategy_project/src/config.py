from __future__ import annotations

import json
import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = ROOT_DIR / "outputs"
METRICS_DIR = OUTPUT_DIR / "metrics"
REPORTS_DIR = OUTPUT_DIR / "reports"
FIGURES_DIR = OUTPUT_DIR / "figures"
WAREHOUSE_PATH = DATA_DIR / "warehouse.duckdb"
LOCAL_CONFIG_DIR = ROOT_DIR / "config"
WRDS_CREDENTIALS_JSON = LOCAL_CONFIG_DIR / "wrds_credentials.json"
FALLBACK_WRDS_CREDENTIALS_JSON = ROOT_DIR.parent / "wrds_optionmetrics_exploration" / "config" / "wrds_credentials.json"
USER_WATCHLIST_PATH = LOCAL_CONFIG_DIR / "watchlist.csv"
USER_WATCHLIST_TEMPLATE_PATH = LOCAL_CONFIG_DIR / "watchlist.template.csv"

UNIVERSE_SOURCE_PATH = RAW_DIR / "universe_source.parquet"
PRICE_PANEL_PATH = RAW_DIR / "price_panel.parquet"
WRDS_UNIVERSE_SOURCE_CSV = RAW_DIR / "wrds_universe_source.csv"
WRDS_UNIVERSE_SOURCE_PARQUET = RAW_DIR / "wrds_universe_source.parquet"
WRDS_STOCK_CACHE_CSV = RAW_DIR / "wrds_stock_price_cache.csv"
WRDS_STOCK_CACHE_PARQUET = RAW_DIR / "wrds_stock_price_cache.parquet"
RECENT_MARKET_CACHE_CSV = RAW_DIR / "recent_market_extension_cache.csv"
RECENT_MARKET_CACHE_PARQUET = RAW_DIR / "recent_market_extension_cache.parquet"
UNIVERSE_MEMBERSHIP_PATH = PROCESSED_DIR / "universe_membership.parquet"
PROCESSED_UNIVERSE_SOURCE_PATH = PROCESSED_DIR / "universe_source_snapshot.parquet"
FEATURE_PANEL_PATH = PROCESSED_DIR / "feature_panel.parquet"
SCORE_PANEL_PATH = PROCESSED_DIR / "score_panel.parquet"
PORTFOLIO_HISTORY_PATH = PROCESSED_DIR / "portfolio_history.parquet"
TRADE_LOG_PATH = PROCESSED_DIR / "trade_log.parquet"
EXPERIMENTAL_PORTFOLIO_HISTORY_PATH = PROCESSED_DIR / "portfolio_history_experimental.parquet"
EXPERIMENTAL_TRADE_LOG_PATH = PROCESSED_DIR / "trade_log_experimental.parquet"
LATEST_PORTFOLIO_PATH = REPORTS_DIR / "latest_target_portfolio.csv"
LATEST_EXPERIMENTAL_PORTFOLIO_PATH = REPORTS_DIR / "latest_experimental_target_portfolio.csv"
CURRENT_HOLDINGS_PATH = REPORTS_DIR / "current_holdings.csv"
CURRENT_EXPERIMENTAL_HOLDINGS_PATH = REPORTS_DIR / "current_experimental_holdings.csv"
LATEST_CORE_RANK_PATH = REPORTS_DIR / "latest_core_rank.csv"
LATEST_EXPERIMENTAL_RANK_PATH = REPORTS_DIR / "latest_experimental_rank.csv"
LATEST_EARLY_RANK_PATH = REPORTS_DIR / "latest_early_rank.csv"
LATEST_WATCHLIST_PATH = REPORTS_DIR / "latest_watchlist_overlay.csv"
ORDER_RECOMMENDATIONS_PATH = REPORTS_DIR / "latest_order_recommendations.csv"
BACKTEST_SUMMARY_PATH = METRICS_DIR / "backtest_summary.json"
EXPERIMENTAL_BACKTEST_SUMMARY_PATH = METRICS_DIR / "experimental_backtest_summary.json"
SCREENER_SUMMARY_PATH = METRICS_DIR / "screener_summary.json"
EXPERIMENTAL_SCREENER_SUMMARY_PATH = METRICS_DIR / "experimental_screener_summary.json"
SIGNAL_COMPARISON_PATH = METRICS_DIR / "signal_variant_comparison.json"
PIPELINE_SUMMARY_PATH = METRICS_DIR / "pipeline_summary.json"
RUN_MANIFEST_PATH = METRICS_DIR / "run_manifest.json"
EQUITY_CURVE_FIGURE = FIGURES_DIR / "equity_curve.png"
RANK_SCATTER_FIGURE = FIGURES_DIR / "latest_rank_scatter.png"

SEED = 42
START_DATE = "2018-01-01"
DEFAULT_DATA_PROVIDER = "wrds" if WRDS_CREDENTIALS_JSON.exists() or FALLBACK_WRDS_CREDENTIALS_JSON.exists() else "free"
DATA_PROVIDER = os.getenv("MOMENTUM_DATA_PROVIDER", DEFAULT_DATA_PROVIDER).lower()
DEFAULT_UNIVERSE_MODE = "watchlist" if USER_WATCHLIST_PATH.exists() else "top_liquid"
UNIVERSE_MODE = os.getenv("MOMENTUM_UNIVERSE_MODE", DEFAULT_UNIVERSE_MODE).lower()

S_AND_P_500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
S_AND_P_400_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
UNIVERSE_TARGET_COUNT = 1000
MIN_PRICE = 10.0
MIN_MEDIAN_DOLLAR_VOLUME = 20_000_000.0
MIN_HISTORY_DAYS = 260

BENCHMARK_TICKERS = ["SPY", "QQQ", "MTUM"]
WATCHLIST = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
    "TSLA",
    "GOOGL",
    "NFLX",
    "AVGO",
    "LLY",
]

SECTOR_ETF_MAP = {
    "Communication Services": "XLC",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Industrials": "XLI",
    "Information Technology": "XLK",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
}

TICKER_SECTOR_OVERRIDES = {
    "AAPL": "Information Technology",
    "ABNB": "Consumer Discretionary",
    "AFRM": "Financials",
    "ALAB": "Information Technology",
    "AMAT": "Information Technology",
    "AMZN": "Consumer Discretionary",
    "APP": "Information Technology",
    "ASTS": "Communication Services",
    "AVGO": "Information Technology",
    "CEG": "Utilities",
    "COIN": "Financials",
    "CRWD": "Information Technology",
    "DDOG": "Information Technology",
    "DKNG": "Consumer Discretionary",
    "FANG": "Energy",
    "FTNT": "Information Technology",
    "GOOG": "Communication Services",
    "GOOGL": "Communication Services",
    "GLW": "Information Technology",
    "HOOD": "Financials",
    "KHC": "Consumer Staples",
    "KLAC": "Information Technology",
    "LRCX": "Information Technology",
    "META": "Communication Services",
    "MDB": "Information Technology",
    "MRNA": "Health Care",
    "MSFT": "Information Technology",
    "MU": "Information Technology",
    "NFLX": "Communication Services",
    "NVDA": "Information Technology",
    "PYPL": "Financials",
    "RIVN": "Consumer Discretionary",
    "SMCI": "Information Technology",
    "SQ": "Financials",
    "TER": "Information Technology",
    "TEAM": "Information Technology",
    "TTD": "Communication Services",
    "TSLA": "Consumer Discretionary",
    "VRSK": "Industrials",
    "WBD": "Communication Services",
    "WDC": "Information Technology",
    "ZS": "Information Technology",
}

CORE_SIGNAL_WEIGHTS = {
    "mom_6_1": 0.30,
    "mom_12_1": 0.20,
    "mom_3_1w": 0.10,
    "risk_adj_mom": 0.15,
    "prox_52w_high": 0.15,
    "sector_rel_strength": 0.10,
}
EXPERIMENTAL_OVERLAY_WEIGHTS = {
    "macd_hist": 0.00,
    "adx_14": 0.02,
    "rsi_14": -0.02,
}
EARLY_SIGNAL_WEIGHTS = {
    "mom_3_1w": 0.35,
    "sector_rel_strength": 0.25,
    "rank_delta_20": 0.25,
    "prox_52w_high": 0.15,
}

WINSOR_LOWER = 0.025
WINSOR_UPPER = 0.975
SMOOTHING_SPAN = 3
ENTRY_PERCENTILE = 0.85
EXIT_PERCENTILE = 0.55
TARGET_POSITIONS = 18
MIN_ACTIVE_NAMES = 8

MAX_NAME_WEIGHT = 0.10
MIN_NAME_WEIGHT = 0.03
MAX_SECTOR_WEIGHT = 0.25
MAX_ORDER_ADV_FRACTION = 0.05

STOP_LOSS_FLOOR = 0.12
STOP_LOSS_ATR_MULTIPLIER = 2.5
TRANSACTION_COST_BPS = 10.0

REGIME_LOOKBACK = 126
REALIZED_VOL_LOOKBACK = 20
REGIME_CASH_CUT = 0.50
REGIME_VOL_THRESHOLD = 0.25

REBALANCE_WEEKDAY = 4

RECENT_SYMBOL_ALIASES = {
    "BRK": ["BRK-B"],
    "SQ": ["XYZ"],
    "FI": ["FISV"],
    "MMC": ["MRSH"],
}


def ensure_output_dirs() -> None:
    for path in (
        DATA_DIR,
        RAW_DIR,
        PROCESSED_DIR,
        OUTPUT_DIR,
        METRICS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        LOCAL_CONFIG_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
