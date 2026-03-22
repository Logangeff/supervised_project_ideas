# Momentum Stock Strategy Project

This repo implements a serious personal quant momentum system with two outputs:

- a **daily end-of-day ranked screener**
- a **weekly rebalanced long-only portfolio**

It is designed as a realistic personal V1:

- transparent composite ranking instead of black-box ML
- robust medium-term momentum features
- daily ranking with weekly trading to control churn
- long-only plus cash with regime-aware exposure cuts
- free working data path now, clean upgrade path to better market data later

## Project structure

- [docs/project_direction.md](docs/project_direction.md): full strategy design and research roadmap
- [docs/quick_guide.md](docs/quick_guide.md): short how-to guide for running and reading outputs
- `src/`: data, features, ranking, portfolio, backtest, reporting
- `app/`: Streamlit dashboard
- `outputs/`: saved reports, figures, and metrics
- `config/watchlist.csv`: user-controlled WRDS stock list for watchlist mode

## What is implemented

- WRDS-backed stock pull from CRSP for user watchlists or a liquid-equity universe
- incremental WRDS cache with append-only behavior by `permno`
- raw WRDS CSV exports for direct inspection
- benchmark and sector ETF pull via `yfinance`
- core V1 signals:
  - `12-1` momentum
  - `6-1` momentum
  - `3-1w` acceleration
  - risk-adjusted momentum
  - 52-week-high proximity
  - sector-relative strength
  - trend filter
  - volume confirmation
- cross-sectional winsorization, z-scoring, and weighted composite ranking
- smoothed `core_rank` and `early_leader_rank`
- weekly rebalance portfolio with:
  - score-adjusted inverse-volatility sizing
  - sector caps
  - regime-aware gross exposure cuts
  - daily event exits for rank decay, trend breaks, and stop breaches
- screener and backtest reporting
- DuckDB warehouse export
- Streamlit dashboard

## Current implementation status

This is a **working V1 research and screening system**. It is appropriate for:

- personal momentum screening
- research iteration on signal design
- baseline long-only portfolio testing
- later upgrades such as residual momentum or industry momentum

### WRDS mode

If WRDS credentials are available, the project now defaults to `wrds` mode.

- stocks are pulled from WRDS CRSP
- raw stock history is cached in:
  - `data/raw/wrds_stock_price_cache.csv`
  - `data/raw/wrds_universe_source.csv`
- rerunning the code does **not** pull the full stock history again
- if you add a new ticker to `config/watchlist.csv`, only the missing stock history for that new name is appended
- clean universe toggle:
  - `python -m src.main --phase all --universe-mode watchlist`
  - `python -m src.main --phase all --universe-mode top_liquid --limit 300`
- Windows launchers:
  - `run_watchlist.bat`
  - `run_top_liquid.bat`

By default, if `config/watchlist.csv` exists, the WRDS path uses `watchlist` mode. That is the right setup for “my own names” like `AAPL`, `TSLA`, and similar stocks you actually want to monitor.

Current WRDS stock pulls come from:

- `crsp.stocknames` for ticker/permno/name metadata
- `crsp.dsf` for daily stock prices and volume

In this environment, both `crsp.dsf` and `crsp.dsi` currently max out at `2024-12-31`, so that is the latest date the WRDS-backed momentum run can reach without adding another data source.

## Important limitation

The free implementation is intentionally runnable, but it is **not fully survivorship-bias free**, because the free universe source is based on current index membership from Wikipedia.

That is acceptable for:

- live screening
- personal research
- architecture validation

It is **not** the same as a fully institutional historical data stack. For research-grade backtests, upgrade the data layer later.

The WRDS watchlist mode is better for real stock history, but it is still:

- a small watchlist cross-section, not a full cross-sectional equity research universe
- dependent on sector mapping approximations from SIC codes for some names
- limited by the CRSP availability window in your WRDS account

## Run

```bat
run_project.bat
```

For a smaller validation pass:

```bat
python -m src.main --phase smoke
```

## Dashboard

```bat
launch_dashboard.bat
```
