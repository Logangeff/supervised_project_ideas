# Momentum Stock Strategy and Screener

## Executive summary

This project is built as a serious personal quant system with two outputs:

- a ranked daily screener
- a tradable long-only portfolio

The V1 architecture is deliberately conservative:

- universe: US large/mid-cap liquid equities
- cadence: daily end-of-day score recomputation, weekly rebalance
- portfolio mandate: long-only plus cash
- ranking engine: weighted composite, not machine learning
- data stack: free working implementation now, cheap-paid upgrade later

The design goal is not to create a flashy retail strategy. The goal is to build a robust, automatable momentum workflow that can be extended into a research platform.

## Standard momentum signals

The V1 system is centered on signals that are both industry-standard and easy to refresh:

- `12-1` momentum: trailing 252 trading days excluding the most recent 21 days
- `6-1` momentum: trailing 126 trading days excluding the most recent 21 days
- `3-1w` acceleration momentum: trailing 63 trading days excluding the most recent 5 days
- absolute trend filter: close above both 100-day and 200-day moving averages
- risk-adjusted momentum: `mom_6_1 / realized_vol_63`
- 52-week-high proximity: `close / rolling_high_252`
- sector-relative strength: stock 63-day return minus sector ETF 63-day return
- volume confirmation: log ratio of 20-day versus 60-day average dollar volume

These are robust, refreshable, and interpretable. They give the system a realistic chance of surviving transaction costs and weekly portfolio turnover.

## Newer signals worth testing

These are the best research-grade additions for later phases:

- residual momentum
- information discreteness / Frog-in-the-Pan
- industry momentum
- volatility-managed momentum at the portfolio exposure layer
- PEAD and earnings revision momentum if a paid event-data source is available

These are intentionally excluded from the base implementation so that V1 stays transparent and stable.

## Recommended V1 design

### Universe and watchlist

- primary universe: top US liquid common stocks from the current S&P 500 plus S&P 400 list
- implemented practical filters:
  - latest price above `$10`
  - 60-day median dollar volume above `$20M`
  - at least `260` trading days of history
- research overlay: personal watchlist names are highlighted inside the systematic universe

### Core ranking logic

Each day:

1. compute raw factor values
2. winsorize cross-sectionally
3. z-score across the eligible universe
4. combine with fixed signal weights
5. apply a trend multiplier
6. add a small liquidity adjustment
7. smooth final scores with a 3-day EMA

The system publishes two rankings:

- `core_rank`: the portfolio selection rank
- `early_leader_rank`: the watchlist-discovery rank

### Entry and exit logic

- score updates: daily after the close
- portfolio rebalance: weekly
- default entry: top-decile core score plus trend filter
- prefer stocks near 52-week highs or breaking 60-day highs
- exit on:
  - rank deterioration
  - two-day break below the 100-day moving average
  - catastrophic stop breach

### Position sizing and risk control

- target book size: roughly 12 to 20 names, default 15
- sizing: score-adjusted inverse-volatility weights
- risk controls:
  - max name weight 10%
  - min active weight 3%
  - max sector weight 25%
  - regime-based gross exposure cut using SPY trend and medium-term return

## Phase-by-phase implementation plan

### Phase 0: design lock

- freeze V1 mandate
- freeze signal definitions and weights
- freeze rebalance cadence and execution assumptions

### Phase 1: research data layer

- build a reproducible OHLCV store
- build the latest practical universe
- compute benchmark and sector ETF series
- create a processed feature panel

### Phase 2: baseline signals and screener

- implement core signals
- build `core_rank` and `early_leader_rank`
- export the latest top names, watchlist overlay, and research metrics

### Phase 3: portfolio engine

- implement weekly rebalance
- implement daily event-driven exits
- implement risk-aware weighting and sector caps
- export current portfolio and order recommendations

### Phase 4: backtesting and validation

- run walk-forward backtests with realistic next-session execution assumptions
- track CAGR, Sharpe, Sortino, drawdown, turnover, and gross exposure
- track screener metrics such as rank IC and top-bucket forward return quality

### Phase 5: advanced research

- residual momentum first
- Frog-in-the-Pan second
- industry momentum third
- keep machine-learned overlays out of scope until the plain composite is stable

### Phase 6: productionization

- scheduled daily runs
- stale-data detection
- reproducible snapshots
- alerting and portfolio diff outputs

## Data requirements

### Core V1 data

- daily adjusted OHLCV
- sector classification
- universe membership source
- benchmark ETF history
- sector ETF history

### Useful but not required in this implementation

- earnings dates
- point-in-time market cap
- cleaner point-in-time index membership
- analyst revisions
- factor returns for residual momentum

### Data path implemented now

- universe source: Wikipedia S&P 500 and S&P 400 constituent tables
- price source: `yfinance`

### Recommended upgrade path

- Polygon, Tiingo, or another EOD-capable provider
- point-in-time fundamentals and event data

## Backtesting and validation framework

The current system validates both the portfolio and the screener.

### Portfolio metrics

- CAGR
- annualized volatility
- Sharpe
- Sortino
- max drawdown
- Calmar
- total turnover
- average gross exposure
- average position count

### Screener metrics

- rank IC over 21-day and 63-day forward windows
- top-bucket hit rate
- top-bucket forward return
- top-decile versus median forward-return spread

### Execution assumptions

- signal computed after the close
- scheduled rebalance on the next session after the signal date
- transaction cost penalty applied through turnover

## Automation plan

The project is structured so it can be rerun automatically.

### CLI phases

- `refresh_data`
- `build_features`
- `score`
- `backtest`
- `report`
- `all`
- `smoke`

### Daily workflow

1. refresh data
2. recompute features
3. update rankings
4. refresh portfolio recommendations
5. write reports and figures

### Storage

- raw and processed panels written as Parquet
- latest research tables written as CSV
- warehouse copy written to DuckDB

### UI

- Streamlit dashboard for:
  - current rankings
  - early leaders
  - current portfolio
  - historical equity curve
  - screener validation metrics

## Risks, pitfalls, and what to avoid

- do not over-optimize thresholds
- do not trust free data as a full institutional backtest base
- do not mix a screener and portfolio that follow different logic
- do not chase short-term spikes without a trend filter
- do not add ML before the base feature set is stable
- do not ignore momentum crash risk
- do not assume current index membership is point-in-time clean

## Core signals for V1

- `mom_12_1`
- `mom_6_1`
- `mom_3_1w`
- `risk_adj_mom`
- `prox_52w_high`
- `sector_rel_strength`
- `trend_filter`
- `volume_confirmation`

## Optional advanced signals

- residual momentum
- information discreteness / Frog-in-the-Pan
- industry momentum
- earnings revision momentum
- PEAD-style event continuation
- volatility-managed gross exposure

## Experimental research ideas

- turnover-conditioned short-term momentum
- Kalman-smoothed latent momentum state
- HMM-style regime states
- gradient boosting or learning-to-rank overlay
- sentiment and options overlays
- intraday breakout logic

## Clear next steps

1. keep the current V1 stable and reproducible
2. rerun on a larger universe sample or on a better paid data stack
3. add residual momentum first
4. add Frog-in-the-Pan next
5. only then consider ML or more complex execution logic
