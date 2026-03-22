# Quick Guide

This guide is for using the project as a **momentum screener** and a **portfolio research tool**.

## 1. What the project does

The project has 2 main modes:

- `watchlist`: score only the stocks you care about, like `AAPL`, `TSLA`, `NVDA`
- `top_liquid`: score a broader liquid stock universe from WRDS

It produces:

- a ranked stock screener
- a model portfolio
- backtest and screener summary files
- a Streamlit dashboard

## 2. What data it uses

Historical stock data comes from **WRDS CRSP**:

- `crsp.stocknames`
- `crsp.dsf`

In your current WRDS environment, CRSP stops at **2024-12-31**.

To get more recent data, the project adds a **recent market extension** on top so signals can reach about **T-1**.

Raw inspectable files are stored in:

- `data/raw/wrds_stock_price_cache.csv`
- `data/raw/wrds_universe_source.csv`
- `data/raw/recent_market_extension_cache.csv`

## 3. First-time setup

### WRDS credentials

Put your WRDS credentials in:

- `config/wrds_credentials.json`

If that file exists, the project will use WRDS automatically.

### Watchlist

Edit:

- `config/watchlist.csv`

Put one ticker per row. Example:

```csv
ticker
AAPL
TSLA
NVDA
MSFT
```

## 4. Easiest way to run it

Open a terminal in:

- `c:\Users\xxtri\Desktop\Machine Learning\PROJECT\momentum_stock_strategy_project`

### Option A: run your personal watchlist

```bat
run_watchlist.bat
```

This will:

- pull missing WRDS data for your watchlist
- append only new rows when possible
- compute features
- score the stocks
- run the portfolio logic
- write reports

### Option B: run the broader liquid universe

```bat
run_top_liquid.bat
```

This currently runs a broader WRDS liquid-universe version with a practical limit.

### Option C: launch the dashboard

```bat
launch_dashboard.bat
```

## 5. What happens when you rerun it

The WRDS path is incremental.

That means:

- if you rerun tomorrow, it should append only missing rows
- it does **not** repull the full stock history every time
- if you add a new stock to `config/watchlist.csv`, it pulls history for that new stock too

This is why the CSV caches are kept in `data/raw/`.

## 6. Most important output files

### Current screener / portfolio outputs

Open these first:

- `outputs/reports/latest_core_rank.csv`
- `outputs/reports/latest_early_rank.csv`
- `outputs/reports/latest_watchlist_overlay.csv`
- `outputs/reports/latest_target_portfolio.csv`
- `outputs/reports/latest_order_recommendations.csv`

### Performance summaries

- `outputs/metrics/backtest_summary.json`
- `outputs/metrics/screener_summary.json`
- `outputs/metrics/pipeline_summary.json`

## 7. How to read the main outputs

### `latest_core_rank.csv`

This is the main ranking for the strategy.

Use it to answer:

- which names have the strongest current momentum profile?
- where does my stock rank right now?

### `latest_early_rank.csv`

This is more aggressive.

It gives more weight to:

- shorter-horizon acceleration
- sector-relative strength
- recent rank improvement

Use it as an **idea discovery** list, not a final buy list.

### `latest_watchlist_overlay.csv`

This is the easiest file for your own names.

Use it to see:

- rank of each watchlist stock
- core score
- early-leader score
- whether it passes the trend filter

### `latest_target_portfolio.csv`

This is the model portfolio output.

It shows:

- selected names
- target weights
- signal strength

### `latest_order_recommendations.csv`

This tells you what the model would change relative to the previous portfolio state.

Use it to see:

- adds
- trims
- exits

## 8. How to use it in practice

### For screening

Best workflow:

1. Run `run_watchlist.bat`
2. Open `outputs/reports/latest_watchlist_overlay.csv`
3. Check whether your stocks are:
   - high in `core_rank`
   - improving in `early_leader_rank`
   - still above the trend filter
4. Open the dashboard if you want a visual view

### For broader idea generation

Best workflow:

1. Run `run_top_liquid.bat`
2. Open `outputs/reports/latest_core_rank.csv`
3. Look at the top-ranked names
4. Cross-check with `latest_target_portfolio.csv`

## 9. What the scores roughly mean

Higher is better.

The ranking is built from:

- `12-1` momentum
- `6-1` momentum
- `3-1w` acceleration
- risk-adjusted momentum
- 52-week-high proximity
- sector-relative strength
- trend confirmation

So a strong name usually has:

- medium-term price strength
- price near highs
- good sector-relative behavior
- acceptable volatility
- price above long moving averages

## 10. What not to do

Do **not** treat the current output as a blind auto-trading system.

Current limitations:

- recent data is extended beyond WRDS with a secondary source
- some ticker mappings can still fail on the recent extension
- the broad-universe Sharpe is only moderate, not elite
- the system is better as a **screening and research tool** than a fully trusted automatic trading engine

So the right use today is:

- idea generation
- ranking your watchlist
- supporting buy/sell research

Not:

- fully automatic live trading with no review

## 11. Useful terminal commands

### Full watchlist run

```bat
python -m src.main --phase all --start 2018-01-01 --universe-mode watchlist
```

### Full broader-universe run

```bat
python -m src.main --phase all --start 2018-01-01 --universe-mode top_liquid --limit 300
```

### Quick smoke test

```bat
python -m src.main --phase smoke --start 2022-01-01 --universe-mode watchlist
```

### Launch dashboard

```bat
streamlit run app/streamlit_app.py
```

## 12. Recommended simple workflow

If you are just starting, do this:

1. edit `config/watchlist.csv`
2. run `run_watchlist.bat`
3. open `outputs/reports/latest_watchlist_overlay.csv`
4. open `outputs/reports/latest_target_portfolio.csv`
5. launch the dashboard with `launch_dashboard.bat`

That is the easiest way to use the project without touching the code.
