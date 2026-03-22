# WRDS + OptionMetrics Idea Shortlist

This folder is a clean exploration branch for structured-data project ideas built around WRDS and OptionMetrics.

## Idea 1: Large Option Repricing / IV-Innovation Classification

### Core question
Can we predict whether a stock's standardized 30-day ATM implied volatility will experience an unusually large move over the next week or month?

### Target
- Binary classification:
  - `1` if the future change in standardized ATM implied volatility is in the top quintile
  - `0` otherwise
- Possible horizons:
  - next 5 trading days
  - next 20 trading days

### Main inputs
- current ATM implied volatility
- term-structure slope
- skew / smile features
- option volume and open interest
- recent stock returns and realized volatility

### Why it is strong
- directly option-focused
- financially meaningful for repricing and risk management
- easy to compare against simple persistence baselines
- scalable later into option-return or mispricing work

### Main caution
This area is real and important, but not brand-new. The project should be framed as a specific supervised prediction problem on standardized option features, not as a novel literature area.

## Idea 2: Realized-Volatility Regime Change Classification

### Core question
Can option-implied features improve prediction of whether a stock will move into a high-realized-volatility regime over the next month, relative to stock-only baselines?

### Target
- Binary classification:
  - `1` if next 20-trading-day realized volatility is in a high-volatility regime
  - `0` otherwise
- A practical first version:
  - define `high-volatility regime` as the top quintile of future 20-day realized volatility in the training set

### Main inputs
- stock-only features:
  - recent returns
  - recent realized volatility
  - turnover or volume if available
- option-implied features:
  - ATM implied volatility
  - term structure
  - skew
  - option volume / open interest
  - simple implied-moment proxies if practical

### Why it is strong
- very finance-relevant
- safer and cleaner than stock-direction prediction
- realistic for a student project
- easy to justify as an incremental-information study:
  - stock-only baseline
  - option-only baseline
  - combined model
- naturally scalable into stronger volatility forecasting or regime models later

### Main caution
Volatility forecasting is already a standard area. The contribution should come from the exact framing:
- supervised regime-change classification
- option-vs-stock incremental information
- careful standardized feature construction

## Current preference

We are leaning toward **Idea 2: Realized-Volatility Regime Change Classification** because it looks like the best balance of:
- feasibility
- finance relevance
- clean supervised framing
- course-level finishability
- room to scale later
