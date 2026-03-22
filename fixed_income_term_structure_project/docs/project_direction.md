# Project Direction

## Why This Project

This project was chosen because the course notes naturally support a phased fixed-income term-structure project:

- **Chapter 2** supports bootstrapping, Treasury curves, swap curves, and yield-curve smoothing.
- **Chapter 3** supports the one-factor short-rate model core: **Vasicek**, **CIR**, and **Hull-White**.
- **Chapter 5** supports implementation, fitting, and calibration logic.
- **Chapter 6 and later** are deliberately postponed in v1.

That makes the project course-aligned without trying to do too much at once.

## Core v1 Design

The v1 project follows this logic:

1. build Treasury-based par, zero, discount, and forward curves
2. smooth the benchmark zero curve with **Nelson-Siegel-Svensson**
3. compare one-factor models:
   - **Vasicek**
   - **CIR**
   - **Hull-White 1F**
4. test pricing-related usefulness through simple bond and par-swap outputs

## Why U.S. Treasury Data

U.S. Treasury term-structure data is the right v1 choice because it is:

- public
- easy to source repeatedly
- sufficient for curve construction and one-factor model comparison
- independent of Bloomberg, WRDS, or swap-market subscriptions

That keeps the project feasible and reproducible.

## Chosen Model Shortlist

### Curve construction and smoothing
- bootstrapped Treasury zero curve
- discount curve
- forward curve
- **Nelson-Siegel-Svensson** smoothed benchmark

### One-factor models
- **Vasicek**
- **CIR**
- **Hull-White 1F**

## Interpretation Rules

- **Vasicek** and **CIR** are the parsimonious equilibrium-model comparators.
- **Hull-White 1F** is the arbitrage-free exact-fit benchmark.

That means Hull-White is not presented as a fair free-parameter fit competitor on current-curve fit error. Its role is to anchor exactly to the observed benchmark curve and provide a pricing-oriented reference.

## What Is Explicitly Out of Scope in v1

The following are intentionally excluded from the first version:

- full OIS discounting
- multi-curve swap pricing
- HJM
- cap / swaption calibration
- Kalman filter estimation
- Diebold-Li forecasting
- credit-risk chapters

These are valid extensions only after the base project is complete and stable.

## Phase Roadmap

### Phase 1: Curve construction and smoothing
- fetch public U.S. Treasury series
- build monthly end-of-period snapshots
- bootstrap par, zero, discount, and forward curves
- fit Nelson-Siegel-Svensson as the smooth benchmark

### Phase 2: One-factor model comparison
- fit Vasicek and CIR to the benchmark zero curve
- construct Hull-White as the exact-fit current-curve benchmark
- compare fit, stability, and curve quality

### Phase 3: Pricing and usefulness
- build model-implied zero, discount, and forward curves
- compute simple single-curve par swap rates
- price simple coupon bonds relative to the benchmark curve

## Expected Contribution

The project’s main contribution is not a new model. It is a disciplined comparison of:

- a public-data curve-construction benchmark
- a smooth parametric representation
- equilibrium short-rate models
- an arbitrage-free anchored benchmark

under a single, transparent workflow that links curve construction to pricing usefulness.
