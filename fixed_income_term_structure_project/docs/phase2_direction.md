# Phase 2 Direction

## Purpose

Phase 2 is a clean extension beyond the existing Phase 1 fixed-income project.

Phase 1 remains the standalone term-structure construction and one-factor model comparison project:
- observed Treasury benchmark
- bootstrapped zero / discount / forward curves
- NSS smoothing
- Vasicek / CIR / Hull-White 1F comparison

Phase 2 adds a new question:

**What changes when we move from a single-curve Treasury workflow to a public-only multi-curve swap-pricing prototype?**

## Separation Rule

Phase 2 must not redefine Phase 1.

- Phase 1 outputs stay unchanged.
- Phase 1 dashboard stays unchanged.
- Phase 2 reads Phase 1 artifacts as inputs, but writes only into `data/processed/phase2` and `outputs/phase2`.
- Phase 2 has its own dashboard and its own explanation layer.

## Current Phase 2 Design

### Discounting
- public daily SOFR is used as the overnight anchor
- compounded SOFR windows create 1M / 3M / 6M / 12M short-end nodes
- the curve is extended beyond 1Y by shifting the Phase 1 NSS curve to join continuously at 1Y

This is labeled a **public discount proxy**, not a vendor-grade OIS discount curve.

### Projection
- the floating-leg projection curve is derived from the Phase 1 bootstrapped Treasury forward curve
- it is kept separate from the discount curve on every snapshot

### Pricing
- spot-start USD fixed-for-floating swaps
- 3M floating leg
- semiannual fixed leg
- 2Y / 5Y / 10Y maturities

### Comparison
- baseline: Phase 1 single-curve NSS swap pricing
- extension: Phase 2 multi-curve pricing
- decomposition:
  - discounting effect
  - projection effect
  - total gap

## Why This Is Useful

Phase 2 turns the term-structure project into a pricing architecture project:

- not just fitting curves
- but showing how curve choice changes swap rates, PV, and sensitivities

This is the intended "master class" extension beyond the standalone Phase 1 result.

## Explicit Limitations

Phase 2 does not try to be:
- a Bloomberg-grade OIS calibration engine
- a swaption/cap calibration project
- an HJM project
- a forecasting project

It is a public-only prototype focused on architecture, pricing mechanics, and sensitivity interpretation.
