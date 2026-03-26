# Bloomberg Upgrade Guide

## Bottom Line

For the **current public-data project**, I think we have pushed **Phase 1 about as far as we reasonably can without Bloomberg**.

Why:
- we already use observed Treasury quotes
- we already use the official Fed nominal curve as an external benchmark
- we already validate our own NSS smoothing against that benchmark
- we already compare Vasicek, CIR, and Hull-White in a coherent way

So the next meaningful jump is **not more public-data polishing**. The next meaningful jump is:
- real market curve inputs from Bloomberg
- real discount / forwarding curve construction
- real swap-market validation

In other words:

> **Yes, for now I think we should mostly wait for Bloomberg data before trying to force a much bigger Phase 1 upgrade.**

---

## What Bloomberg Would Change

With Bloomberg, the project can move from:
- strong academic public-data curve construction

to:
- more realistic Treasury / OIS / swap curve construction
- proper multi-curve discounting and forwarding
- better swap pricing validation
- possibly cap / swaption calibration later

The best immediate Bloomberg upgrade would be:
1. Treasury curve nodes
2. SOFR / OIS discount curve nodes
3. USD plain-vanilla swap curve nodes

That is enough to materially upgrade both Phase 1 and Phase 2.

---

## Recommended Minimum Bloomberg Data Pull

### 1. Treasury benchmark curve nodes
Use historical daily or weekly yields for standard maturities:
- `1Y`
- `2Y`
- `3Y`
- `5Y`
- `7Y`
- `10Y`
- `20Y`
- `30Y`

This lets you:
- compare Bloomberg Treasury inputs vs FRED Treasury inputs
- rebuild a cleaner Treasury curve
- compare our public-data NSS fit against a richer market-data source

### 2. SOFR / OIS discount curve nodes
Use historical daily or weekly rates for:
- short end: `1M`, `3M`, `6M`, `12M`
- long end: `2Y`, `3Y`, `5Y`, `7Y`, `10Y`, `30Y`

This lets you:
- build a proper discount curve
- replace the current Phase 2 public SOFR proxy

### 3. USD swap curve nodes
Use historical par swap rates for:
- `1Y`
- `2Y`
- `3Y`
- `5Y`
- `7Y`
- `10Y`
- `30Y`

This lets you:
- validate swap pricing directly on swap-market data
- build a real forwarding curve workflow

---

## Data Volume

For this project, Bloomberg data volume should stay small.

If you only pull:
- curve nodes
- daily or weekly history
- a few maturities
- one or two fields

then the export will usually be:
- **small**
- nowhere near `65 GB`
- usually in the **MB** range, not huge files

The volume becomes large only if you start pulling:
- intraday tick data
- huge security universes
- quote-by-quote histories
- broad derivatives surfaces across long periods

That is **not necessary** for the current project.

---

## Practical Bloomberg Workflow

## A. General terminal commands you will use repeatedly

These are the core Bloomberg functions you will almost certainly use:

- `SRCH <GO>`
  - search for the right security / rate / curve member
- `DES <GO>`
  - inspect and validate what the ticker actually is
- `HP <GO>`
  - historical price / rate table
- `GP <GO>`
  - chart the time series first before exporting
- `FLDS <GO>`
  - check which Bloomberg field to export
- `SWPM <GO>`
  - swap pricing and market convention reference

These are the most important commands for the project.

---

## B. Treasury benchmark curve nodes

### Terminal workflow
1. Search / validate the ticker:
   - `SRCH <GO>`
   - then `DES <GO>` on the candidate
2. Visual sanity check:
   - `GP <GO>`
3. Historical table:
   - `HP <GO>`
4. Field check:
   - `FLDS <GO>`

### Typical Bloomberg generic yield tickers
These are standard examples and should be validated on terminal with `DES <GO>`:

- `USGG1YR Index`
- `USGG2YR Index`
- `USGG3YR Index`
- `USGG5YR Index`
- `USGG7YR Index`
- `USGG10YR Index`
- `USGG20YR Index`
- `USGG30YR Index`

### Excel extraction
If you already know the ticker:

```excel
=BDH("USGG10YR Index","PX_LAST","20190101","20260320","Per=Mn","Fill=P")
```

For several nodes, it is better to put the ticker list in column `A` and use:

```excel
=BDH(A2,"PX_LAST",$B$1,$C$1,"Per=Mn","Fill=P")
```

Where:
- `A2` = validated Bloomberg ticker
- `B1` = start date like `20190101`
- `C1` = end date like `20260320`

### What to save
Save one file like:

`treasury_curve_history.xlsx`

Recommended columns:
- `date`
- `ticker`
- `maturity_label`
- `rate`

---

## C. SOFR / OIS discount curve nodes

### Terminal workflow
Because Bloomberg identifiers here can vary by curve family and setup, use:

1. `SRCH <GO>`
   - search for USD SOFR OIS rate instruments / curve members
2. `DES <GO>`
   - verify instrument and conventions
3. `HP <GO>`
   - inspect history
4. `FLDS <GO>`
   - confirm export field, usually a rate field such as `PX_LAST`

### Important note
For OIS / SOFR, **do not rely on a guessed ticker without validating it**.
These identifiers can depend on:
- region
- curve source
- Bloomberg curve family naming
- market conventions

So the **exact command path** is:

- `SRCH <GO>`
- `DES <GO>`
- `HP <GO>`

### Excel extraction
Once the validated ticker is in cell `A2`:

```excel
=BDH(A2,"PX_LAST",$B$1,$C$1,"Per=Mn","Fill=P")
```

### What to save
Save one file like:

`sofr_ois_curve_history.xlsx`

Recommended columns:
- `date`
- `ticker`
- `maturity_label`
- `rate`

---

## D. USD swap curve nodes

### Terminal workflow
Again, validate the exact ticker first:

1. `SRCH <GO>`
   - search for USD vanilla swap rates / swap curve members
2. `DES <GO>`
   - confirm maturity, floating leg, fixed leg conventions
3. `HP <GO>`
   - verify historical coverage
4. `SWPM <GO>`
   - use for conventions / sanity checking

### Excel extraction
Once the ticker is validated and placed in `A2`:

```excel
=BDH(A2,"PX_LAST",$B$1,$C$1,"Per=Mn","Fill=P")
```

### What to save
Save one file like:

`usd_swap_curve_history.xlsx`

Recommended columns:
- `date`
- `ticker`
- `maturity_label`
- `par_swap_rate`

---

## E. Optional later data: caps / swaptions

This is **not needed right away**.

Only pull this if you want a later calibration project:
- cap volatilities
- swaption volatility matrix / surface

Terminal tools you would likely use:
- `SRCH <GO>`
- `DES <GO>`
- `HP <GO>`
- `SWPM <GO>`

I would **not** start there.

---

## Quick Guide: Exporting to Excel

## Option 1: Direct terminal export
1. Open the instrument
2. Use `HP <GO>` for the history table
3. Set:
   - date range
   - periodicity (`daily`, `weekly`, or `monthly`)
4. Use the terminal menu:
   - `Actions` -> `Output` -> `Excel`

This is the easiest first method.

## Option 2: Bloomberg Excel Add-In
If the Bloomberg Excel add-in is installed:

1. Open Excel
2. Use the Bloomberg ribbon
3. Use **Import Data** -> **Historical End of Day**
4. Or use formulas directly:

```excel
=BDH(A2,"PX_LAST",$B$1,$C$1,"Per=Mn","Fill=P")
```

Useful companion formulas:

```excel
=BDP(A2,"SECURITY_DES")
```

```excel
=BDP(A2,"ID_BB_GLOBAL")
```

Use `FLDS <GO>` on terminal if you are unsure about the correct field name.

---

## Recommended Extraction Order

If you only have limited Bloomberg time, do this in order:

1. **Treasury curve nodes**
2. **SOFR / OIS curve nodes**
3. **USD swap curve nodes**
4. caps / swaptions only later

This gives the biggest return for the least effort.

---

## Suggested Files to Create

At minimum:

- `treasury_curve_history.xlsx`
- `sofr_ois_curve_history.xlsx`
- `usd_swap_curve_history.xlsx`

If possible, store them later as CSV too:

- `treasury_curve_history.csv`
- `sofr_ois_curve_history.csv`
- `usd_swap_curve_history.csv`

That will make coding much easier.

---

## Final Recommendation

For the **current project state**:

- **Phase 1 public-data version is already strong**
- **the next major upgrade should come from Bloomberg**
- **do not keep endlessly expanding public-data complexity**

So my recommendation is:

> Keep the current Phase 1 as the completed public-data benchmark version, and wait for Bloomberg to build the next serious upgrade: Treasury + OIS + swap curve construction with real market inputs.
