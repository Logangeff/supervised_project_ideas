# Project Direction

This note is the current working plan for the new project. Its purpose is to keep the scope stable and to record, in one place, what the project is actually trying to do, why it is split into two stages, and how far the proposed contribution can be defended relative to existing literature.

## 1. Current overall direction

The project will be organized in two stages.

**Stage 1** will be the core course project: numerical claim detection in financial text. The task is supervised, focused, and recent enough to be interesting, but still safe enough to execute cleanly for the course.

**Stage 2** will be a narrower downstream extension: use the Stage 1 detector inside a simple event-level finance prediction pipeline. The goal of Stage 2 is not to pretend that numerical claim detection plus market analysis is a new broad idea. The goal is to formulate one specific predictive question that is small enough to implement, clear enough to explain, and distinct enough to count as a real extension.

At this point, Stage 1 is the part that must remain solid no matter what. Stage 2 is the contribution layer that can strengthen the project if the data and setup remain manageable.

## 2. Why the project is split into two stages

Stage 1 is the safe and defensible part. It has a clear label space, a recent finance-domain paper, and an implementation path that is mostly about modeling and evaluation rather than difficult data collection. This makes it a strong course project on its own.

Stage 2 exists because Stage 1 by itself is already a published task. That does not make it a bad project, but it does mean that the contribution should come from a more specific downstream question rather than from pretending that the core task is new. The cleanest way to do that is to use the numerical-claim detector as one module inside a downstream predictive experiment and ask whether it improves a finance-relevant target.

The intended logic is therefore simple:

- Stage 1 gives a strong, feasible supervised NLP project.
- Stage 2 gives the project a more personal and finance-facing contribution.

## 3. What is already claimed in the literature

The broad idea of numerical claim detection in finance is already claimed. More importantly, the broad idea of combining numerical claim detection with downstream financial analysis is also already present in the literature.

The 2024 paper *Numerical Claim Detection in Finance: A New Financial Dataset, Weak-Supervision Model, and Market Analysis* does more than sentence classification. The authors explicitly discuss the influence of detected claims on financial outcomes, construct a claim-derived optimism measure, and study its relationship with earnings surprise and abnormal returns. So a generic proposal of the form "detect numerical claims and then study market outcomes" would not be sufficiently distinctive.

This matters for how the project should be framed. The project should not claim novelty at the level of the broad two-stage idea. Instead, it should claim a narrower and more concrete extension built around a clearly defined prediction setup.

## 4. What still looks defensible

What still looks defensible is a narrower event-level prediction problem built around a clear ablation. In other words, the contribution should not be "we also study market effects," but rather "we test whether automatically detected numerical-claim content improves a specific predictive task under a fixed protocol."

The most defensible version currently identified is:

> Does automatically detected numerical-claim information improve event-level prediction of next-day return direction from financial news headlines, relative to market-only and full-text baselines?

This is narrower than the existing paper in three useful ways. First, it is cast as a direct supervised forecasting comparison rather than a broad market-association study. Second, it is tied to event-level prediction from dated, asset-linked headlines. Third, it produces a clean ablation question: does claim-filtering add value beyond simply using all headline text?

## 5. Restricted Stage 2 plan

To keep Stage 2 from turning into a data-engineering project, it needs to be restricted hard from the start. The current minimal version is the one below.

### 5.1 Research question

The Stage 2 question will be:

> Does automatically detected numerical-claim information improve prediction of next-day stock return direction compared with market-only and full-text baselines?

This is deliberately narrower than abnormal-return modeling, event ranking, sector analysis, or claim typing. Those can stay as later extensions if the core version works.

### 5.2 Dataset choice

The safest downstream dataset currently identified is:

- Hugging Face: `luckycat37/financial-news-dataset`
- original repository: `FelixDrinkall/financial-news-dataset`

This dataset is attractive because it already includes the fields that make a first predictive experiment possible without building a large data pipeline:

- `title`
- `date_publish`
- `mentioned_companies`
- `prev_day_price_{ticker}`
- `curr_day_price_{ticker}`
- `next_day_price_{ticker}`

That means the downstream label can be defined from the dataset itself, and the company linkage is already largely handled. This is exactly what makes it safer than more ambitious alternatives.

### 5.3 Unit of analysis

The unit of analysis will be one article title, one company ticker, and one publication date.

To keep the data clean, the initial version of Stage 2 should keep only rows that satisfy all of the following:

- the article is in English
- `title` is present
- `mentioned_companies` contains exactly one clearly mapped company ticker
- the relevant `prev_day_price`, `curr_day_price`, and `next_day_price` fields for that ticker are all present

This restriction is important. It removes the need for ambiguous multi-company attribution in the first version and keeps the project focused on modeling rather than entity resolution.

### 5.4 Prediction target

The safest first target is a binary next-day direction label defined by the sign of the move from publication day close to next-day close:

```
y = 1 if next_day_price > curr_day_price
y = 0 if next_day_price < curr_day_price
```

Rows with exactly zero price change can be dropped.

More explicitly, any row with `next_day_price == curr_day_price` will be removed from the binary classification dataset.

This is not the most sophisticated finance target, but it is the safest one for a first extension because it does not require building an abnormal-return pipeline before the basic idea has been tested. If Stage 2 works and remains manageable, the target can later be upgraded to abnormal-return direction or market-movingness.

### 5.5 Split and evaluation protocol

The downstream experiment should use a chronological split rather than a random split. That means train, validation, and test periods will be separated by publication date in order to avoid leakage from future news into past predictions.

In the first version, this should be implemented as simple ordered date blocks, for example the earliest 70% of eligible rows for training, the next 15% for validation, and the final 15% for testing. No row from a later period should be allowed to leak into training.

The first version should report at least:

- accuracy
- macro-F1
- class balance
- confusion matrices for the main models

The evaluation should stay deliberately simple. The point of Stage 2 is to test whether claim-filtering helps, not to build a large financial forecasting benchmark.

### 5.6 Models to compare

The restricted Stage 2 comparison should stay small.

**Market-only baseline.**  
This model will ignore text and use only a minimal set of price-derived information available around the event. In the strict minimal version, that can be the same-day return implied by `prev_day_price` and `curr_day_price`. This baseline is intentionally simple and intentionally weak: its role is not to be a strong financial forecasting system, but to answer the narrow question of whether text adds useful signal beyond immediately available price information. If needed, a single benchmark market series such as SPY can later be added from `yfinance`, but the first version should avoid expanding the feature set too much.

**All-text baseline.**  
This model will use the full article title as input. A simple TF-IDF plus Logistic Regression pipeline is sufficient for the first version.

**Claim-filtered-text baseline.**  
This model will use the Stage 1 detector to identify claim-bearing titles, but it should remain on the same downstream event universe as the other baselines. The cleanest first version is therefore a claim-aware variant of the all-text model that adds a claim flag or claim score derived from Stage 1 while keeping the same eligible rows. This makes the main comparison fairer because market-only, all-text, and claim-aware models are all evaluated on the same events.

The most operational first implementation is to append a single extra scalar feature to the TF-IDF representation, either a binary claim flag or the Stage 1 predicted claim probability.

If time permits, a secondary analysis can then be added on the subset of events predicted to contain numerical claims. That subset analysis is interesting, but it should remain secondary rather than replace the main same-universe comparison.

### 5.7 What Stage 2 is explicitly not

For the initial implementation, Stage 2 should **not** include:

- multiple downstream targets
- claim-type annotation
- sector-by-sector analysis
- pre- versus post-earnings timing analysis
- trading simulation
- a large multi-model benchmark

All of these are plausible later ideas, but they would make the first extension too broad and too fragile.

## 6. Practical interpretation

The project therefore has a clean shape.

Stage 1 is a complete supervised NLP project on its own: detect numerical claims in financial text. If that part is implemented carefully, the course project is already defensible.

Stage 2 is the controlled extension: take a downstream news dataset with built-in company linkage and price fields, define a very simple event-level direction target, and test whether restricting attention to claim-bearing headlines improves prediction relative to simpler baselines.

That is the version that seems strongest right now because it adds a concrete contribution without immediately turning into a full-scale market-prediction project.

## 7. Current recommendation

The current recommendation is:

1. Build Stage 1 first and treat it as the non-negotiable core.
2. If Stage 1 is stable, implement the restricted version of Stage 2 exactly as described above.
3. Only after that, consider whether a stronger target such as abnormal-return direction is worth the extra work.

This keeps the project honest, feasible, and explainable.

## 8. Sources

- ACL Anthology paper: https://aclanthology.org/2024.fever-1.21/
- arXiv version: https://arxiv.org/abs/2402.11728
- Hugging Face dataset: https://huggingface.co/datasets/luckycat37/financial-news-dataset
- Original dataset repository: https://github.com/FelixDrinkall/financial-news-dataset
- FNSPID paper: https://arxiv.org/abs/2402.06698
- FNSPID repository: https://github.com/Zdong104/FNSPID_Financial_News_Dataset
