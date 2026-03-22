# TSLA Company News Exploration

This is a separate exploration branch built around one ticker, `TSLA`, using:

- Yahoo Finance daily price history
- Google News RSS historical headline collection
- the frozen Stage 1 claim detector from `numerical_claim_detection_project`
- frozen `ProsusAI/finbert` inference for sentiment

## Setup

```bat
cd tsla_company_news_exploration
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Main commands

```bat
python -m src.main --phase collect_data
python -m src.main --phase build_dataset
python -m src.main --phase stage1_materiality
python -m src.main --phase stage2_direction
python -m src.main --phase stage3_sentiment
python -m src.main --phase stage3_amplitude
python -m src.main --phase results
python -m src.main --phase all
```

## Main outputs

- `outputs/summaries/collection_summary.json`
- `outputs/summaries/daily_dataset_summary.json`
- `outputs/metrics/stage2_direction_summary.json`
- `outputs/metrics/stage3_sentiment_summary.json`
- `outputs/metrics/stage3_amplitude_summary.json`
- `outputs/figures/`
