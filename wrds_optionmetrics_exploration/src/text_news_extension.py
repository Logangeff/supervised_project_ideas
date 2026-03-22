from __future__ import annotations

import json
import lzma
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import (
    BUILD_TEXT_NEWS_PANEL_SUMMARY_PATH,
    CLASSICAL_C_GRID,
    LOCKED_CLAIM_DETECTOR_META,
    OPTION_FEATURES,
    PART1_LABEL_NAMES,
    SEED,
    STOCK_FEATURES,
    STOCK_IDENTIFIER_MAP_PATH,
    SURFACE_EXTENSION_PANEL_PATH,
    SURFACE_FEATURES,
    TEXT_FINBERT_BATCH_SIZE,
    TEXT_FINBERT_MAX_LENGTH,
    TEXT_FINBERT_MODEL_ID,
    TEXT_NEWS_ARCHIVE_DIR,
    TEXT_NEWS_ARTICLE_PANEL_PATH,
    TEXT_NEWS_CLAIM_FINBERT_FIGURE,
    TEXT_NEWS_CLAIM_FINBERT_MODEL_PATH,
    TEXT_NEWS_CLAIM_SENTIMENT_FIGURE,
    TEXT_NEWS_CLAIM_SENTIMENT_MODEL_PATH,
    TEXT_NEWS_CORE_FIGURE,
    TEXT_NEWS_DAILY_FEATURE_PANEL_PATH,
    TEXT_NEWS_END_DATE,
    TEXT_NEWS_EVENT_CLAIM_FINBERT_FIGURE,
    TEXT_NEWS_EVENT_CLAIM_FINBERT_MODEL_PATH,
    TEXT_NEWS_EVENT_CORE_FIGURE,
    TEXT_NEWS_EVENT_FINBERT_FIGURE,
    TEXT_NEWS_EVENT_FINBERT_MODEL_PATH,
    TEXT_NEWS_EVENT_PANEL_PATH,
    TEXT_NEWS_EXTENSION_METRICS_CSV,
    TEXT_NEWS_EXTENSION_METRICS_JSON,
    TEXT_NEWS_EXTENSION_PANEL_PATH,
    TEXT_NEWS_EXTENSION_PREDICTIONS_CSV,
    TEXT_NEWS_FEATURES,
    TEXT_NEWS_FINBERT_FIGURE,
    TEXT_NEWS_FINBERT_MODEL_PATH,
    TEXT_NEWS_NEWS_ONLY_FIGURE,
    TEXT_NEWS_NEWS_ONLY_MODEL_PATH,
    TEXT_NEWS_SENTIMENT_FIGURE,
    TEXT_NEWS_SENTIMENT_MODEL_PATH,
    TEXT_NEWS_START_DATE,
    TEXT_NEWS_TIMEZONE,
    TRAIN_END_DATE,
    VALIDATION_END_DATE,
)
from .evaluation import compute_metrics, save_confusion_matrix_figure
from .utils import normalize_ticker, write_json, write_rows_csv


CORE_FEATURES = STOCK_FEATURES + OPTION_FEATURES + SURFACE_FEATURES
DATASET_SENTIMENT_NEWS_FEATURES = [
    "news_count",
    "news_count_log1p",
    "sentiment_neg_mean",
    "sentiment_neu_mean",
    "sentiment_pos_mean",
    "sentiment_net_mean",
]
DAILY_FINBERT_NEWS_FEATURES = [
    "news_count",
    "news_count_log1p",
    "finbert_neg_mean",
    "finbert_neu_mean",
    "finbert_pos_mean",
    "finbert_net_mean",
]
DAILY_CLAIM_FINBERT_NEWS_FEATURES = [
    "news_count",
    "news_count_log1p",
    "claim_prob_mean",
    "claim_prob_max",
    "claim_count_above_05",
    "finbert_neg_mean",
    "finbert_neu_mean",
    "finbert_pos_mean",
    "finbert_net_mean",
]
ALL_DAILY_NEWS_FEATURES = sorted(set(TEXT_NEWS_FEATURES + DAILY_FINBERT_NEWS_FEATURES))
EVENT_FINBERT_FEATURES = [
    "news_count",
    "news_count_log1p",
    "event_finbert_neg",
    "event_finbert_neu",
    "event_finbert_pos",
    "event_finbert_net",
]
EVENT_CLAIM_FINBERT_FEATURES = [
    "news_count",
    "news_count_log1p",
    "event_claim_prob",
    "event_claim_flag",
    "event_finbert_neg",
    "event_finbert_neu",
    "event_finbert_pos",
    "event_finbert_net",
    "event_sentiment_neg",
    "event_sentiment_neu",
    "event_sentiment_pos",
    "event_sentiment_net",
]


def _load_surface_extension_panel() -> pd.DataFrame:
    if not SURFACE_EXTENSION_PANEL_PATH.exists():
        raise RuntimeError("Surface extension panel is missing. Run build_surface_factors first.")
    frame = pd.read_parquet(SURFACE_EXTENSION_PANEL_PATH)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    return frame


def _news_years() -> list[int]:
    return list(range(int(TEXT_NEWS_START_DATE[:4]), int(TEXT_NEWS_END_DATE[:4]) + 1))


def _news_year_path(year: int) -> Path:
    return TEXT_NEWS_ARCHIVE_DIR / f"{year}_processed.json.xz"


def _resolve_claim_model_path() -> Path:
    metadata = json.loads(LOCKED_CLAIM_DETECTOR_META.read_text(encoding="utf-8"))
    if metadata.get("model_type") != "classical":
        raise RuntimeError("The locked claim detector is not a classical bundle in the current environment.")
    return LOCKED_CLAIM_DETECTOR_META.parent.parent.parent / metadata["path"]


def _score_claim_probabilities(texts: list[str]) -> np.ndarray:
    bundle = joblib.load(_resolve_claim_model_path())
    features = bundle["vectorizer"].transform(texts)
    return np.asarray(bundle["classifier"].predict_proba(features)[:, 1], dtype=np.float32)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_finbert_for_inference() -> tuple[AutoTokenizer, AutoModelForSequenceClassification, torch.device]:
    if not hasattr(_load_finbert_for_inference, "_cache"):
        tokenizer = AutoTokenizer.from_pretrained(TEXT_FINBERT_MODEL_ID)
        model = AutoModelForSequenceClassification.from_pretrained(TEXT_FINBERT_MODEL_ID)
        device = _device()
        model.to(device)
        model.eval()
        _load_finbert_for_inference._cache = (tokenizer, model, device)
    return _load_finbert_for_inference._cache  # type: ignore[attr-defined]


def _score_finbert_titles(texts: list[str]) -> pd.DataFrame:
    if not texts:
        return pd.DataFrame(columns=["finbert_pos", "finbert_neg", "finbert_neu", "finbert_net"])
    tokenizer, model, device = _load_finbert_for_inference()
    id2label = {int(idx): str(label).strip().lower() for idx, label in model.config.id2label.items()}
    rows: list[dict[str, float]] = []
    with torch.no_grad():
        for start in range(0, len(texts), TEXT_FINBERT_BATCH_SIZE):
            batch_texts = texts[start : start + TEXT_FINBERT_BATCH_SIZE]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=TEXT_FINBERT_MAX_LENGTH,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            probabilities = torch.softmax(model(**encoded).logits, dim=-1).detach().cpu().tolist()
            for probs in probabilities:
                prob_map = {id2label[idx]: float(prob) for idx, prob in enumerate(probs)}
                positive = prob_map.get("positive", 0.0)
                negative = prob_map.get("negative", 0.0)
                neutral = prob_map.get("neutral", 0.0)
                rows.append(
                    {
                        "finbert_pos": positive,
                        "finbert_neg": negative,
                        "finbert_neu": neutral,
                        "finbert_net": positive - negative,
                    }
                )
    return pd.DataFrame(rows)


def _assign_trade_dates(published_at_local: pd.Series, trade_dates: pd.Series) -> pd.Series:
    sorted_trade_dates = pd.Series(pd.to_datetime(trade_dates).sort_values().unique())

    def assign_one(timestamp: pd.Timestamp):
        if pd.isna(timestamp):
            return pd.NaT
        candidate = timestamp.tz_localize(None).normalize()
        if timestamp.hour >= 16:
            candidate = candidate + pd.Timedelta(days=1)
        idx = sorted_trade_dates.searchsorted(candidate)
        if idx >= len(sorted_trade_dates):
            return pd.NaT
        return sorted_trade_dates.iloc[int(idx)]

    return published_at_local.map(assign_one)


def _cached_build_available() -> bool:
    required_paths = (
        TEXT_NEWS_ARTICLE_PANEL_PATH,
        TEXT_NEWS_DAILY_FEATURE_PANEL_PATH,
        TEXT_NEWS_EXTENSION_PANEL_PATH,
        TEXT_NEWS_EVENT_PANEL_PATH,
        BUILD_TEXT_NEWS_PANEL_SUMMARY_PATH,
    )
    if not all(path.exists() for path in required_paths):
        return False
    try:
        article_columns = set(pd.read_parquet(TEXT_NEWS_ARTICLE_PANEL_PATH, columns=["finbert_neg", "finbert_pos"]).columns)
        daily_columns = set(pd.read_parquet(TEXT_NEWS_DAILY_FEATURE_PANEL_PATH).columns)
        event_columns = set(pd.read_parquet(TEXT_NEWS_EVENT_PANEL_PATH).columns)
    except Exception:
        return False
    return (
        {"finbert_neg", "finbert_pos"} <= article_columns
        and {"finbert_neg_mean", "finbert_pos_mean", "claim_prob_mean"} <= daily_columns
        and {"event_claim_prob", "event_finbert_net"} <= event_columns
    )


def build_text_news_panel() -> dict[str, object]:
    if _cached_build_available():
        return json.loads(BUILD_TEXT_NEWS_PANEL_SUMMARY_PATH.read_text(encoding="utf-8"))

    surface_panel = _load_surface_extension_panel()
    stock_identifier_map = pd.read_parquet(STOCK_IDENTIFIER_MAP_PATH)
    ticker_to_permno = {
        normalize_ticker(str(row["universe_ticker"])): int(row["permno"])
        for row in stock_identifier_map.to_dict("records")
    }
    universe_tickers = set(ticker_to_permno)

    period_mask = (
        (surface_panel["trade_date"] >= pd.Timestamp(TEXT_NEWS_START_DATE))
        & (surface_panel["trade_date"] <= pd.Timestamp(TEXT_NEWS_END_DATE))
    )
    restricted_surface_panel = surface_panel.loc[period_mask].copy()
    trade_dates = restricted_surface_panel["trade_date"]

    article_rows: list[dict[str, object]] = []
    rows_by_year: dict[str, int] = {}
    expanded_rows_by_year: dict[str, int] = {}
    for year in _news_years():
        year_path = _news_year_path(year)
        if not year_path.exists():
            raise RuntimeError(f"News archive file is missing: {year_path}")
        with lzma.open(year_path, "rt", encoding="utf-8") as handle:
            records = json.load(handle)
        rows_by_year[str(year)] = int(len(records))
        expanded_count = 0
        for record in records:
            if (record.get("language") or "").lower() != "en":
                continue
            title = (
                (record.get("title") or "").strip()
                or (record.get("title_rss") or "").strip()
                or (record.get("title_page") or "").strip()
                or (record.get("description") or "").strip()
            )
            raw_published = record.get("date_publish")
            if not title or not raw_published:
                continue
            mentioned = sorted(
                {
                    normalize_ticker(str(value))
                    for value in record.get("mentioned_companies", [])
                    if normalize_ticker(str(value)) in universe_tickers
                }
            )
            if not mentioned:
                continue
            published_utc = pd.to_datetime(raw_published, utc=True, errors="coerce")
            if pd.isna(published_utc):
                continue
            published_local = published_utc.tz_convert(TEXT_NEWS_TIMEZONE)
            sentiment = record.get("sentiment") or {}
            for ticker in mentioned:
                article_rows.append(
                    {
                        "permno": ticker_to_permno[ticker],
                        "universe_ticker": ticker,
                        "title": title,
                        "url": record.get("url") or "",
                        "publisher": record.get("news_outlet") or record.get("source_domain") or "",
                        "published_at_utc": published_utc.isoformat(),
                        "published_at_local": published_local.isoformat(),
                        "sentiment_neg": float(sentiment.get("negative", 0.0) or 0.0),
                        "sentiment_neu": float(sentiment.get("neutral", 0.0) or 0.0),
                        "sentiment_pos": float(sentiment.get("positive", 0.0) or 0.0),
                    }
                )
                expanded_count += 1
        expanded_rows_by_year[str(year)] = expanded_count

    if not article_rows:
        raise RuntimeError("No overlapping Yahoo Finance news articles were found for the configured universe.")

    articles = pd.DataFrame(article_rows)
    articles["published_at_local"] = pd.to_datetime(articles["published_at_local"], utc=True).dt.tz_convert(TEXT_NEWS_TIMEZONE)
    articles["published_at_utc"] = pd.to_datetime(articles["published_at_utc"], utc=True)
    articles = (
        articles.sort_values(["permno", "published_at_utc", "title"])
        .drop_duplicates(subset=["permno", "url", "published_at_utc", "title"])
        .reset_index(drop=True)
    )
    articles["claim_prob"] = _score_claim_probabilities(articles["title"].astype(str).tolist())
    finbert_scores = _score_finbert_titles(articles["title"].astype(str).tolist())
    articles = pd.concat([articles.reset_index(drop=True), finbert_scores.reset_index(drop=True)], axis=1)
    articles["assigned_trade_date"] = _assign_trade_dates(articles["published_at_local"], trade_dates)
    articles = articles.dropna(subset=["assigned_trade_date"]).reset_index(drop=True)
    articles["trade_date"] = pd.to_datetime(articles["assigned_trade_date"])
    articles["claim_count_above_05"] = (articles["claim_prob"] >= 0.5).astype(int)
    articles["event_claim_flag"] = (articles["claim_prob"] >= 0.5).astype(int)
    articles["sentiment_net"] = articles["sentiment_pos"] - articles["sentiment_neg"]
    articles["finbert_abs_net"] = articles["finbert_net"].abs()
    articles.to_parquet(TEXT_NEWS_ARTICLE_PANEL_PATH, index=False)

    daily_features = (
        articles.groupby(["permno", "universe_ticker", "trade_date"], as_index=False)
        .agg(
            news_count=("title", "count"),
            claim_prob_mean=("claim_prob", "mean"),
            claim_prob_max=("claim_prob", "max"),
            claim_count_above_05=("claim_count_above_05", "sum"),
            sentiment_neg_mean=("sentiment_neg", "mean"),
            sentiment_neu_mean=("sentiment_neu", "mean"),
            sentiment_pos_mean=("sentiment_pos", "mean"),
            sentiment_net_mean=("sentiment_net", "mean"),
            finbert_neg_mean=("finbert_neg", "mean"),
            finbert_neu_mean=("finbert_neu", "mean"),
            finbert_pos_mean=("finbert_pos", "mean"),
            finbert_net_mean=("finbert_net", "mean"),
        )
    )
    daily_features["news_count_log1p"] = np.log1p(daily_features["news_count"].astype(float))
    daily_features = daily_features[
        ["permno", "universe_ticker", "trade_date"] + ALL_DAILY_NEWS_FEATURES
    ].copy()
    daily_features.to_parquet(TEXT_NEWS_DAILY_FEATURE_PANEL_PATH, index=False)

    event_frame = (
        articles.sort_values(
            ["permno", "trade_date", "claim_prob", "finbert_abs_net", "published_at_utc"],
            ascending=[True, True, False, False, True],
        )
        .groupby(["permno", "universe_ticker", "trade_date"], as_index=False)
        .first()
    )
    event_frame = event_frame.rename(
        columns={
            "claim_prob": "event_claim_prob",
            "event_claim_flag": "event_claim_flag",
            "finbert_neg": "event_finbert_neg",
            "finbert_neu": "event_finbert_neu",
            "finbert_pos": "event_finbert_pos",
            "finbert_net": "event_finbert_net",
            "sentiment_neg": "event_sentiment_neg",
            "sentiment_neu": "event_sentiment_neu",
            "sentiment_pos": "event_sentiment_pos",
            "sentiment_net": "event_sentiment_net",
            "title": "event_title",
            "publisher": "event_publisher",
            "url": "event_url",
        }
    )
    event_columns = [
        "permno",
        "universe_ticker",
        "trade_date",
        "event_claim_prob",
        "event_claim_flag",
        "event_finbert_neg",
        "event_finbert_neu",
        "event_finbert_pos",
        "event_finbert_net",
        "event_sentiment_neg",
        "event_sentiment_neu",
        "event_sentiment_pos",
        "event_sentiment_net",
        "event_title",
        "event_publisher",
        "event_url",
    ]
    event_frame = event_frame[event_columns].copy()

    covered_tickers = sorted(daily_features["universe_ticker"].astype(str).unique())
    extension_panel = restricted_surface_panel[restricted_surface_panel["universe_ticker"].isin(covered_tickers)].copy()
    extension_panel = extension_panel.merge(
        daily_features.drop(columns=["universe_ticker"]),
        on=["permno", "trade_date"],
        how="left",
    )
    for feature in ALL_DAILY_NEWS_FEATURES:
        extension_panel[feature] = pd.to_numeric(extension_panel[feature], errors="coerce").fillna(0.0)
    extension_panel.to_parquet(TEXT_NEWS_EXTENSION_PANEL_PATH, index=False)

    event_panel = extension_panel[extension_panel["news_count"] > 0].copy()
    event_panel = event_panel.merge(
        event_frame.drop(columns=["universe_ticker"]),
        on=["permno", "trade_date"],
        how="left",
    )
    event_panel.to_parquet(TEXT_NEWS_EVENT_PANEL_PATH, index=False)

    summary = {
        "years_used": _news_years(),
        "news_rows_by_year": rows_by_year,
        "expanded_article_rows_by_year": expanded_rows_by_year,
        "article_ticker_rows": int(len(articles)),
        "covered_ticker_count": int(len(covered_tickers)),
        "covered_tickers": covered_tickers,
        "daily_feature_rows": int(len(daily_features)),
        "extension_panel_rows": int(len(extension_panel)),
        "event_panel_rows": int(len(event_panel)),
        "news_active_rows": int((extension_panel["news_count"] > 0).sum()),
        "empty_news_rows": int((extension_panel["news_count"] == 0).sum()),
        "date_range": {
            "min_date": str(extension_panel["trade_date"].min().date()),
            "max_date": str(extension_panel["trade_date"].max().date()),
        },
        "split_sizes": {
            "train": int(len(extension_panel[extension_panel["trade_date"] <= pd.Timestamp(TRAIN_END_DATE)])),
            "validation": int(
                len(
                    extension_panel[
                        (extension_panel["trade_date"] > pd.Timestamp(TRAIN_END_DATE))
                        & (extension_panel["trade_date"] <= pd.Timestamp(VALIDATION_END_DATE))
                    ]
                )
            ),
            "test": int(len(extension_panel[extension_panel["trade_date"] > pd.Timestamp(VALIDATION_END_DATE)])),
        },
        "event_split_sizes": {
            "train": int(len(event_panel[event_panel["trade_date"] <= pd.Timestamp(TRAIN_END_DATE)])),
            "validation": int(
                len(
                    event_panel[
                        (event_panel["trade_date"] > pd.Timestamp(TRAIN_END_DATE))
                        & (event_panel["trade_date"] <= pd.Timestamp(VALIDATION_END_DATE))
                    ]
                )
            ),
            "test": int(len(event_panel[event_panel["trade_date"] > pd.Timestamp(VALIDATION_END_DATE)])),
        },
        "feature_coverage_before_fill": {
            feature: float(
                extension_panel.loc[extension_panel["news_count"] > 0, feature].notna().mean()
                if (extension_panel["news_count"] > 0).any()
                else 0.0
            )
            for feature in ALL_DAILY_NEWS_FEATURES
        },
        "finbert_device": str(_device()),
    }
    write_json(BUILD_TEXT_NEWS_PANEL_SUMMARY_PATH, summary)
    return summary


def _load_text_news_extension_panel() -> pd.DataFrame:
    if not TEXT_NEWS_EXTENSION_PANEL_PATH.exists():
        build_text_news_panel()
    frame = pd.read_parquet(TEXT_NEWS_EXTENSION_PANEL_PATH)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    return frame


def _load_text_news_event_panel() -> pd.DataFrame:
    if not TEXT_NEWS_EVENT_PANEL_PATH.exists():
        build_text_news_panel()
    frame = pd.read_parquet(TEXT_NEWS_EVENT_PANEL_PATH)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    return frame


def _split_by_date(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    train = frame[frame["trade_date"] <= pd.Timestamp(TRAIN_END_DATE)].reset_index(drop=True)
    validation = frame[
        (frame["trade_date"] > pd.Timestamp(TRAIN_END_DATE)) & (frame["trade_date"] <= pd.Timestamp(VALIDATION_END_DATE))
    ].reset_index(drop=True)
    test = frame[
        (frame["trade_date"] > pd.Timestamp(VALIDATION_END_DATE)) & (frame["trade_date"] <= pd.Timestamp(TEXT_NEWS_END_DATE))
    ].reset_index(drop=True)
    if train.empty or validation.empty or test.empty:
        raise RuntimeError("One of the chronological text-news splits is empty.")
    return {"train": train, "validation": validation, "test": test}


def _fit_best_logreg(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: list[int],
) -> tuple[Pipeline, float, dict[str, object]]:
    best_pipeline = None
    best_c = None
    best_metrics = None
    best_score = None
    for c_value in CLASSICAL_C_GRID:
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=5000, C=c_value, random_state=SEED)),
            ]
        )
        pipeline.fit(X_train, y_train)
        validation_prob = pipeline.predict_proba(X_validation)[:, 1].tolist()
        validation_pred = pipeline.predict(X_validation).tolist()
        metrics = compute_metrics(y_validation, validation_pred, validation_prob, PART1_LABEL_NAMES)
        score = (float(metrics["macro_f1"]), float(metrics["balanced_accuracy"]), float(metrics["accuracy"]))
        if best_score is None or score > best_score:
            best_pipeline = pipeline
            best_c = c_value
            best_metrics = metrics
            best_score = score
    return best_pipeline, float(best_c), best_metrics


def _train_single_model(
    splits: dict[str, pd.DataFrame],
    feature_columns: list[str],
    model_name: str,
) -> tuple[dict[str, object], Pipeline]:
    X_train = splits["train"][feature_columns].to_numpy()
    y_train = splits["train"]["high_rv_regime"].astype(int).to_numpy()
    X_validation = splits["validation"][feature_columns].to_numpy()
    y_validation = splits["validation"]["high_rv_regime"].astype(int).tolist()
    X_test = splits["test"][feature_columns].to_numpy()
    y_test = splits["test"]["high_rv_regime"].astype(int).tolist()

    best_pipeline, best_c, validation_metrics = _fit_best_logreg(X_train, y_train, X_validation, y_validation)
    test_prob = best_pipeline.predict_proba(X_test)[:, 1].tolist()
    test_pred = best_pipeline.predict(X_test).tolist()

    summary = {
        "model": model_name,
        "best_c": best_c,
        "feature_columns": feature_columns,
        "validation": validation_metrics,
        "test": compute_metrics(y_test, test_pred, test_prob, PART1_LABEL_NAMES),
    }
    return summary, best_pipeline


def _train_model_group(
    panel: pd.DataFrame,
    model_specs: dict[str, list[str]],
    panel_type: str,
) -> tuple[dict[str, dict[str, object]], dict[str, Pipeline], list[dict[str, object]], list[dict[str, object]]]:
    panel = panel.dropna(subset=CORE_FEATURES + ["high_rv_regime"]).reset_index(drop=True)
    splits = _split_by_date(panel)

    model_summaries: dict[str, dict[str, object]] = {}
    fitted_models: dict[str, Pipeline] = {}
    metrics_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []

    for model_name, feature_columns in model_specs.items():
        summary, pipeline = _train_single_model(splits, feature_columns, model_name)
        model_summaries[model_name] = summary
        fitted_models[model_name] = pipeline
        for split_name in ("validation", "test"):
            split_metrics = summary[split_name]
            metrics_rows.append(
                {
                    "panel_type": panel_type,
                    "model": model_name,
                    "split": split_name,
                    "accuracy": split_metrics["accuracy"],
                    "macro_f1": split_metrics["macro_f1"],
                    "balanced_accuracy": split_metrics["balanced_accuracy"],
                    "auroc": split_metrics["auroc"],
                    "pr_auc": split_metrics["pr_auc"],
                }
            )

    for split_name, split_frame in (("validation", splits["validation"]), ("test", splits["test"])):
        split_reset = split_frame.reset_index(drop=True)
        probabilities = {
            model_name: fitted_models[model_name].predict_proba(split_reset[feature_columns].to_numpy())[:, 1]
            for model_name, feature_columns in model_specs.items()
        }
        predictions = {
            model_name: fitted_models[model_name].predict(split_reset[model_specs[model_name]].to_numpy())
            for model_name in model_specs
        }
        for idx, row in split_reset.iterrows():
            entry = {
                "panel_type": panel_type,
                "permno": int(row["permno"]),
                "universe_ticker": row["universe_ticker"],
                "trade_date": row["trade_date"].date().isoformat(),
                "split": split_name,
                "label": int(row["high_rv_regime"]),
                "news_count": float(row.get("news_count", 0.0)),
            }
            for model_name in model_specs:
                entry[f"{model_name}_prob"] = float(probabilities[model_name][idx])
                entry[f"{model_name}_pred"] = int(predictions[model_name][idx])
            prediction_rows.append(entry)

    return model_summaries, fitted_models, metrics_rows, prediction_rows


def _save_model_bundle(path: Path, model_name: str, pipeline: Pipeline, feature_columns: list[str]) -> None:
    joblib.dump({"model_name": model_name, "pipeline": pipeline, "feature_columns": feature_columns}, path)


def train_text_news_extension() -> dict[str, object]:
    full_panel = _load_text_news_extension_panel()
    event_panel = _load_text_news_event_panel()

    full_model_specs = {
        "option_only_news_common": OPTION_FEATURES,
        "all_features_core_news_common": CORE_FEATURES,
        "news_only_logreg": TEXT_NEWS_FEATURES,
        "all_features_plus_sentiment_logreg": CORE_FEATURES + DATASET_SENTIMENT_NEWS_FEATURES,
        "all_features_plus_claim_sentiment_logreg": CORE_FEATURES + TEXT_NEWS_FEATURES,
        "all_features_plus_finbert_logreg": CORE_FEATURES + DAILY_FINBERT_NEWS_FEATURES,
        "all_features_plus_claim_finbert_logreg": CORE_FEATURES + DAILY_CLAIM_FINBERT_NEWS_FEATURES,
    }
    event_model_specs = {
        "all_features_core_event_common": CORE_FEATURES,
        "event_only_logreg": EVENT_CLAIM_FINBERT_FEATURES,
        "all_features_plus_event_finbert_logreg": CORE_FEATURES + EVENT_FINBERT_FEATURES,
        "all_features_plus_event_claim_finbert_logreg": CORE_FEATURES + EVENT_CLAIM_FINBERT_FEATURES,
    }

    full_models, full_pipelines, full_rows, full_predictions = _train_model_group(full_panel, full_model_specs, "full_panel")
    event_models, event_pipelines, event_rows, event_predictions = _train_model_group(event_panel, event_model_specs, "event_active_panel")

    _save_model_bundle(TEXT_NEWS_NEWS_ONLY_MODEL_PATH, "news_only_logreg", full_pipelines["news_only_logreg"], TEXT_NEWS_FEATURES)
    _save_model_bundle(
        TEXT_NEWS_SENTIMENT_MODEL_PATH,
        "all_features_plus_sentiment_logreg",
        full_pipelines["all_features_plus_sentiment_logreg"],
        CORE_FEATURES + DATASET_SENTIMENT_NEWS_FEATURES,
    )
    _save_model_bundle(
        TEXT_NEWS_CLAIM_SENTIMENT_MODEL_PATH,
        "all_features_plus_claim_sentiment_logreg",
        full_pipelines["all_features_plus_claim_sentiment_logreg"],
        CORE_FEATURES + TEXT_NEWS_FEATURES,
    )
    _save_model_bundle(
        TEXT_NEWS_FINBERT_MODEL_PATH,
        "all_features_plus_finbert_logreg",
        full_pipelines["all_features_plus_finbert_logreg"],
        CORE_FEATURES + DAILY_FINBERT_NEWS_FEATURES,
    )
    _save_model_bundle(
        TEXT_NEWS_CLAIM_FINBERT_MODEL_PATH,
        "all_features_plus_claim_finbert_logreg",
        full_pipelines["all_features_plus_claim_finbert_logreg"],
        CORE_FEATURES + DAILY_CLAIM_FINBERT_NEWS_FEATURES,
    )
    _save_model_bundle(
        TEXT_NEWS_EVENT_FINBERT_MODEL_PATH,
        "all_features_plus_event_finbert_logreg",
        event_pipelines["all_features_plus_event_finbert_logreg"],
        CORE_FEATURES + EVENT_FINBERT_FEATURES,
    )
    _save_model_bundle(
        TEXT_NEWS_EVENT_CLAIM_FINBERT_MODEL_PATH,
        "all_features_plus_event_claim_finbert_logreg",
        event_pipelines["all_features_plus_event_claim_finbert_logreg"],
        CORE_FEATURES + EVENT_CLAIM_FINBERT_FEATURES,
    )

    full_test = _split_by_date(full_panel)["test"].reset_index(drop=True)
    event_test = _split_by_date(event_panel)["test"].reset_index(drop=True)
    y_full_test = full_test["high_rv_regime"].astype(int).tolist()
    y_event_test = event_test["high_rv_regime"].astype(int).tolist()

    save_confusion_matrix_figure(
        y_full_test,
        full_pipelines["all_features_core_news_common"].predict(full_test[CORE_FEATURES].to_numpy()).tolist(),
        PART1_LABEL_NAMES,
        TEXT_NEWS_CORE_FIGURE,
        "Text News Core (Test)",
    )
    save_confusion_matrix_figure(
        y_full_test,
        full_pipelines["news_only_logreg"].predict(full_test[TEXT_NEWS_FEATURES].to_numpy()).tolist(),
        PART1_LABEL_NAMES,
        TEXT_NEWS_NEWS_ONLY_FIGURE,
        "Text News News-only (Test)",
    )
    save_confusion_matrix_figure(
        y_full_test,
        full_pipelines["all_features_plus_sentiment_logreg"].predict(full_test[CORE_FEATURES + DATASET_SENTIMENT_NEWS_FEATURES].to_numpy()).tolist(),
        PART1_LABEL_NAMES,
        TEXT_NEWS_SENTIMENT_FIGURE,
        "Text News Core + Dataset Sentiment (Test)",
    )
    save_confusion_matrix_figure(
        y_full_test,
        full_pipelines["all_features_plus_claim_sentiment_logreg"].predict(full_test[CORE_FEATURES + TEXT_NEWS_FEATURES].to_numpy()).tolist(),
        PART1_LABEL_NAMES,
        TEXT_NEWS_CLAIM_SENTIMENT_FIGURE,
        "Text News Core + Claim + Dataset Sentiment (Test)",
    )
    save_confusion_matrix_figure(
        y_full_test,
        full_pipelines["all_features_plus_finbert_logreg"].predict(full_test[CORE_FEATURES + DAILY_FINBERT_NEWS_FEATURES].to_numpy()).tolist(),
        PART1_LABEL_NAMES,
        TEXT_NEWS_FINBERT_FIGURE,
        "Text News Core + FinBERT (Test)",
    )
    save_confusion_matrix_figure(
        y_full_test,
        full_pipelines["all_features_plus_claim_finbert_logreg"].predict(full_test[CORE_FEATURES + DAILY_CLAIM_FINBERT_NEWS_FEATURES].to_numpy()).tolist(),
        PART1_LABEL_NAMES,
        TEXT_NEWS_CLAIM_FINBERT_FIGURE,
        "Text News Core + Claim + FinBERT (Test)",
    )
    save_confusion_matrix_figure(
        y_event_test,
        event_pipelines["all_features_core_event_common"].predict(event_test[CORE_FEATURES].to_numpy()).tolist(),
        PART1_LABEL_NAMES,
        TEXT_NEWS_EVENT_CORE_FIGURE,
        "Text Event Core (Test)",
    )
    save_confusion_matrix_figure(
        y_event_test,
        event_pipelines["all_features_plus_event_finbert_logreg"].predict(event_test[CORE_FEATURES + EVENT_FINBERT_FEATURES].to_numpy()).tolist(),
        PART1_LABEL_NAMES,
        TEXT_NEWS_EVENT_FINBERT_FIGURE,
        "Text Event Core + FinBERT (Test)",
    )
    save_confusion_matrix_figure(
        y_event_test,
        event_pipelines["all_features_plus_event_claim_finbert_logreg"].predict(event_test[CORE_FEATURES + EVENT_CLAIM_FINBERT_FEATURES].to_numpy()).tolist(),
        PART1_LABEL_NAMES,
        TEXT_NEWS_EVENT_CLAIM_FINBERT_FIGURE,
        "Text Event Core + Claim + FinBERT (Test)",
    )

    full_coefficients = {
        model_name: [
            {"feature": feature, "coefficient": float(value)}
            for feature, value in zip(full_model_specs[model_name], pipeline.named_steps["classifier"].coef_[0])
        ]
        for model_name, pipeline in full_pipelines.items()
    }
    event_coefficients = {
        model_name: [
            {"feature": feature, "coefficient": float(value)}
            for feature, value in zip(event_model_specs[model_name], pipeline.named_steps["classifier"].coef_[0])
        ]
        for model_name, pipeline in event_pipelines.items()
    }

    summary = {
        "period": {"start_date": TEXT_NEWS_START_DATE, "end_date": TEXT_NEWS_END_DATE},
        "covered_ticker_count": int(full_panel["universe_ticker"].nunique()),
        "full_panel_rows": int(len(full_panel)),
        "event_panel_rows": int(len(event_panel)),
        "full_panel_split_sizes": {split_name: int(len(split_frame)) for split_name, split_frame in _split_by_date(full_panel).items()},
        "event_panel_split_sizes": {split_name: int(len(split_frame)) for split_name, split_frame in _split_by_date(event_panel).items()},
        "full_panel_models": full_models,
        "event_panel_models": event_models,
        "full_panel_coefficients": full_coefficients,
        "event_panel_coefficients": event_coefficients,
        "rows": full_rows + event_rows,
    }
    write_rows_csv(TEXT_NEWS_EXTENSION_METRICS_CSV, full_rows + event_rows)
    write_rows_csv(TEXT_NEWS_EXTENSION_PREDICTIONS_CSV, full_predictions + event_predictions)
    write_json(TEXT_NEWS_EXTENSION_METRICS_JSON, summary)
    return summary
