from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .claim_signal import score_claim_probabilities
from .config import (
    AMPLITUDE_CSV,
    AMPLITUDE_LABEL_NAMES,
    AMPLITUDE_MARKET_FIGURE_PATH,
    AMPLITUDE_MARKET_MODEL_PATH,
    AMPLITUDE_PREDICTIONS_CSV,
    AMPLITUDE_QUANTILE,
    AMPLITUDE_STRUCTURED_FIGURE_PATH,
    AMPLITUDE_STRUCTURED_MODEL_PATH,
    AMPLITUDE_SUMMARY_PATH,
    CLASSICAL_C_GRID,
    CLASSICAL_MIN_DF,
    DIRECTION_LABEL_NAMES,
    ENRICHED_DAILY_DATASET_PARQUET,
    HEADLINE_CLAIM_SCORES_PARQUET,
    HEADLINE_FINBERT_SCORES_PARQUET,
    MATERIALITY_SUMMARY_PATH,
    SEED,
    STAGE2_CLAIM_AWARE_FIGURE_PATH,
    STAGE2_CLAIM_AWARE_MODEL_PATH,
    STAGE2_DIRECTION_CSV,
    STAGE2_DIRECTION_PREDICTIONS_CSV,
    STAGE2_DIRECTION_SUMMARY_PATH,
    STAGE2_FULL_TEXT_FIGURE_PATH,
    STAGE2_FULL_TEXT_MODEL_PATH,
    STAGE2_MARKET_FIGURE_PATH,
    STAGE2_MARKET_MODEL_PATH,
    STAGE3_CLAIM_SENTIMENT_MODEL_PATH,
    STAGE3_SENTIMENT_CSV,
    STAGE3_SENTIMENT_FIGURE_PATH,
    STAGE3_SENTIMENT_PREDICTIONS_CSV,
    STAGE3_SENTIMENT_SUMMARY_PATH,
    ensure_project_dirs,
    set_global_seed,
)
from .data_pipeline import (
    _assign_headlines_to_trading_days,
    _load_collected_headlines,
    _load_collected_prices,
    compute_chronological_splits,
    load_daily_dataset,
)
from .evaluation import compute_metrics, save_confusion_matrix_figure, write_json, write_rows_csv
from .finbert_signal import score_finbert_headlines
from .text_utils import tokenize


PRICE_FEATURES = ["ret_1d", "ret_5d", "vol_5d"]
CLAIM_FEATURES = ["claim_prob_mean", "claim_prob_max", "claim_count_above_05"]
SENTIMENT_FEATURES = ["finbert_pos_mean", "finbert_neg_mean", "finbert_neu_mean", "finbert_sentiment_mean"]


def _score_tuple(metrics: dict[str, object], model_name: str) -> tuple[float, float, str]:
    return (float(metrics["macro_f1"]), float(metrics["accuracy"]), model_name)


def _build_text_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        tokenizer=tokenize,
        preprocessor=None,
        lowercase=False,
        token_pattern=None,
        min_df=CLASSICAL_MIN_DF,
        ngram_range=(1, 1),
    )


def run_stage1_materiality() -> dict[str, object]:
    ensure_project_dirs()
    dataset = load_daily_dataset()
    headlines = _load_collected_headlines().sort_values("published_at_local").reset_index(drop=True)
    probabilities = score_claim_probabilities(headlines["title"].tolist())
    headline_scores = headlines[["headline_id", "title", "published_at_local"]].copy()
    headline_scores["claim_prob"] = probabilities
    headline_scores.to_parquet(HEADLINE_CLAIM_SCORES_PARQUET, index=False)

    assigned = _assign_headlines_to_trading_days(
        headlines[["headline_id", "published_at_local"]].merge(
            headline_scores[["headline_id", "claim_prob"]],
            on="headline_id",
            how="left",
        ),
        _load_collected_prices()["trade_date"],
    )
    aggregated = assigned.groupby("assigned_trade_date").agg(
        claim_prob_mean=("claim_prob", "mean"),
        claim_prob_max=("claim_prob", "max"),
        claim_count_above_05=("claim_prob", lambda values: int((pd.Series(values) >= 0.5).sum())),
    ).reset_index().rename(columns={"assigned_trade_date": "trade_date"})

    enriched = dataset.merge(aggregated, on="trade_date", how="left")
    for column in CLAIM_FEATURES:
        enriched[column] = enriched[column].fillna(0.0)
    enriched.to_parquet(ENRICHED_DAILY_DATASET_PARQUET, index=False)

    summary = {
        "headline_rows": int(len(headline_scores)),
        "headline_claim_prob_mean": float(headline_scores["claim_prob"].mean()),
        "headline_claim_prob_max": float(headline_scores["claim_prob"].max()),
        "daily_rows": int(len(enriched)),
        "daily_rows_with_claim_signal": int((enriched["claim_count_above_05"] > 0).sum()),
    }
    write_json(MATERIALITY_SUMMARY_PATH, summary)
    return summary


def _load_enriched_daily_dataset() -> pd.DataFrame:
    if not ENRICHED_DAILY_DATASET_PARQUET.exists():
        run_stage1_materiality()
    frame = pd.read_parquet(ENRICHED_DAILY_DATASET_PARQUET)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    return frame


def _train_market_only(split_frames: dict[str, pd.DataFrame]):
    X_train = split_frames["train"][PRICE_FEATURES].to_numpy()
    y_train = split_frames["train"]["direction_label"].astype(int).tolist()
    X_validation = split_frames["validation"][PRICE_FEATURES].to_numpy()
    y_validation = split_frames["validation"]["direction_label"].astype(int).tolist()
    X_test = split_frames["test"][PRICE_FEATURES].to_numpy()
    y_test = split_frames["test"]["direction_label"].astype(int).tolist()

    best_pipeline = None
    best_c = None
    best_metrics = None
    for c_value in CLASSICAL_C_GRID:
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=2000, C=c_value, random_state=SEED)),
            ]
        )
        pipeline.fit(X_train, y_train)
        metrics = compute_metrics(y_validation, pipeline.predict(X_validation).tolist(), DIRECTION_LABEL_NAMES)
        if best_metrics is None or _score_tuple(metrics, "market_only") > _score_tuple(best_metrics, "market_only"):
            best_pipeline = pipeline
            best_c = c_value
            best_metrics = metrics

    validation_predictions = best_pipeline.predict(X_validation).tolist()
    test_predictions = best_pipeline.predict(X_test).tolist()
    bundle = {"model_name": "market_only", "pipeline": best_pipeline, "feature_columns": PRICE_FEATURES}
    summary = {
        "model": "market_only",
        "best_c": best_c,
        "validation": compute_metrics(y_validation, validation_predictions, DIRECTION_LABEL_NAMES),
        "test": compute_metrics(y_test, test_predictions, DIRECTION_LABEL_NAMES),
    }
    return bundle, summary


def _train_text_classifier(model_name: str, split_frames: dict[str, pd.DataFrame], scalar_columns: list[str]):
    train_texts = split_frames["train"]["daily_text"].tolist()
    validation_texts = split_frames["validation"]["daily_text"].tolist()
    test_texts = split_frames["test"]["daily_text"].tolist()
    y_train = split_frames["train"]["direction_label"].astype(int).tolist()
    y_validation = split_frames["validation"]["direction_label"].astype(int).tolist()
    y_test = split_frames["test"]["direction_label"].astype(int).tolist()

    vectorizer = _build_text_vectorizer()
    X_train_text = vectorizer.fit_transform(train_texts)
    X_validation_text = vectorizer.transform(validation_texts)
    X_test_text = vectorizer.transform(test_texts)
    if scalar_columns:
        X_train = sparse.hstack([X_train_text, sparse.csr_matrix(split_frames["train"][scalar_columns].to_numpy())], format="csr")
        X_validation = sparse.hstack([X_validation_text, sparse.csr_matrix(split_frames["validation"][scalar_columns].to_numpy())], format="csr")
        X_test = sparse.hstack([X_test_text, sparse.csr_matrix(split_frames["test"][scalar_columns].to_numpy())], format="csr")
    else:
        X_train, X_validation, X_test = X_train_text, X_validation_text, X_test_text

    best_classifier = None
    best_c = None
    best_metrics = None
    for c_value in CLASSICAL_C_GRID:
        classifier = LogisticRegression(max_iter=2000, C=c_value, random_state=SEED)
        classifier.fit(X_train, y_train)
        metrics = compute_metrics(y_validation, classifier.predict(X_validation).tolist(), DIRECTION_LABEL_NAMES)
        if best_metrics is None or _score_tuple(metrics, model_name) > _score_tuple(best_metrics, model_name):
            best_classifier = classifier
            best_c = c_value
            best_metrics = metrics

    validation_predictions = best_classifier.predict(X_validation).tolist()
    test_predictions = best_classifier.predict(X_test).tolist()
    bundle = {
        "model_name": model_name,
        "vectorizer": vectorizer,
        "classifier": best_classifier,
        "scalar_columns": scalar_columns,
    }
    summary = {
        "model": model_name,
        "best_c": best_c,
        "feature_count": int(X_train.shape[1]),
        "validation": compute_metrics(y_validation, validation_predictions, DIRECTION_LABEL_NAMES),
        "test": compute_metrics(y_test, test_predictions, DIRECTION_LABEL_NAMES),
    }
    return bundle, summary


def run_stage2_direction() -> dict[str, object]:
    ensure_project_dirs()
    set_global_seed(SEED)
    split_frames = compute_chronological_splits(_load_enriched_daily_dataset())

    market_bundle, market_summary = _train_market_only(split_frames)
    full_bundle, full_summary = _train_text_classifier("full_text", split_frames, scalar_columns=[])
    claim_bundle, claim_summary = _train_text_classifier("claim_aware", split_frames, scalar_columns=CLAIM_FEATURES)

    joblib.dump(market_bundle, STAGE2_MARKET_MODEL_PATH)
    joblib.dump(full_bundle, STAGE2_FULL_TEXT_MODEL_PATH)
    joblib.dump(claim_bundle, STAGE2_CLAIM_AWARE_MODEL_PATH)

    save_confusion_matrix_figure(
        split_frames["test"]["direction_label"].astype(int).tolist(),
        market_bundle["pipeline"].predict(split_frames["test"][PRICE_FEATURES].to_numpy()).tolist(),
        DIRECTION_LABEL_NAMES,
        STAGE2_MARKET_FIGURE_PATH,
        "TSLA Stage 2 Market-only (Test)",
    )
    X_full_test = full_bundle["vectorizer"].transform(split_frames["test"]["daily_text"].tolist())
    save_confusion_matrix_figure(
        split_frames["test"]["direction_label"].astype(int).tolist(),
        full_bundle["classifier"].predict(X_full_test).tolist(),
        DIRECTION_LABEL_NAMES,
        STAGE2_FULL_TEXT_FIGURE_PATH,
        "TSLA Stage 2 Full-text (Test)",
    )
    X_claim_test = sparse.hstack([X_full_test, sparse.csr_matrix(split_frames["test"][CLAIM_FEATURES].to_numpy())], format="csr")
    save_confusion_matrix_figure(
        split_frames["test"]["direction_label"].astype(int).tolist(),
        claim_bundle["classifier"].predict(X_claim_test).tolist(),
        DIRECTION_LABEL_NAMES,
        STAGE2_CLAIM_AWARE_FIGURE_PATH,
        "TSLA Stage 2 Claim-aware (Test)",
    )

    rows = []
    prediction_rows = []
    for summary in (market_summary, full_summary, claim_summary):
        for split_name in ("validation", "test"):
            rows.append(
                {
                    "model": summary["model"],
                    "split": split_name,
                    "accuracy": summary[split_name]["accuracy"],
                    "macro_f1": summary[split_name]["macro_f1"],
                }
            )
    for split_name, frame in split_frames.items():
        market_prob = market_bundle["pipeline"].predict_proba(frame[PRICE_FEATURES].to_numpy())[:, 1]
        X_full = full_bundle["vectorizer"].transform(frame["daily_text"].tolist())
        full_prob = full_bundle["classifier"].predict_proba(X_full)[:, 1]
        X_claim = sparse.hstack([X_full, sparse.csr_matrix(frame[CLAIM_FEATURES].to_numpy())], format="csr")
        claim_prob = claim_bundle["classifier"].predict_proba(X_claim)[:, 1]
        for idx, row in frame.reset_index(drop=True).iterrows():
            prediction_rows.append(
                {
                    "trade_date": row["trade_date"].date().isoformat(),
                    "split": split_name,
                    "headline_count": int(row["headline_count"]),
                    "direction_label": int(row["direction_label"]),
                    "market_only_prob": float(market_prob[idx]),
                    "full_text_prob": float(full_prob[idx]),
                    "claim_aware_prob": float(claim_prob[idx]),
                }
            )
    write_rows_csv(STAGE2_DIRECTION_CSV, rows)
    write_rows_csv(STAGE2_DIRECTION_PREDICTIONS_CSV, prediction_rows)

    summary = {
        "split_sizes": {split_name: int(len(frame)) for split_name, frame in split_frames.items()},
        "models": {"market_only": market_summary, "full_text": full_summary, "claim_aware": claim_summary},
        "rows": rows,
    }
    write_json(STAGE2_DIRECTION_SUMMARY_PATH, summary)
    return summary


def run_stage3_sentiment() -> dict[str, object]:
    ensure_project_dirs()
    set_global_seed(SEED)
    dataset = _load_enriched_daily_dataset()
    headlines = _load_collected_headlines().sort_values("published_at_local").reset_index(drop=True)
    finbert_scores = score_finbert_headlines(headlines["title"].tolist())
    finbert_scores.insert(0, "headline_id", headlines["headline_id"].tolist())
    finbert_scores.to_parquet(HEADLINE_FINBERT_SCORES_PARQUET, index=False)

    assigned = _assign_headlines_to_trading_days(
        headlines[["headline_id", "published_at_local"]].merge(finbert_scores, on="headline_id", how="left"),
        _load_collected_prices()["trade_date"],
    )
    aggregated = assigned.groupby("assigned_trade_date").agg(
        finbert_pos_mean=("finbert_pos", "mean"),
        finbert_neg_mean=("finbert_neg", "mean"),
        finbert_neu_mean=("finbert_neu", "mean"),
        finbert_sentiment_mean=("finbert_sentiment", "mean"),
    ).reset_index().rename(columns={"assigned_trade_date": "trade_date"})

    enriched = dataset.merge(aggregated, on="trade_date", how="left")
    for column in SENTIMENT_FEATURES:
        enriched[column] = enriched[column].fillna(0.0)
    enriched.to_parquet(ENRICHED_DAILY_DATASET_PARQUET, index=False)

    split_frames = compute_chronological_splits(enriched)
    sentiment_bundle, sentiment_summary = _train_text_classifier(
        "claim_sentiment_aware",
        split_frames,
        scalar_columns=CLAIM_FEATURES + SENTIMENT_FEATURES,
    )
    joblib.dump(sentiment_bundle, STAGE3_CLAIM_SENTIMENT_MODEL_PATH)

    X_test = sentiment_bundle["vectorizer"].transform(split_frames["test"]["daily_text"].tolist())
    X_test = sparse.hstack([X_test, sparse.csr_matrix(split_frames["test"][CLAIM_FEATURES + SENTIMENT_FEATURES].to_numpy())], format="csr")
    save_confusion_matrix_figure(
        split_frames["test"]["direction_label"].astype(int).tolist(),
        sentiment_bundle["classifier"].predict(X_test).tolist(),
        DIRECTION_LABEL_NAMES,
        STAGE3_SENTIMENT_FIGURE_PATH,
        "TSLA Stage 3 Claim+Sentiment Aware (Test)",
    )

    rows = [
        {
            "model": sentiment_summary["model"],
            "split": split_name,
            "accuracy": sentiment_summary[split_name]["accuracy"],
            "macro_f1": sentiment_summary[split_name]["macro_f1"],
        }
        for split_name in ("validation", "test")
    ]
    prediction_rows = []
    for split_name, frame in split_frames.items():
        if split_name == "train":
            continue
        X = sentiment_bundle["vectorizer"].transform(frame["daily_text"].tolist())
        X = sparse.hstack([X, sparse.csr_matrix(frame[CLAIM_FEATURES + SENTIMENT_FEATURES].to_numpy())], format="csr")
        probs = sentiment_bundle["classifier"].predict_proba(X)[:, 1]
        preds = sentiment_bundle["classifier"].predict(X)
        for idx, row in frame.reset_index(drop=True).iterrows():
            prediction_rows.append(
                {
                    "trade_date": row["trade_date"].date().isoformat(),
                    "split": split_name,
                    "direction_label": int(row["direction_label"]),
                    "claim_sentiment_aware_prob": float(probs[idx]),
                    "claim_sentiment_aware_pred": int(preds[idx]),
                }
            )
    write_rows_csv(STAGE3_SENTIMENT_CSV, rows)
    write_rows_csv(STAGE3_SENTIMENT_PREDICTIONS_CSV, prediction_rows)

    summary = {
        "headline_rows": int(len(finbert_scores)),
        "daily_rows_with_sentiment_signal": int((enriched["headline_count"] > 0).sum()),
        "model": sentiment_summary,
        "rows": rows,
    }
    write_json(STAGE3_SENTIMENT_SUMMARY_PATH, summary)
    return summary


def run_stage3_amplitude() -> dict[str, object]:
    ensure_project_dirs()
    set_global_seed(SEED)
    dataset = _load_enriched_daily_dataset()
    if not all(column in dataset.columns for column in SENTIMENT_FEATURES):
        run_stage3_sentiment()
        dataset = _load_enriched_daily_dataset()
    split_frames = compute_chronological_splits(dataset)

    price_ratios = (dataset["close_t_plus_1"] / dataset["close_t"]).replace([np.inf, -np.inf], np.nan).dropna()
    if not bool(((price_ratios > 0.5) & (price_ratios < 2.0)).all()):
        summary = {"skipped": True, "reason": "price_sanity_gate_failed"}
        write_json(AMPLITUDE_SUMMARY_PATH, summary)
        return summary

    threshold = float(split_frames["train"]["next_day_return"].abs().quantile(AMPLITUDE_QUANTILE))
    for frame in split_frames.values():
        frame["amplitude_label"] = (frame["next_day_return"].abs() >= threshold).astype(int)
    if any(int(frame["amplitude_label"].sum()) < 20 for frame in split_frames.values()):
        summary = {"skipped": True, "reason": "insufficient_large_move_rows", "threshold": threshold}
        write_json(AMPLITUDE_SUMMARY_PATH, summary)
        return summary

    def train_structured(model_name: str, feature_columns: list[str]):
        X_train = split_frames["train"][feature_columns].to_numpy()
        y_train = split_frames["train"]["amplitude_label"].astype(int).tolist()
        X_validation = split_frames["validation"][feature_columns].to_numpy()
        y_validation = split_frames["validation"]["amplitude_label"].astype(int).tolist()
        X_test = split_frames["test"][feature_columns].to_numpy()
        y_test = split_frames["test"]["amplitude_label"].astype(int).tolist()
        best_pipeline = None
        best_c = None
        best_metrics = None
        for c_value in CLASSICAL_C_GRID:
            pipeline = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("classifier", LogisticRegression(max_iter=2000, C=c_value, random_state=SEED)),
                ]
            )
            pipeline.fit(X_train, y_train)
            metrics = compute_metrics(y_validation, pipeline.predict(X_validation).tolist(), AMPLITUDE_LABEL_NAMES)
            if best_metrics is None or _score_tuple(metrics, model_name) > _score_tuple(best_metrics, model_name):
                best_pipeline = pipeline
                best_c = c_value
                best_metrics = metrics
        test_predictions = best_pipeline.predict(X_test).tolist()
        summary = {
            "model": model_name,
            "best_c": best_c,
            "validation": best_metrics,
            "test": compute_metrics(y_test, test_predictions, AMPLITUDE_LABEL_NAMES),
        }
        return best_pipeline, summary, y_test, test_predictions

    market_bundle, market_summary, market_y_test, market_preds = train_structured("market_only_amp", PRICE_FEATURES)
    structured_features = PRICE_FEATURES + CLAIM_FEATURES + SENTIMENT_FEATURES
    structured_bundle, structured_summary, structured_y_test, structured_preds = train_structured(
        "claim_sentiment_structured_amp",
        structured_features,
    )

    joblib.dump({"model_name": "market_only_amp", "pipeline": market_bundle, "feature_columns": PRICE_FEATURES}, AMPLITUDE_MARKET_MODEL_PATH)
    joblib.dump({"model_name": "claim_sentiment_structured_amp", "pipeline": structured_bundle, "feature_columns": structured_features}, AMPLITUDE_STRUCTURED_MODEL_PATH)
    save_confusion_matrix_figure(market_y_test, market_preds, AMPLITUDE_LABEL_NAMES, AMPLITUDE_MARKET_FIGURE_PATH, "TSLA Amplitude Market-only (Test)")
    save_confusion_matrix_figure(structured_y_test, structured_preds, AMPLITUDE_LABEL_NAMES, AMPLITUDE_STRUCTURED_FIGURE_PATH, "TSLA Amplitude Claim+Sentiment Structured (Test)")

    rows = []
    for summary in (market_summary, structured_summary):
        for split_name in ("validation", "test"):
            rows.append(
                {
                    "model": summary["model"],
                    "split": split_name,
                    "accuracy": summary[split_name]["accuracy"],
                    "macro_f1": summary[split_name]["macro_f1"],
                }
            )
    prediction_rows = []
    for split_name, frame in split_frames.items():
        if split_name == "train":
            continue
        market_prob = market_bundle.predict_proba(frame[PRICE_FEATURES].to_numpy())[:, 1]
        struct_prob = structured_bundle.predict_proba(frame[structured_features].to_numpy())[:, 1]
        for idx, row in frame.reset_index(drop=True).iterrows():
            prediction_rows.append(
                {
                    "trade_date": row["trade_date"].date().isoformat(),
                    "split": split_name,
                    "amplitude_label": int(row["amplitude_label"]),
                    "market_only_amp_prob": float(market_prob[idx]),
                    "claim_sentiment_structured_amp_prob": float(struct_prob[idx]),
                }
            )
    write_rows_csv(AMPLITUDE_CSV, rows)
    write_rows_csv(AMPLITUDE_PREDICTIONS_CSV, prediction_rows)

    summary = {
        "skipped": False,
        "threshold": threshold,
        "split_large_move_counts": {split_name: int(frame["amplitude_label"].sum()) for split_name, frame in split_frames.items()},
        "models": {"market_only_amp": market_summary, "claim_sentiment_structured_amp": structured_summary},
        "rows": rows,
    }
    write_json(AMPLITUDE_SUMMARY_PATH, summary)
    return summary
