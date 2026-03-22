from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import (
    BUILD_STOCK_PANEL_SUMMARY_PATH,
    CLASSICAL_C_GRID,
    CRSP_DAILY_TABLE,
    CRSP_NAMES_TABLE_CANDIDATES,
    EXTRACT_STOCK_SUMMARY_PATH,
    FORWARD_RV_WINDOW,
    HIGH_RV_QUANTILE,
    LONG_RETURN_WINDOW,
    PART1_LABEL_NAMES,
    PART1_METRICS_CSV,
    PART1_METRICS_JSON,
    PART1_PERSISTENCE_FIGURE,
    PART1_PREDICTIONS_CSV,
    PART1_STOCK_LOGREG_FIGURE,
    PART1_STOCK_MODEL_PATH,
    PROJECT_END_DATE,
    PROJECT_START_DATE,
    RAW_STOCK_DAILY_PATH,
    RAW_STOCK_NAMES_PATH,
    SEED,
    SHORT_RETURN_WINDOW,
    STOCK_FEATURES,
    STOCK_IDENTIFIER_MAP_PATH,
    STOCK_PANEL_PATH,
    TRAIN_END_DATE,
    TRADING_DAYS_PER_YEAR,
    UNIVERSE_CSV,
    VALIDATION_END_DATE,
)
from .data_access import connect_wrds, execute_candidate_queries
from .evaluation import compute_metrics, save_confusion_matrix_figure
from .utils import csv_quote_join, normalize_ticker, read_universe_csv, ticker_variants, write_json, write_rows_csv


def _load_universe() -> pd.DataFrame:
    universe = read_universe_csv(UNIVERSE_CSV).copy()
    universe["ticker"] = universe["ticker"].map(normalize_ticker)
    return universe


def _stock_names_query_candidates(universe_variants: list[str]) -> list[str]:
    tickers = csv_quote_join(universe_variants)
    queries = []
    for table_name in CRSP_NAMES_TABLE_CANDIDATES:
        queries.append(
            f"""
            select
                permno,
                ticker,
                comnam,
                ncusip,
                namedt::date as namedt,
                nameenddt::date as nameenddt,
                shrcd,
                exchcd
            from {table_name}
            where upper(ticker) in ({tickers})
              and coalesce(shrcd, 10) in (10, 11)
            """
        )
        queries.append(
            f"""
            select
                permno,
                ticker,
                comnam,
                ncusip,
                namedt::date as namedt,
                nameendt::date as nameenddt,
                shrcd,
                exchcd
            from {table_name}
            where upper(ticker) in ({tickers})
              and coalesce(shrcd, 10) in (10, 11)
            """
        )
    return queries


def _stock_daily_query(permnos: list[int]) -> str:
    permno_list = ", ".join(str(int(value)) for value in sorted(set(permnos)))
    return f"""
        select
            permno,
            date::date as trade_date,
            ret,
            prc,
            vol,
            shrout
        from {CRSP_DAILY_TABLE}
        where permno in ({permno_list})
          and date between '{PROJECT_START_DATE}' and '{PROJECT_END_DATE}'
    """


def _resolve_universe_permnos(stock_names: pd.DataFrame, universe: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    names = stock_names.copy()
    names["ticker_normalized"] = names["ticker"].map(normalize_ticker)
    names["ncusip8"] = names["ncusip"].fillna("").astype(str).str[:8]
    names["nameenddt"] = pd.to_datetime(names["nameenddt"], errors="coerce").fillna(pd.Timestamp("2099-12-31"))
    names["namedt"] = pd.to_datetime(names["namedt"], errors="coerce").fillna(pd.Timestamp("1900-01-01"))

    selected_rows: list[dict[str, object]] = []
    unresolved: list[str] = []
    for row in universe.to_dict("records"):
        variants = ticker_variants(str(row["ticker"]))
        matches = names[names["ticker_normalized"].isin(variants)].copy()
        if matches.empty:
            unresolved.append(str(row["ticker"]))
            continue
        matches = matches.sort_values(["nameenddt", "namedt"], ascending=[False, False])
        best = matches.iloc[0]
        selected_rows.append(
            {
                "universe_ticker": str(row["ticker"]),
                "company_name": row.get("company_name", ""),
                "sector": row.get("sector", ""),
                "permno": int(best["permno"]),
                "crsp_ticker": best["ticker"],
                "comnam": best.get("comnam", ""),
                "ncusip8": best.get("ncusip8", ""),
            }
        )
    identifier_map = pd.DataFrame(selected_rows).drop_duplicates(subset=["permno"]).reset_index(drop=True)
    return identifier_map, unresolved


def extract_stock_data() -> dict[str, object]:
    universe = _load_universe()
    if RAW_STOCK_NAMES_PATH.exists() and RAW_STOCK_DAILY_PATH.exists() and STOCK_IDENTIFIER_MAP_PATH.exists():
        stock_names = pd.read_parquet(RAW_STOCK_NAMES_PATH)
        stock_daily = pd.read_parquet(RAW_STOCK_DAILY_PATH)
        identifier_map = pd.read_parquet(STOCK_IDENTIFIER_MAP_PATH)
        summary = {
            "source": "cached_local_files",
            "universe_size": int(len(universe)),
            "resolved_permnos": int(identifier_map["permno"].nunique()),
            "stock_name_rows": int(len(stock_names)),
            "stock_daily_rows": int(len(stock_daily)),
            "date_range": {
                "min_date": str(pd.to_datetime(stock_daily["trade_date"]).min().date()),
                "max_date": str(pd.to_datetime(stock_daily["trade_date"]).max().date()),
            },
        }
        write_json(EXTRACT_STOCK_SUMMARY_PATH, summary)
        return summary

    variants = sorted({variant for ticker in universe["ticker"] for variant in ticker_variants(str(ticker))})
    connection = connect_wrds()
    stock_names = execute_candidate_queries(connection, _stock_names_query_candidates(variants))
    identifier_map, unresolved = _resolve_universe_permnos(stock_names, universe)
    if identifier_map.empty:
        raise RuntimeError("Could not resolve any PERMNOs for the configured universe.")

    stock_daily = connection.raw_sql(_stock_daily_query(identifier_map["permno"].tolist()))
    stock_daily["trade_date"] = pd.to_datetime(stock_daily["trade_date"])

    RAW_STOCK_NAMES_PATH.parent.mkdir(parents=True, exist_ok=True)
    stock_names.to_parquet(RAW_STOCK_NAMES_PATH, index=False)
    stock_daily.to_parquet(RAW_STOCK_DAILY_PATH, index=False)
    identifier_map.to_parquet(STOCK_IDENTIFIER_MAP_PATH, index=False)

    summary = {
        "source": "wrds_query",
        "universe_size": int(len(universe)),
        "resolved_permnos": int(identifier_map["permno"].nunique()),
        "unresolved_tickers": unresolved,
        "stock_name_rows": int(len(stock_names)),
        "stock_daily_rows": int(len(stock_daily)),
        "date_range": {
            "min_date": str(stock_daily["trade_date"].min().date()),
            "max_date": str(stock_daily["trade_date"].max().date()),
        },
    }
    write_json(EXTRACT_STOCK_SUMMARY_PATH, summary)
    return summary


def _future_realized_volatility(returns: pd.Series, window: int = FORWARD_RV_WINDOW) -> np.ndarray:
    values = pd.to_numeric(returns, errors="coerce").to_numpy(dtype=float)
    output = np.full(len(values), np.nan, dtype=float)
    if len(values) <= window:
        return output
    shifted = values[1:]
    if len(shifted) < window:
        return output
    windows = sliding_window_view(shifted, window_shape=window)
    output[: len(windows)] = np.nanstd(windows, axis=1, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
    return output


def _compounded_return(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).apply(lambda values: float(np.prod(1.0 + values) - 1.0), raw=True)


def _compute_stock_features(frame: pd.DataFrame) -> pd.DataFrame:
    panel = frame.sort_values(["permno", "trade_date"]).reset_index(drop=True).copy()
    panel["ret"] = pd.to_numeric(panel["ret"], errors="coerce")
    panel["prc"] = pd.to_numeric(panel["prc"], errors="coerce").abs()
    panel["vol"] = pd.to_numeric(panel["vol"], errors="coerce")
    panel["shrout"] = pd.to_numeric(panel["shrout"], errors="coerce")
    panel = panel.dropna(subset=["ret", "prc", "vol", "shrout"]).reset_index(drop=True)
    panel["turnover_1d"] = panel["vol"] / (panel["shrout"] * 1000.0)

    pieces = []
    for _, group in panel.groupby("permno", sort=False):
        local = group.sort_values("trade_date").copy()
        local["ret_1d"] = local["ret"]
        local["ret_5d"] = _compounded_return(local["ret"], SHORT_RETURN_WINDOW)
        local["ret_20d"] = _compounded_return(local["ret"], LONG_RETURN_WINDOW)
        local["rv_5d"] = local["ret"].rolling(SHORT_RETURN_WINDOW).std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
        local["rv_20d"] = local["ret"].rolling(LONG_RETURN_WINDOW).std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
        local["turnover_20d"] = local["turnover_1d"].rolling(LONG_RETURN_WINDOW).mean()
        local["future_rv_20d"] = _future_realized_volatility(local["ret"], FORWARD_RV_WINDOW)
        pieces.append(local)
    return pd.concat(pieces, ignore_index=True)


def _split_by_date(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    train = frame[frame["trade_date"] <= pd.Timestamp(TRAIN_END_DATE)].reset_index(drop=True)
    validation = frame[
        (frame["trade_date"] > pd.Timestamp(TRAIN_END_DATE)) & (frame["trade_date"] <= pd.Timestamp(VALIDATION_END_DATE))
    ].reset_index(drop=True)
    test = frame[(frame["trade_date"] > pd.Timestamp(VALIDATION_END_DATE)) & (frame["trade_date"] <= pd.Timestamp(PROJECT_END_DATE))].reset_index(drop=True)
    if train.empty or validation.empty or test.empty:
        raise RuntimeError("One of the chronological stock-panel splits is empty.")
    return {"train": train, "validation": validation, "test": test}


def build_stock_panel() -> dict[str, object]:
    if not (RAW_STOCK_NAMES_PATH.exists() and RAW_STOCK_DAILY_PATH.exists() and STOCK_IDENTIFIER_MAP_PATH.exists()):
        extract_stock_data()

    stock_daily = pd.read_parquet(RAW_STOCK_DAILY_PATH)
    stock_daily["trade_date"] = pd.to_datetime(stock_daily["trade_date"])
    identifier_map = pd.read_parquet(STOCK_IDENTIFIER_MAP_PATH)

    panel = stock_daily.merge(identifier_map, on="permno", how="inner")
    panel = _compute_stock_features(panel)
    panel = panel.dropna(subset=STOCK_FEATURES + ["future_rv_20d"]).reset_index(drop=True)

    splits = _split_by_date(panel)
    threshold = float(splits["train"]["future_rv_20d"].quantile(HIGH_RV_QUANTILE))
    panel["high_rv_regime"] = (panel["future_rv_20d"] >= threshold).astype(int)
    panel.to_parquet(STOCK_PANEL_PATH, index=False)

    summary = {
        "panel_rows": int(len(panel)),
        "permno_count": int(panel["permno"].nunique()),
        "date_range": {
            "min_date": str(panel["trade_date"].min().date()),
            "max_date": str(panel["trade_date"].max().date()),
        },
        "train_threshold_future_rv_20d": threshold,
        "split_sizes": {split_name: int(len(split_frame)) for split_name, split_frame in splits.items()},
        "positive_share_by_split": {
            split_name: float((split_frame["future_rv_20d"] >= threshold).mean()) for split_name, split_frame in splits.items()
        },
    }
    write_json(BUILD_STOCK_PANEL_SUMMARY_PATH, summary)
    return summary


def _load_stock_panel() -> pd.DataFrame:
    if not STOCK_PANEL_PATH.exists():
        build_stock_panel()
    frame = pd.read_parquet(STOCK_PANEL_PATH)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    return frame


def train_part1() -> dict[str, object]:
    panel = _load_stock_panel()
    splits = _split_by_date(panel)

    train_threshold_rv20 = float(splits["train"]["rv_20d"].quantile(HIGH_RV_QUANTILE))

    y_validation = splits["validation"]["high_rv_regime"].astype(int).tolist()
    y_test = splits["test"]["high_rv_regime"].astype(int).tolist()

    persistence_validation_pred = (splits["validation"]["rv_20d"] >= train_threshold_rv20).astype(int).tolist()
    persistence_test_pred = (splits["test"]["rv_20d"] >= train_threshold_rv20).astype(int).tolist()
    persistence_validation_prob = splits["validation"]["rv_20d"].astype(float).fillna(0.0).tolist()
    persistence_test_prob = splits["test"]["rv_20d"].astype(float).fillna(0.0).tolist()
    persistence_summary = {
        "model": "persistence_baseline",
        "rv_20d_threshold": train_threshold_rv20,
        "validation": compute_metrics(y_validation, persistence_validation_pred, persistence_validation_prob, PART1_LABEL_NAMES),
        "test": compute_metrics(y_test, persistence_test_pred, persistence_test_prob, PART1_LABEL_NAMES),
    }

    X_train = splits["train"][STOCK_FEATURES].to_numpy()
    y_train = splits["train"]["high_rv_regime"].astype(int).to_numpy()
    X_validation = splits["validation"][STOCK_FEATURES].to_numpy()
    X_test = splits["test"][STOCK_FEATURES].to_numpy()

    best_pipeline = None
    best_c = None
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
            best_score = score

    validation_prob = best_pipeline.predict_proba(X_validation)[:, 1].tolist()
    validation_pred = best_pipeline.predict(X_validation).tolist()
    test_prob = best_pipeline.predict_proba(X_test)[:, 1].tolist()
    test_pred = best_pipeline.predict(X_test).tolist()

    stock_summary = {
        "model": "stock_only_logreg",
        "best_c": best_c,
        "feature_columns": STOCK_FEATURES,
        "validation": compute_metrics(y_validation, validation_pred, validation_prob, PART1_LABEL_NAMES),
        "test": compute_metrics(y_test, test_pred, test_prob, PART1_LABEL_NAMES),
    }
    joblib.dump({"model_name": "stock_only_logreg", "pipeline": best_pipeline, "feature_columns": STOCK_FEATURES}, PART1_STOCK_MODEL_PATH)

    save_confusion_matrix_figure(y_test, persistence_test_pred, PART1_LABEL_NAMES, PART1_PERSISTENCE_FIGURE, "Part 1 Persistence (Test)")
    save_confusion_matrix_figure(y_test, test_pred, PART1_LABEL_NAMES, PART1_STOCK_LOGREG_FIGURE, "Part 1 Stock-only (Test)")

    rows = []
    for summary in (persistence_summary, stock_summary):
        for split_name in ("validation", "test"):
            rows.append(
                {
                    "model": summary["model"],
                    "split": split_name,
                    "accuracy": summary[split_name]["accuracy"],
                    "macro_f1": summary[split_name]["macro_f1"],
                    "balanced_accuracy": summary[split_name]["balanced_accuracy"],
                    "auroc": summary[split_name]["auroc"],
                    "pr_auc": summary[split_name]["pr_auc"],
                }
            )

    prediction_rows = []
    for split_name, frame in (("validation", splits["validation"]), ("test", splits["test"])):
        stock_prob = best_pipeline.predict_proba(frame[STOCK_FEATURES].to_numpy())[:, 1]
        stock_pred = best_pipeline.predict(frame[STOCK_FEATURES].to_numpy())
        persistence_pred = (frame["rv_20d"] >= train_threshold_rv20).astype(int).to_numpy()
        persistence_prob = frame["rv_20d"].astype(float).to_numpy()
        for idx, row in frame.reset_index(drop=True).iterrows():
            prediction_rows.append(
                {
                    "permno": int(row["permno"]),
                    "universe_ticker": row["universe_ticker"],
                    "trade_date": row["trade_date"].date().isoformat(),
                    "split": split_name,
                    "label": int(row["high_rv_regime"]),
                    "persistence_prob": float(persistence_prob[idx]),
                    "persistence_pred": int(persistence_pred[idx]),
                    "stock_only_prob": float(stock_prob[idx]),
                    "stock_only_pred": int(stock_pred[idx]),
                }
            )

    write_rows_csv(PART1_METRICS_CSV, rows)
    write_rows_csv(PART1_PREDICTIONS_CSV, prediction_rows)

    classifier = best_pipeline.named_steps["classifier"]
    summary = {
        "split_sizes": {split_name: int(len(split_frame)) for split_name, split_frame in splits.items()},
        "models": {"persistence_baseline": persistence_summary, "stock_only_logreg": stock_summary},
        "stock_logreg_coefficients": [
            {"feature": feature, "coefficient": float(coefficient)}
            for feature, coefficient in zip(STOCK_FEATURES, classifier.coef_[0])
        ],
        "rows": rows,
    }
    write_json(PART1_METRICS_JSON, summary)
    return summary
