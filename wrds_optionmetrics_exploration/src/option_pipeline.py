from __future__ import annotations

import joblib
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import (
    BUILD_OPTION_FEATURES_SUMMARY_PATH,
    CLASSICAL_C_GRID,
    COMPLETE_CASE_PANEL_PATH,
    EXTRACT_OPTION_SUMMARY_PATH,
    OPTION_FEATURE_PANEL_PATH,
    OPTION_FEATURES,
    OPTION_SECURITIES_TABLE,
    OPTION_SECURITY_LINK_PATH,
    PART1_LABEL_NAMES,
    PART2_COMBINED_LOGREG_FIGURE,
    PART2_COMBINED_MODEL_PATH,
    PART2_METRICS_CSV,
    PART2_METRICS_JSON,
    PART2_OPTION_LOGREG_FIGURE,
    PART2_OPTION_MODEL_PATH,
    PART2_PREDICTIONS_CSV,
    PART2_STOCK_LOGREG_FIGURE,
    PART2_STOCK_MODEL_PATH,
    PROJECT_END_DATE,
    PROJECT_START_DATE,
    RAW_OPTION_PRICES_PATH,
    RAW_OPTION_SECURITIES_PATH,
    SEED,
    STOCK_FEATURES,
    STOCK_IDENTIFIER_MAP_PATH,
    STOCK_PANEL_PATH,
    TRAIN_END_DATE,
    VALIDATION_END_DATE,
)
from .data_access import connect_wrds, execute_candidate_queries
from .evaluation import compute_metrics, save_confusion_matrix_figure
from .utils import csv_quote_join, normalize_ticker, ticker_variants, write_json, write_rows_csv


def _option_price_table_refs(connection) -> list[str]:
    start_year = int(PROJECT_START_DATE[:4])
    end_year = int(PROJECT_END_DATE[:4])
    libraries = set(connection.list_libraries())
    table_refs: list[str] = []
    for library_name in ("optionm", "optionm_all"):
        if library_name not in libraries:
            continue
        table_names = {str(name) for name in connection.list_tables(library=library_name)}
        for year in range(start_year, end_year + 1):
            table_name = f"opprcd{year}"
            if table_name in table_names:
                table_refs.append(f"{library_name}.{table_name}")
    if not table_refs:
        raise RuntimeError("Could not find any yearly OptionMetrics price tables for the configured date window.")
    return table_refs


def _option_years() -> list[int]:
    return list(range(int(PROJECT_START_DATE[:4]), int(PROJECT_END_DATE[:4]) + 1))


def _option_year_path(year: int) -> Path:
    return RAW_OPTION_SECURITIES_PATH.parent / f"optionm_option_prices_{year}.parquet"


def _load_cached_option_prices() -> DataFrame:
    yearly_paths = [path for year in _option_years() if (path := _option_year_path(year)).exists()]
    if yearly_paths:
        frames = [pd.read_parquet(path) for path in yearly_paths]
        option_prices = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    elif RAW_OPTION_PRICES_PATH.exists():
        option_prices = pd.read_parquet(RAW_OPTION_PRICES_PATH)
    else:
        raise RuntimeError("Option price data is missing. Run extract_option_data first.")
    option_prices["trade_date"] = pd.to_datetime(option_prices["trade_date"])
    option_prices["exdate"] = pd.to_datetime(option_prices["exdate"])
    return option_prices


def _fetch_option_prices_for_window(connection, secid_list_sql: str, table_ref: str, start_date: str, end_date: str) -> DataFrame:
    sql_templates = [
        """
        select
            secid,
            date::date as trade_date,
            exdate::date as exdate,
            cp_flag,
            delta,
            impl_volatility,
            open_interest,
            volume,
            best_bid,
            best_offer
        from {table_ref}
        where secid in ({secid_list_sql})
          and date between '{project_start_date}' and '{project_end_date}'
        """,
        """
        select
            secid,
            date::date as trade_date,
            exdate::date as exdate,
            cp_flag,
            delta,
            impl_vola as impl_volatility,
            open_interest,
            volume,
            best_bid,
            best_offer
        from {table_ref}
        where secid in ({secid_list_sql})
          and date between '{project_start_date}' and '{project_end_date}'
        """,
        """
        select
            secid,
            date::date as trade_date,
            exdate::date as exdate,
            cp_flag,
            delta,
            impl_volatility,
            open_interest,
            volume,
            bid as best_bid,
            offer as best_offer
        from {table_ref}
        where secid in ({secid_list_sql})
          and date between '{project_start_date}' and '{project_end_date}'
        """,
    ]
    queries = [
        template.format(
            table_ref=table_ref,
            secid_list_sql=secid_list_sql,
            project_start_date=start_date,
            project_end_date=end_date,
        )
        for template in sql_templates
    ]
    return execute_candidate_queries(connection, queries)


def _load_stock_panel() -> pd.DataFrame:
    if not STOCK_PANEL_PATH.exists():
        raise RuntimeError("Stock panel is missing. Run build_stock_panel first.")
    frame = pd.read_parquet(STOCK_PANEL_PATH)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    return frame


def _split_by_date(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    train = frame[frame["trade_date"] <= pd.Timestamp(TRAIN_END_DATE)].reset_index(drop=True)
    validation = frame[
        (frame["trade_date"] > pd.Timestamp(TRAIN_END_DATE)) & (frame["trade_date"] <= pd.Timestamp(VALIDATION_END_DATE))
    ].reset_index(drop=True)
    test = frame[(frame["trade_date"] > pd.Timestamp(VALIDATION_END_DATE)) & (frame["trade_date"] <= pd.Timestamp(PROJECT_END_DATE))].reset_index(drop=True)
    if train.empty or validation.empty or test.empty:
        raise RuntimeError("One of the chronological complete-case splits is empty.")
    return {"train": train, "validation": validation, "test": test}


def extract_option_data() -> dict[str, object]:
    yearly_paths = [path for year in _option_years() if (path := _option_year_path(year)).exists()]
    if RAW_OPTION_SECURITIES_PATH.exists() and OPTION_SECURITY_LINK_PATH.exists() and (RAW_OPTION_PRICES_PATH.exists() or len(yearly_paths) == len(_option_years())):
        securities = pd.read_parquet(RAW_OPTION_SECURITIES_PATH)
        option_prices = _load_cached_option_prices()
        security_link = pd.read_parquet(OPTION_SECURITY_LINK_PATH)
        summary = {
            "source": "cached_local_files",
            "secid_count": int(security_link["secid"].nunique()),
            "permno_count": int(security_link["permno"].nunique()),
            "option_security_rows": int(len(securities)),
            "option_price_rows": int(len(option_prices)),
            "option_price_files": [path.name for path in yearly_paths] if yearly_paths else [RAW_OPTION_PRICES_PATH.name],
            "date_range": {
                "min_date": str(pd.to_datetime(option_prices["trade_date"]).min().date()),
                "max_date": str(pd.to_datetime(option_prices["trade_date"]).max().date()),
            },
        }
        write_json(EXTRACT_OPTION_SUMMARY_PATH, summary)
        return summary

    stock_identifier_map = pd.read_parquet(STOCK_IDENTIFIER_MAP_PATH)
    ticker_variants_all = sorted(
        {
            variant
            for ticker in stock_identifier_map["universe_ticker"].astype(str).tolist()
            for variant in ticker_variants(ticker)
        }
    )
    connection = connect_wrds()

    ticker_list = csv_quote_join(ticker_variants_all)
    securities = execute_candidate_queries(
        connection,
        [
            f"""
            select
                secid,
                ticker,
                issuer,
                cusip
            from {OPTION_SECURITIES_TABLE}
            where upper(ticker) in ({ticker_list})
            """,
            f"""
            select
                secid,
                ticker,
                issue_name as issuer,
                cusip
            from {OPTION_SECURITIES_TABLE}
            where upper(ticker) in ({ticker_list})
            """,
        ],
    )
    if securities.empty:
        raise RuntimeError("No OptionMetrics security records matched the configured universe.")

    securities["ticker_normalized"] = securities["ticker"].map(normalize_ticker)
    securities["cusip8"] = securities["cusip"].fillna("").astype(str).str[:8]

    stock_identifier_map["ticker_normalized"] = stock_identifier_map["universe_ticker"].map(normalize_ticker)
    stock_identifier_map["ncusip8"] = stock_identifier_map["ncusip8"].fillna("").astype(str).str[:8]

    by_cusip = securities.merge(
        stock_identifier_map[["permno", "universe_ticker", "ticker_normalized", "ncusip8"]],
        left_on="cusip8",
        right_on="ncusip8",
        how="inner",
    )
    by_cusip["match_method"] = "cusip8"

    remaining = securities[~securities["secid"].isin(by_cusip["secid"])]
    by_ticker = remaining.merge(
        stock_identifier_map[["permno", "universe_ticker", "ticker_normalized"]],
        on="ticker_normalized",
        how="inner",
    )
    by_ticker["match_method"] = "ticker"

    security_link = pd.concat([by_cusip, by_ticker], ignore_index=True)
    security_link = security_link.drop_duplicates(subset=["secid", "permno"]).reset_index(drop=True)
    if security_link.empty:
        raise RuntimeError("Could not derive any SECID-to-PERMNO links for the static universe.")

    secid_list = ", ".join(str(int(value)) for value in sorted(security_link["secid"].unique()))
    RAW_OPTION_SECURITIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    securities.to_parquet(RAW_OPTION_SECURITIES_PATH, index=False)
    security_link.to_parquet(OPTION_SECURITY_LINK_PATH, index=False)

    table_refs = _option_price_table_refs(connection)
    table_ref_by_year: dict[int, str] = {}
    for table_ref in table_refs:
        year = int(table_ref[-4:])
        if year not in table_ref_by_year:
            table_ref_by_year[year] = table_ref

    frames: list[DataFrame] = []
    rows_by_year: dict[str, int] = {}
    saved_files: list[str] = []
    for year in _option_years():
        year_path = _option_year_path(year)
        if year_path.exists():
            print(f"Using cached OptionMetrics file for {year}: {year_path.name}")
            year_frame = pd.read_parquet(year_path)
        else:
            table_ref = table_ref_by_year.get(year)
            if table_ref is None:
                raise RuntimeError(f"Could not find an OptionMetrics opprcd table for year {year}.")
            print(f"Querying OptionMetrics {table_ref} for {year}...")
            year_frame = _fetch_option_prices_for_window(connection, secid_list, table_ref, f"{year}-01-01", f"{year}-12-31")
            year_frame["trade_date"] = pd.to_datetime(year_frame["trade_date"])
            year_frame["exdate"] = pd.to_datetime(year_frame["exdate"])
            year_frame.to_parquet(year_path, index=False)
        rows_by_year[str(year)] = int(len(year_frame))
        saved_files.append(year_path.name)
        if not year_frame.empty:
            frames.append(year_frame)

    if not frames:
        raise RuntimeError("No OptionMetrics price rows were returned for the configured SECIDs and date window.")

    option_prices = pd.concat(frames, ignore_index=True)
    option_prices["trade_date"] = pd.to_datetime(option_prices["trade_date"])
    option_prices["exdate"] = pd.to_datetime(option_prices["exdate"])

    summary = {
        "source": "wrds_query",
        "secid_count": int(security_link["secid"].nunique()),
        "permno_count": int(security_link["permno"].nunique()),
        "option_security_rows": int(len(securities)),
        "option_price_rows": int(len(option_prices)),
        "option_price_rows_by_year": rows_by_year,
        "option_price_files": saved_files,
        "match_method_counts": security_link["match_method"].value_counts().astype(int).to_dict(),
        "date_range": {
            "min_date": str(option_prices["trade_date"].min().date()),
            "max_date": str(option_prices["trade_date"].max().date()),
        },
    }
    write_json(EXTRACT_OPTION_SUMMARY_PATH, summary)
    return summary


def _median_if_min_count(series: pd.Series, min_count: int = 2) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) < min_count:
        return float("nan")
    return float(clean.median())


def build_option_features() -> dict[str, object]:
    yearly_paths = [path for year in _option_years() if (path := _option_year_path(year)).exists()]
    if not (RAW_OPTION_SECURITIES_PATH.exists() and OPTION_SECURITY_LINK_PATH.exists() and (RAW_OPTION_PRICES_PATH.exists() or yearly_paths)):
        extract_option_data()

    stock_panel = _load_stock_panel()
    option_prices = _load_cached_option_prices()
    security_link = pd.read_parquet(OPTION_SECURITY_LINK_PATH)

    option_prices = option_prices.merge(security_link[["secid", "permno", "universe_ticker"]], on="secid", how="inner")
    option_prices["cp_flag"] = option_prices["cp_flag"].astype(str).str.upper().str[0]
    for column in ("delta", "impl_volatility", "open_interest", "volume", "best_bid", "best_offer"):
        option_prices[column] = pd.to_numeric(option_prices[column], errors="coerce")
    option_prices["dte"] = (option_prices["exdate"] - option_prices["trade_date"]).dt.days
    option_prices["abs_delta"] = option_prices["delta"].abs()

    eligible = option_prices[
        option_prices["impl_volatility"].gt(0)
        & option_prices["open_interest"].gt(0)
        & option_prices["best_bid"].gt(0)
        & option_prices["best_offer"].gt(0)
        & option_prices["best_bid"].le(option_prices["best_offer"])
        & option_prices["delta"].notna()
    ].copy()

    short = eligible[eligible["dte"].between(20, 45)]
    long_atm = eligible[eligible["dte"].between(46, 90) & (eligible["abs_delta"].sub(0.50).abs().le(0.10))]
    atm_short = short[short["abs_delta"].sub(0.50).abs().le(0.10)]
    otm_put_short = short[(short["cp_flag"] == "P") & short["abs_delta"].between(0.15, 0.35)]

    group_keys = ["permno", "trade_date"]
    iv_atm_short = (
        atm_short.groupby(group_keys)["impl_volatility"]
        .agg(lambda series: _median_if_min_count(series, 2))
        .rename("iv_atm_short")
        .reset_index()
    )
    iv_atm_long = (
        long_atm.groupby(group_keys)["impl_volatility"]
        .agg(lambda series: _median_if_min_count(series, 2))
        .rename("iv_atm_long")
        .reset_index()
    )
    put_skew = (
        otm_put_short.groupby(group_keys)["impl_volatility"]
        .agg(lambda series: _median_if_min_count(series, 2))
        .rename("otm_put_iv_short")
        .reset_index()
    )
    short_ratios = (
        short.groupby(group_keys + ["cp_flag"])
        .agg(total_open_interest=("open_interest", "sum"), total_volume=("volume", "sum"))
        .reset_index()
    )
    short_ratios = short_ratios.pivot(index=group_keys, columns="cp_flag", values=["total_open_interest", "total_volume"])
    short_ratios.columns = ["_".join(column).lower() for column in short_ratios.columns]
    short_ratios = short_ratios.reset_index()
    for column in ("total_open_interest_p", "total_open_interest_c", "total_volume_p", "total_volume_c"):
        if column not in short_ratios.columns:
            short_ratios[column] = np.nan
    short_totals = short.groupby(group_keys).agg(total_oi_short=("open_interest", "sum")).reset_index()

    option_features = iv_atm_short.merge(iv_atm_long, on=group_keys, how="outer")
    option_features = option_features.merge(put_skew, on=group_keys, how="outer")
    option_features = option_features.merge(short_ratios, on=group_keys, how="outer")
    option_features = option_features.merge(short_totals, on=group_keys, how="outer")

    option_features["iv_term_slope"] = option_features["iv_atm_long"] - option_features["iv_atm_short"]
    option_features["put_skew_short"] = option_features["otm_put_iv_short"] - option_features["iv_atm_short"]
    oi_denominator = option_features["total_open_interest_c"].astype(float).mask(option_features["total_open_interest_c"].eq(0))
    vol_denominator = option_features["total_volume_c"].astype(float).mask(option_features["total_volume_c"].eq(0))
    option_features["pc_oi_ratio_short"] = option_features["total_open_interest_p"].astype(float).div(oi_denominator)
    option_features["pc_vol_ratio_short"] = option_features["total_volume_p"].astype(float).div(vol_denominator)
    option_features["log_total_oi_short"] = np.log1p(option_features["total_oi_short"])
    option_features = option_features[group_keys + OPTION_FEATURES].copy()
    option_features.to_parquet(OPTION_FEATURE_PANEL_PATH, index=False)

    merged = stock_panel.merge(option_features, on=["permno", "trade_date"], how="left")
    complete_case = merged.dropna(subset=STOCK_FEATURES + OPTION_FEATURES + ["high_rv_regime"]).reset_index(drop=True)
    complete_case.to_parquet(COMPLETE_CASE_PANEL_PATH, index=False)

    summary = {
        "option_feature_rows": int(len(option_features)),
        "merged_panel_rows": int(len(merged)),
        "complete_case_rows": int(len(complete_case)),
        "feature_coverage": {feature: float(merged[feature].notna().mean()) for feature in OPTION_FEATURES},
        "split_complete_case_rows": {
            "train": int(len(complete_case[complete_case["trade_date"] <= pd.Timestamp(TRAIN_END_DATE)])),
            "validation": int(
                len(
                    complete_case[
                        (complete_case["trade_date"] > pd.Timestamp(TRAIN_END_DATE))
                        & (complete_case["trade_date"] <= pd.Timestamp(VALIDATION_END_DATE))
                    ]
                )
            ),
            "test": int(len(complete_case[complete_case["trade_date"] > pd.Timestamp(VALIDATION_END_DATE)])),
        },
    }
    write_json(BUILD_OPTION_FEATURES_SUMMARY_PATH, summary)
    return summary


def train_part2() -> dict[str, object]:
    if not COMPLETE_CASE_PANEL_PATH.exists():
        build_option_features()

    frame = pd.read_parquet(COMPLETE_CASE_PANEL_PATH)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    splits = _split_by_date(frame)

    def train_logreg(model_name: str, feature_columns: list[str]):
        X_train = splits["train"][feature_columns].to_numpy()
        y_train = splits["train"]["high_rv_regime"].astype(int).to_numpy()
        X_validation = splits["validation"][feature_columns].to_numpy()
        y_validation = splits["validation"]["high_rv_regime"].astype(int).tolist()
        X_test = splits["test"][feature_columns].to_numpy()
        y_test = splits["test"]["high_rv_regime"].astype(int).tolist()

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
        summary = {
            "model": model_name,
            "best_c": best_c,
            "feature_columns": feature_columns,
            "validation": compute_metrics(y_validation, validation_pred, validation_prob, PART1_LABEL_NAMES),
            "test": compute_metrics(y_test, test_pred, test_prob, PART1_LABEL_NAMES),
        }
        return best_pipeline, summary

    stock_pipeline, stock_summary = train_logreg("stock_only_logreg_complete_case", STOCK_FEATURES)
    option_pipeline, option_summary = train_logreg("option_only_logreg", OPTION_FEATURES)
    combined_pipeline, combined_summary = train_logreg("combined_logreg", STOCK_FEATURES + OPTION_FEATURES)

    joblib.dump({"model_name": "stock_only_logreg_complete_case", "pipeline": stock_pipeline, "feature_columns": STOCK_FEATURES}, PART2_STOCK_MODEL_PATH)
    joblib.dump({"model_name": "option_only_logreg", "pipeline": option_pipeline, "feature_columns": OPTION_FEATURES}, PART2_OPTION_MODEL_PATH)
    joblib.dump({"model_name": "combined_logreg", "pipeline": combined_pipeline, "feature_columns": STOCK_FEATURES + OPTION_FEATURES}, PART2_COMBINED_MODEL_PATH)

    y_test = splits["test"]["high_rv_regime"].astype(int).tolist()
    save_confusion_matrix_figure(y_test, stock_pipeline.predict(splits["test"][STOCK_FEATURES].to_numpy()).tolist(), PART1_LABEL_NAMES, PART2_STOCK_LOGREG_FIGURE, "Part 2 Stock-only Complete-case (Test)")
    save_confusion_matrix_figure(y_test, option_pipeline.predict(splits["test"][OPTION_FEATURES].to_numpy()).tolist(), PART1_LABEL_NAMES, PART2_OPTION_LOGREG_FIGURE, "Part 2 Option-only (Test)")
    save_confusion_matrix_figure(y_test, combined_pipeline.predict(splits["test"][STOCK_FEATURES + OPTION_FEATURES].to_numpy()).tolist(), PART1_LABEL_NAMES, PART2_COMBINED_LOGREG_FIGURE, "Part 2 Combined (Test)")

    rows = []
    for summary in (stock_summary, option_summary, combined_summary):
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
    for split_name, split_frame in (("validation", splits["validation"]), ("test", splits["test"])):
        stock_prob = stock_pipeline.predict_proba(split_frame[STOCK_FEATURES].to_numpy())[:, 1]
        stock_pred = stock_pipeline.predict(split_frame[STOCK_FEATURES].to_numpy())
        option_prob = option_pipeline.predict_proba(split_frame[OPTION_FEATURES].to_numpy())[:, 1]
        option_pred = option_pipeline.predict(split_frame[OPTION_FEATURES].to_numpy())
        combined_prob = combined_pipeline.predict_proba(split_frame[STOCK_FEATURES + OPTION_FEATURES].to_numpy())[:, 1]
        combined_pred = combined_pipeline.predict(split_frame[STOCK_FEATURES + OPTION_FEATURES].to_numpy())
        for idx, row in split_frame.reset_index(drop=True).iterrows():
            prediction_rows.append(
                {
                    "permno": int(row["permno"]),
                    "universe_ticker": row["universe_ticker"],
                    "trade_date": row["trade_date"].date().isoformat(),
                    "split": split_name,
                    "label": int(row["high_rv_regime"]),
                    "stock_only_prob": float(stock_prob[idx]),
                    "stock_only_pred": int(stock_pred[idx]),
                    "option_only_prob": float(option_prob[idx]),
                    "option_only_pred": int(option_pred[idx]),
                    "combined_prob": float(combined_prob[idx]),
                    "combined_pred": int(combined_pred[idx]),
                }
            )

    write_rows_csv(PART2_METRICS_CSV, rows)
    write_rows_csv(PART2_PREDICTIONS_CSV, prediction_rows)

    option_classifier = option_pipeline.named_steps["classifier"]
    combined_classifier = combined_pipeline.named_steps["classifier"]
    summary = {
        "split_sizes": {split_name: int(len(split_frame)) for split_name, split_frame in splits.items()},
        "models": {
            "stock_only_logreg_complete_case": stock_summary,
            "option_only_logreg": option_summary,
            "combined_logreg": combined_summary,
        },
        "option_only_coefficients": [
            {"feature": feature, "coefficient": float(coefficient)}
            for feature, coefficient in zip(OPTION_FEATURES, option_classifier.coef_[0])
        ],
        "combined_coefficients": [
            {"feature": feature, "coefficient": float(coefficient)}
            for feature, coefficient in zip(STOCK_FEATURES + OPTION_FEATURES, combined_classifier.coef_[0])
        ],
        "rows": rows,
    }
    write_json(PART2_METRICS_JSON, summary)
    return summary
