from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .config import (
    BUILD_SURFACE_FACTORS_SUMMARY_PATH,
    CLASSICAL_C_GRID,
    TRAILING_RETRAIN_MONTHS,
    TRAILING_WINDOW_YEARS,
    XGBOOST_PARAM_GRID,
    OPTION_FEATURE_PANEL_PATH,
    OPTION_FEATURES,
    OPTION_SECURITY_LINK_PATH,
    PART1_LABEL_NAMES,
    PROJECT_END_DATE,
    RAW_OPTION_PRICES_PATH,
    RAW_OPTIONS_DIR,
    SEED,
    STOCK_FEATURES,
    STOCK_PANEL_PATH,
    SURFACE_EXTENSION_METRICS_CSV,
    SURFACE_EXTENSION_METRICS_JSON,
    SURFACE_EXTENSION_PANEL_PATH,
    SURFACE_EXTENSION_PREDICTIONS_CSV,
    SURFACE_FACTOR_PANEL_PATH,
    SURFACE_FEATURES,
    SURFACE_FEATURES_RAW,
    SURFACE_ONLY_FIGURE,
    SURFACE_ONLY_MODEL_PATH,
    STOCK_SURFACE_FIGURE,
    STOCK_SURFACE_MODEL_PATH,
    ALL_FEATURES_FIGURE,
    ALL_FEATURES_MODEL_PATH,
    STOCK_ONLY_XGB_MODEL_PATH,
    OPTION_ONLY_XGB_MODEL_PATH,
    ALL_FEATURES_XGB_MODEL_PATH,
    TRAIN_END_DATE,
    VALIDATION_END_DATE,
)
from .evaluation import compute_metrics, save_confusion_matrix_figure
from .utils import write_json, write_rows_csv


def _split_by_date(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    train = frame[frame["trade_date"] <= pd.Timestamp(TRAIN_END_DATE)].reset_index(drop=True)
    validation = frame[
        (frame["trade_date"] > pd.Timestamp(TRAIN_END_DATE)) & (frame["trade_date"] <= pd.Timestamp(VALIDATION_END_DATE))
    ].reset_index(drop=True)
    test = frame[(frame["trade_date"] > pd.Timestamp(VALIDATION_END_DATE)) & (frame["trade_date"] <= pd.Timestamp(PROJECT_END_DATE))].reset_index(drop=True)
    if train.empty or validation.empty or test.empty:
        raise RuntimeError("One of the chronological surface-extension splits is empty.")
    return {"train": train, "validation": validation, "test": test}


def _quarter_starts(frame: pd.DataFrame) -> list[pd.Timestamp]:
    quarter_starts = (
        frame.assign(quarter=frame["trade_date"].dt.to_period("Q"))
        .groupby("quarter", sort=True)["trade_date"]
        .min()
        .tolist()
    )
    return [pd.Timestamp(value) for value in quarter_starts]


def _fit_logreg_pipeline(X_train: np.ndarray, y_train: np.ndarray, c_value: float) -> Pipeline:
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=5000, C=c_value, random_state=SEED)),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def _trailing_window_predictions(
    panel: pd.DataFrame,
    feature_columns: list[str],
    c_value: float,
    trailing_years: int,
) -> pd.DataFrame:
    prediction_frame = panel[panel["trade_date"] > pd.Timestamp(TRAIN_END_DATE)].copy()
    prediction_frame["split"] = np.where(
        prediction_frame["trade_date"] <= pd.Timestamp(VALIDATION_END_DATE),
        "validation",
        "test",
    )
    prediction_frame = prediction_frame.sort_values(["trade_date", "permno"]).reset_index(drop=True)
    if prediction_frame.empty:
        raise RuntimeError(f"No rows are available for the trailing {trailing_years}y benchmark.")

    prediction_frame["quarter"] = prediction_frame["trade_date"].dt.to_period("Q")
    quarter_starts = _quarter_starts(prediction_frame)
    payloads: list[pd.DataFrame] = []

    for anchor_date in quarter_starts:
        quarter_period = anchor_date.to_period("Q")
        quarter_mask = prediction_frame["quarter"] == quarter_period
        quarter_rows = prediction_frame.loc[quarter_mask].copy()
        if quarter_rows.empty:
            continue

        train_start = anchor_date - pd.DateOffset(years=trailing_years)
        train_mask = (panel["trade_date"] >= train_start) & (panel["trade_date"] < anchor_date)
        train_rows = panel.loc[train_mask].dropna(subset=feature_columns + ["high_rv_regime"]).copy()
        if train_rows.empty:
            continue

        X_train = train_rows[feature_columns].to_numpy()
        y_train = train_rows["high_rv_regime"].astype(int).to_numpy()
        X_quarter = quarter_rows[feature_columns].to_numpy()

        pipeline = _fit_logreg_pipeline(X_train, y_train, c_value=c_value)
        quarter_rows["probability"] = pipeline.predict_proba(X_quarter)[:, 1]
        quarter_rows["prediction"] = pipeline.predict(X_quarter)
        quarter_rows["training_rows"] = int(len(train_rows))
        quarter_rows["anchor_date"] = anchor_date
        payloads.append(
            quarter_rows[
                [
                    "permno",
                    "trade_date",
                    "split",
                    "probability",
                    "prediction",
                    "training_rows",
                    "anchor_date",
                ]
            ]
        )

    if not payloads:
        raise RuntimeError(f"Trailing {trailing_years}y benchmark produced no predictions.")

    trailing_predictions = pd.concat(payloads, ignore_index=True)
    trailing_predictions = trailing_predictions.drop_duplicates(subset=["permno", "trade_date"], keep="last")
    expected_keys = prediction_frame[["permno", "trade_date"]].drop_duplicates()
    covered_keys = trailing_predictions[["permno", "trade_date"]].drop_duplicates()
    missing_keys = expected_keys.merge(covered_keys, on=["permno", "trade_date"], how="left", indicator=True)
    if (missing_keys["_merge"] != "both").any():
        missing_count = int((missing_keys["_merge"] != "both").sum())
        raise RuntimeError(f"Trailing {trailing_years}y benchmark is missing {missing_count} prediction rows.")
    return trailing_predictions


def _option_year_paths() -> list[pd.PathLike]:
    yearly_paths = sorted(RAW_OPTIONS_DIR.glob("optionm_option_prices_*.parquet"))
    if yearly_paths:
        return yearly_paths
    if RAW_OPTION_PRICES_PATH.exists():
        return [RAW_OPTION_PRICES_PATH]
    raise RuntimeError("No cached option parquet files are available. Run extract_option_data first.")


def _load_stock_panel() -> pd.DataFrame:
    if not STOCK_PANEL_PATH.exists():
        raise RuntimeError("Stock panel is missing. Run build_stock_panel first.")
    frame = pd.read_parquet(STOCK_PANEL_PATH)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    return frame


def _load_option_feature_panel() -> pd.DataFrame:
    if not OPTION_FEATURE_PANEL_PATH.exists():
        raise RuntimeError("Option feature panel is missing. Run build_option_features first.")
    frame = pd.read_parquet(OPTION_FEATURE_PANEL_PATH)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    return frame


def _load_surface_factor_panel() -> pd.DataFrame:
    if not SURFACE_FACTOR_PANEL_PATH.exists():
        raise RuntimeError("Surface factor panel is missing. Run build_surface_factors first.")
    frame = pd.read_parquet(SURFACE_FACTOR_PANEL_PATH)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    return frame


def _median_if_min_count(series: pd.Series, min_count: int = 2) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) < min_count:
        return float("nan")
    return float(clean.median())


def _build_bucket_feature(frame: pd.DataFrame, mask: pd.Series, output_name: str, group_keys: list[str]) -> pd.DataFrame:
    subset = frame.loc[mask, group_keys + ["impl_volatility"]].copy()
    if subset.empty:
        return pd.DataFrame(columns=group_keys + [output_name])
    return (
        subset.groupby(group_keys)["impl_volatility"]
        .agg(lambda series: _median_if_min_count(series, 2))
        .rename(output_name)
        .reset_index()
    )


def build_surface_factors() -> dict[str, object]:
    if not OPTION_SECURITY_LINK_PATH.exists():
        raise RuntimeError("Option security link is missing. Run extract_option_data first.")

    security_link = pd.read_parquet(OPTION_SECURITY_LINK_PATH)[["secid", "permno", "universe_ticker"]].drop_duplicates()
    group_keys = ["permno", "universe_ticker", "trade_date"]

    per_year_panels: list[pd.DataFrame] = []
    quote_summary_rows: list[dict[str, object]] = []
    total_raw_rows = 0
    total_eligible_rows = 0

    for option_path in _option_year_paths():
        print(f"Building surface descriptors from {option_path.name}...")
        frame = pd.read_parquet(
            option_path,
            columns=["secid", "trade_date", "exdate", "cp_flag", "delta", "impl_volatility", "open_interest", "volume", "best_bid", "best_offer"],
        )
        total_raw_rows += int(len(frame))
        frame = frame.merge(security_link, on="secid", how="inner")
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        frame["exdate"] = pd.to_datetime(frame["exdate"])
        frame["cp_flag"] = frame["cp_flag"].astype(str).str.upper().str[0]
        for column in ("delta", "impl_volatility", "open_interest", "volume", "best_bid", "best_offer"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame["dte"] = (frame["exdate"] - frame["trade_date"]).dt.days
        frame["abs_delta"] = frame["delta"].abs()

        eligible = frame[
            frame["impl_volatility"].gt(0)
            & frame["open_interest"].gt(0)
            & frame["best_bid"].gt(0)
            & frame["best_offer"].gt(0)
            & frame["best_bid"].le(frame["best_offer"])
            & frame["delta"].notna()
            & frame["dte"].between(14, 180)
            & frame["abs_delta"].between(0.05, 0.60)
            & frame["cp_flag"].isin(["C", "P"])
        ].copy()
        total_eligible_rows += int(len(eligible))

        short_maturity = eligible["dte"].between(20, 45)
        long_maturity = eligible["dte"].between(46, 90)
        atm = eligible["abs_delta"].between(0.40, 0.60)
        wing_25 = eligible["abs_delta"].between(0.15, 0.35)
        is_call = eligible["cp_flag"].eq("C")
        is_put = eligible["cp_flag"].eq("P")

        atm_short = _build_bucket_feature(eligible, short_maturity & atm, "surface_atm_short_raw", group_keys)
        atm_long = _build_bucket_feature(eligible, long_maturity & atm, "surface_atm_long_raw", group_keys)
        put_25_short = _build_bucket_feature(eligible, short_maturity & wing_25 & is_put, "surface_put_25_short_raw", group_keys)
        call_25_short = _build_bucket_feature(eligible, short_maturity & wing_25 & is_call, "surface_call_25_short_raw", group_keys)
        quote_counts = eligible.groupby(group_keys).size().rename("surface_quote_count").reset_index()

        year_panel = atm_short.merge(atm_long, on=group_keys, how="outer")
        year_panel = year_panel.merge(put_25_short, on=group_keys, how="outer")
        year_panel = year_panel.merge(call_25_short, on=group_keys, how="outer")
        year_panel = year_panel.merge(quote_counts, on=group_keys, how="outer")
        year_panel["surface_term_slope_raw"] = year_panel["surface_atm_long_raw"] - year_panel["surface_atm_short_raw"]
        year_panel["surface_rr_25_short_raw"] = year_panel["surface_call_25_short_raw"] - year_panel["surface_put_25_short_raw"]
        year_panel["surface_bf_25_short_raw"] = (
            0.5 * (year_panel["surface_call_25_short_raw"] + year_panel["surface_put_25_short_raw"]) - year_panel["surface_atm_short_raw"]
        )
        per_year_panels.append(year_panel)

        quote_summary_rows.append(
            {
                "year_file": option_path.name,
                "raw_rows": int(len(frame)),
                "eligible_rows": int(len(eligible)),
                "surface_rows": int(len(year_panel)),
                "surface_complete_rows": int(len(year_panel.dropna(subset=SURFACE_FEATURES_RAW))),
            }
        )

    if not per_year_panels:
        raise RuntimeError("No yearly surface panels were created from the cached option data.")

    surface_panel = pd.concat(per_year_panels, ignore_index=True)
    surface_panel["trade_date"] = pd.to_datetime(surface_panel["trade_date"])
    surface_panel = surface_panel.sort_values(["permno", "trade_date"]).reset_index(drop=True)

    for raw_name, smooth_name in zip(SURFACE_FEATURES_RAW, SURFACE_FEATURES):
        surface_panel[smooth_name] = (
            surface_panel.groupby("permno", sort=False)[raw_name]
            .transform(lambda series: series.astype(float).rolling(window=5, min_periods=1).mean())
        )

    surface_panel.to_parquet(SURFACE_FACTOR_PANEL_PATH, index=False)

    stock_panel = _load_stock_panel()[["permno", "trade_date"] + STOCK_FEATURES + ["high_rv_regime"]]
    merged = stock_panel.merge(surface_panel[group_keys + SURFACE_FEATURES + ["surface_quote_count"]], on=["permno", "trade_date"], how="left")
    extension_panel = merged.dropna(subset=STOCK_FEATURES + SURFACE_FEATURES + ["high_rv_regime"]).reset_index(drop=True)
    extension_panel.to_parquet(SURFACE_EXTENSION_PANEL_PATH, index=False)

    summary = {
        "source": "cached_option_raw_files",
        "option_year_files": [path.name for path in _option_year_paths()],
        "total_raw_option_rows_seen": int(total_raw_rows),
        "total_eligible_option_rows_seen": int(total_eligible_rows),
        "surface_panel_rows": int(len(surface_panel)),
        "surface_complete_rows": int(len(surface_panel.dropna(subset=SURFACE_FEATURES_RAW))),
        "surface_extension_rows": int(len(extension_panel)),
        "feature_coverage_raw": {feature: float(surface_panel[feature].notna().mean()) for feature in SURFACE_FEATURES_RAW},
        "feature_coverage_smoothed": {feature: float(surface_panel[feature].notna().mean()) for feature in SURFACE_FEATURES},
        "surface_quote_count_summary": {
            "median": float(surface_panel["surface_quote_count"].median()),
            "p10": float(surface_panel["surface_quote_count"].quantile(0.10)),
            "p90": float(surface_panel["surface_quote_count"].quantile(0.90)),
        },
        "split_surface_extension_rows": {
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
        "yearly_quote_summary": quote_summary_rows,
    }
    write_json(BUILD_SURFACE_FACTORS_SUMMARY_PATH, summary)
    return summary


def train_surface_extension() -> dict[str, object]:
    if not SURFACE_FACTOR_PANEL_PATH.exists():
        build_surface_factors()

    stock_panel = _load_stock_panel()
    option_panel = _load_option_feature_panel()
    surface_panel = _load_surface_factor_panel()

    merged = stock_panel.merge(option_panel, on=["permno", "trade_date"], how="left")
    merged = merged.merge(surface_panel[["permno", "trade_date"] + SURFACE_FEATURES + ["surface_quote_count"]], on=["permno", "trade_date"], how="left")
    common_panel = merged.dropna(subset=STOCK_FEATURES + OPTION_FEATURES + SURFACE_FEATURES + ["high_rv_regime"]).reset_index(drop=True)
    if common_panel.empty:
        raise RuntimeError("The merged surface-extension common panel is empty.")
    common_panel.to_parquet(SURFACE_EXTENSION_PANEL_PATH, index=False)

    splits = _split_by_date(common_panel)
    feature_sets = [
        STOCK_FEATURES,
        OPTION_FEATURES,
        STOCK_FEATURES + OPTION_FEATURES,
        SURFACE_FEATURES,
        STOCK_FEATURES + SURFACE_FEATURES,
        STOCK_FEATURES + OPTION_FEATURES + SURFACE_FEATURES,
    ]
    X_train_cache = {tuple(columns): splits["train"][columns].to_numpy() for columns in feature_sets}
    X_validation_cache = {tuple(columns): splits["validation"][columns].to_numpy() for columns in feature_sets}
    X_test_cache = {tuple(columns): splits["test"][columns].to_numpy() for columns in feature_sets}
    y_train = splits["train"]["high_rv_regime"].astype(int).to_numpy()
    y_validation = splits["validation"]["high_rv_regime"].astype(int).tolist()
    y_test = splits["test"]["high_rv_regime"].astype(int).tolist()

    def train_logreg(model_name: str, feature_columns: list[str]):
        X_train = X_train_cache[tuple(feature_columns)]
        X_validation = X_validation_cache[tuple(feature_columns)]
        X_test = X_test_cache[tuple(feature_columns)]

        best_pipeline = None
        best_c = None
        best_score = None
        for c_value in CLASSICAL_C_GRID:
            pipeline = _fit_logreg_pipeline(X_train, y_train, c_value=c_value)
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

    def train_xgboost(model_name: str, feature_columns: list[str]):
        X_train = X_train_cache[tuple(feature_columns)]
        X_validation = X_validation_cache[tuple(feature_columns)]
        X_test = X_test_cache[tuple(feature_columns)]

        best_model = None
        best_params = None
        best_score = None
        for params in XGBOOST_PARAM_GRID:
            model = XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=SEED,
                n_jobs=4,
                **params,
            )
            model.fit(X_train, y_train)
            validation_prob = model.predict_proba(X_validation)[:, 1].tolist()
            validation_pred = model.predict(X_validation).tolist()
            metrics = compute_metrics(y_validation, validation_pred, validation_prob, PART1_LABEL_NAMES)
            score = (float(metrics["macro_f1"]), float(metrics["balanced_accuracy"]), float(metrics["accuracy"]))
            if best_score is None or score > best_score:
                best_model = model
                best_params = params
                best_score = score

        validation_prob = best_model.predict_proba(X_validation)[:, 1].tolist()
        validation_pred = best_model.predict(X_validation).tolist()
        test_prob = best_model.predict_proba(X_test)[:, 1].tolist()
        test_pred = best_model.predict(X_test).tolist()
        summary = {
            "model": model_name,
            "best_params": best_params,
            "feature_columns": feature_columns,
            "validation": compute_metrics(y_validation, validation_pred, validation_prob, PART1_LABEL_NAMES),
            "test": compute_metrics(y_test, test_pred, test_prob, PART1_LABEL_NAMES),
        }
        return best_model, summary

    logreg_specs = [
        ("stock_only_logreg_surface_common", STOCK_FEATURES),
        ("option_only_logreg_surface_common", OPTION_FEATURES),
        ("combined_original_logreg_surface_common", STOCK_FEATURES + OPTION_FEATURES),
        ("surface_only_logreg", SURFACE_FEATURES),
        ("stock_surface_logreg", STOCK_FEATURES + SURFACE_FEATURES),
        ("all_features_logreg", STOCK_FEATURES + OPTION_FEATURES + SURFACE_FEATURES),
    ]
    xgb_specs = [
        ("stock_only_xgb_surface_common", STOCK_FEATURES),
        ("option_only_xgb_surface_common", OPTION_FEATURES),
        ("all_features_xgb_surface_common", STOCK_FEATURES + OPTION_FEATURES + SURFACE_FEATURES),
    ]

    trained_models: dict[str, object] = {}
    model_summaries: dict[str, dict[str, object]] = {}
    for model_name, feature_columns in logreg_specs:
        trained_models[model_name], model_summaries[model_name] = train_logreg(model_name, feature_columns)
    for model_name, feature_columns in xgb_specs:
        trained_models[model_name], model_summaries[model_name] = train_xgboost(model_name, feature_columns)

    trailing_specs = [
        ("stock_only", STOCK_FEATURES),
        ("option_only", OPTION_FEATURES),
        ("all_features", STOCK_FEATURES + OPTION_FEATURES + SURFACE_FEATURES),
    ]
    trailing_predictions_map: dict[str, pd.DataFrame] = {}
    trailing_feature_columns: dict[str, list[str]] = {}
    for base_name, feature_columns in trailing_specs:
        logreg_key = {
            "stock_only": "stock_only_logreg_surface_common",
            "option_only": "option_only_logreg_surface_common",
            "all_features": "all_features_logreg",
        }[base_name]
        best_c = float(model_summaries[logreg_key]["best_c"])
        for trailing_years in TRAILING_WINDOW_YEARS:
            prediction_key = f"{base_name}_trail{trailing_years}y_logreg_surface_common"
            trailing_frame = _trailing_window_predictions(
                common_panel,
                feature_columns=feature_columns,
                c_value=best_c,
                trailing_years=trailing_years,
            )
            trailing_predictions_map[prediction_key] = trailing_frame
            trailing_feature_columns[prediction_key] = feature_columns
            for split_name in ("validation", "test"):
                split_frame = trailing_frame[trailing_frame["split"] == split_name].copy()
                labels = (
                    splits[split_name][["permno", "trade_date", "high_rv_regime"]]
                    .merge(split_frame[["permno", "trade_date", "probability", "prediction"]], on=["permno", "trade_date"], how="inner")
                    .sort_values(["trade_date", "permno"])
                )
                if labels.empty:
                    raise RuntimeError(f"Trailing benchmark {prediction_key} produced no {split_name} predictions.")
                model_summaries[prediction_key] = model_summaries.get(
                    prediction_key,
                    {
                        "model": prediction_key,
                        "best_c": best_c,
                        "feature_columns": feature_columns,
                        "trailing_years": trailing_years,
                        "retrain_months": TRAILING_RETRAIN_MONTHS,
                    },
                )
                model_summaries[prediction_key][split_name] = compute_metrics(
                    labels["high_rv_regime"].astype(int).tolist(),
                    labels["prediction"].astype(int).tolist(),
                    labels["probability"].astype(float).tolist(),
                    PART1_LABEL_NAMES,
                )

    joblib.dump(
        {"model_name": "surface_only_logreg", "pipeline": trained_models["surface_only_logreg"], "feature_columns": SURFACE_FEATURES},
        SURFACE_ONLY_MODEL_PATH,
    )
    joblib.dump(
        {"model_name": "stock_surface_logreg", "pipeline": trained_models["stock_surface_logreg"], "feature_columns": STOCK_FEATURES + SURFACE_FEATURES},
        STOCK_SURFACE_MODEL_PATH,
    )
    joblib.dump(
        {
            "model_name": "all_features_logreg",
            "pipeline": trained_models["all_features_logreg"],
            "feature_columns": STOCK_FEATURES + OPTION_FEATURES + SURFACE_FEATURES,
        },
        ALL_FEATURES_MODEL_PATH,
    )
    joblib.dump(
        {"model_name": "stock_only_xgb_surface_common", "pipeline": trained_models["stock_only_xgb_surface_common"], "feature_columns": STOCK_FEATURES},
        STOCK_ONLY_XGB_MODEL_PATH,
    )
    joblib.dump(
        {"model_name": "option_only_xgb_surface_common", "pipeline": trained_models["option_only_xgb_surface_common"], "feature_columns": OPTION_FEATURES},
        OPTION_ONLY_XGB_MODEL_PATH,
    )
    joblib.dump(
        {
            "model_name": "all_features_xgb_surface_common",
            "pipeline": trained_models["all_features_xgb_surface_common"],
            "feature_columns": STOCK_FEATURES + OPTION_FEATURES + SURFACE_FEATURES,
        },
        ALL_FEATURES_XGB_MODEL_PATH,
    )

    save_confusion_matrix_figure(
        y_test,
        trained_models["surface_only_logreg"].predict(splits["test"][SURFACE_FEATURES].to_numpy()).tolist(),
        PART1_LABEL_NAMES,
        SURFACE_ONLY_FIGURE,
        "Surface-only (Test)",
    )
    save_confusion_matrix_figure(
        y_test,
        trained_models["stock_surface_logreg"].predict(splits["test"][STOCK_FEATURES + SURFACE_FEATURES].to_numpy()).tolist(),
        PART1_LABEL_NAMES,
        STOCK_SURFACE_FIGURE,
        "Stock + Surface (Test)",
    )
    save_confusion_matrix_figure(
        y_test,
        trained_models["all_features_logreg"].predict(splits["test"][STOCK_FEATURES + OPTION_FEATURES + SURFACE_FEATURES].to_numpy()).tolist(),
        PART1_LABEL_NAMES,
        ALL_FEATURES_FIGURE,
        "All Features (Test)",
    )

    metric_rows = []
    for model_name, summary in model_summaries.items():
        for split_name in ("validation", "test"):
            metric_rows.append(
                {
                    "model": model_name,
                    "split": split_name,
                    "accuracy": summary[split_name]["accuracy"],
                    "macro_f1": summary[split_name]["macro_f1"],
                    "balanced_accuracy": summary[split_name]["balanced_accuracy"],
                    "auroc": summary[split_name]["auroc"],
                    "pr_auc": summary[split_name]["pr_auc"],
                }
            )

    prediction_rows = []
    prediction_feature_map = {
        "stock_only_prob": ("stock_only_logreg_surface_common", STOCK_FEATURES),
        "option_only_prob": ("option_only_logreg_surface_common", OPTION_FEATURES),
        "combined_original_prob": ("combined_original_logreg_surface_common", STOCK_FEATURES + OPTION_FEATURES),
        "surface_only_prob": ("surface_only_logreg", SURFACE_FEATURES),
        "stock_surface_prob": ("stock_surface_logreg", STOCK_FEATURES + SURFACE_FEATURES),
        "all_features_prob": ("all_features_logreg", STOCK_FEATURES + OPTION_FEATURES + SURFACE_FEATURES),
        "stock_only_xgb_prob": ("stock_only_xgb_surface_common", STOCK_FEATURES),
        "option_only_xgb_prob": ("option_only_xgb_surface_common", OPTION_FEATURES),
        "all_features_xgb_prob": ("all_features_xgb_surface_common", STOCK_FEATURES + OPTION_FEATURES + SURFACE_FEATURES),
    }
    trailing_prediction_columns = {
        "stock_only_trail2y_prob": "stock_only_trail2y_logreg_surface_common",
        "stock_only_trail5y_prob": "stock_only_trail5y_logreg_surface_common",
        "option_only_trail2y_prob": "option_only_trail2y_logreg_surface_common",
        "option_only_trail5y_prob": "option_only_trail5y_logreg_surface_common",
        "all_features_trail2y_prob": "all_features_trail2y_logreg_surface_common",
        "all_features_trail5y_prob": "all_features_trail5y_logreg_surface_common",
    }
    for split_name, split_frame in (("validation", splits["validation"]), ("test", splits["test"])):
        cached_predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for column_name, (model_name, feature_columns) in prediction_feature_map.items():
            pipeline = trained_models[model_name]
            X_split = split_frame[feature_columns].to_numpy()
            cached_predictions[column_name] = (
                pipeline.predict_proba(X_split)[:, 1],
                pipeline.predict(X_split),
            )
        trailing_split_frames = {
            column_name: trailing_predictions_map[prediction_key][trailing_predictions_map[prediction_key]["split"] == split_name]
            .sort_values(["trade_date", "permno"])
            .reset_index(drop=True)
            for column_name, prediction_key in trailing_prediction_columns.items()
        }
        split_lookup = split_frame[["permno", "trade_date"]].reset_index(drop=True)
        for column_name, trailing_frame in trailing_split_frames.items():
            aligned = split_lookup.merge(
                trailing_frame[["permno", "trade_date", "probability", "prediction"]],
                on=["permno", "trade_date"],
                how="left",
            )
            if aligned["probability"].isna().any():
                raise RuntimeError(f"{column_name} is missing aligned trailing predictions for {split_name}.")
            cached_predictions[column_name] = (
                aligned["probability"].to_numpy(),
                aligned["prediction"].astype(int).to_numpy(),
            )
        for idx, row in split_frame.reset_index(drop=True).iterrows():
            payload = {
                "permno": int(row["permno"]),
                "universe_ticker": row["universe_ticker"],
                "trade_date": row["trade_date"].date().isoformat(),
                "split": split_name,
                "label": int(row["high_rv_regime"]),
            }
            for column_name, (probabilities, predictions) in cached_predictions.items():
                payload[column_name] = float(probabilities[idx])
                payload[column_name.replace("_prob", "_pred")] = int(predictions[idx])
            prediction_rows.append(payload)

    write_rows_csv(SURFACE_EXTENSION_METRICS_CSV, metric_rows)
    write_rows_csv(SURFACE_EXTENSION_PREDICTIONS_CSV, prediction_rows)

    surface_classifier = trained_models["surface_only_logreg"].named_steps["classifier"]
    stock_surface_classifier = trained_models["stock_surface_logreg"].named_steps["classifier"]
    all_features_classifier = trained_models["all_features_logreg"].named_steps["classifier"]

    summary = {
        "split_sizes": {split_name: int(len(split_frame)) for split_name, split_frame in splits.items()},
        "models": model_summaries,
        "surface_only_coefficients": [
            {"feature": feature, "coefficient": float(coefficient)}
            for feature, coefficient in zip(SURFACE_FEATURES, surface_classifier.coef_[0])
        ],
        "stock_surface_coefficients": [
            {"feature": feature, "coefficient": float(coefficient)}
            for feature, coefficient in zip(STOCK_FEATURES + SURFACE_FEATURES, stock_surface_classifier.coef_[0])
        ],
        "all_features_coefficients": [
            {"feature": feature, "coefficient": float(coefficient)}
            for feature, coefficient in zip(STOCK_FEATURES + OPTION_FEATURES + SURFACE_FEATURES, all_features_classifier.coef_[0])
        ],
        "stock_only_xgb_feature_importance": [
            {"feature": feature, "importance": float(importance)}
            for feature, importance in zip(STOCK_FEATURES, trained_models["stock_only_xgb_surface_common"].feature_importances_)
        ],
        "option_only_xgb_feature_importance": [
            {"feature": feature, "importance": float(importance)}
            for feature, importance in zip(OPTION_FEATURES, trained_models["option_only_xgb_surface_common"].feature_importances_)
        ],
        "all_features_xgb_feature_importance": [
            {"feature": feature, "importance": float(importance)}
            for feature, importance in zip(
                STOCK_FEATURES + OPTION_FEATURES + SURFACE_FEATURES,
                trained_models["all_features_xgb_surface_common"].feature_importances_,
            )
        ],
        "trailing_benchmarks": {
            model_name: {
                "trailing_years": int(model_summaries[model_name]["trailing_years"]),
                "retrain_months": int(model_summaries[model_name]["retrain_months"]),
                "feature_columns": trailing_feature_columns[model_name],
            }
            for model_name in trailing_predictions_map
        },
        "rows": metric_rows,
    }
    write_json(SURFACE_EXTENSION_METRICS_JSON, summary)
    return summary
