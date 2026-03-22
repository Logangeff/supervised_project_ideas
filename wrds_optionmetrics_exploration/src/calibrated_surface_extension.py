from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import (
    ALL_EXTENSIONS_FIGURE,
    ALL_EXTENSIONS_MODEL_PATH,
    BUILD_CALIBRATED_SURFACE_SUMMARY_PATH,
    CALIBRATED_BETA_ONLY_FIGURE,
    CALIBRATED_BETA_ONLY_MODEL_PATH,
    CALIBRATED_SURFACE_EXTENSION_METRICS_CSV,
    CALIBRATED_SURFACE_EXTENSION_METRICS_JSON,
    CALIBRATED_SURFACE_EXTENSION_PANEL_PATH,
    CALIBRATED_SURFACE_EXTENSION_PREDICTIONS_CSV,
    CALIBRATED_SURFACE_FEATURES,
    CALIBRATED_SURFACE_FEATURES_RAW,
    CALIBRATED_SURFACE_PANEL_PATH,
    CLASSICAL_C_GRID,
    EXTRACT_CALIBRATED_SURFACE_INPUTS_SUMMARY_PATH,
    OPTION_BETA_FIGURE,
    OPTION_BETA_MODEL_PATH,
    OPTION_FEATURE_PANEL_PATH,
    OPTION_FEATURES,
    OPTION_SECURITY_LINK_PATH,
    PART1_LABEL_NAMES,
    PROJECT_END_DATE,
    PROJECT_START_DATE,
    RAW_OPTIONS_DIR,
    SEED,
    STOCK_FEATURES,
    STOCK_PANEL_PATH,
    SURFACE_FACTOR_PANEL_PATH,
    SURFACE_FEATURES,
    TRAIN_END_DATE,
    VALIDATION_END_DATE,
)
from .data_access import connect_wrds, execute_candidate_queries
from .evaluation import compute_metrics, save_confusion_matrix_figure
from .option_pipeline import extract_option_data
from .utils import write_json, write_rows_csv


MIN_CALIBRATION_QUOTES = 12
MIN_CALIBRATION_EXPIRIES = 3
MIN_DTE = 14
MAX_DTE = 365
MAX_ABS_MONEYNESS = 1.0
SMOOTHING_WINDOW = 5
SURFACE_T_MAX = 2.0


def _years() -> list[int]:
    return list(range(int(PROJECT_START_DATE[:4]), int(PROJECT_END_DATE[:4]) + 1))


def _calibration_quote_year_path(year: int) -> Path:
    return RAW_OPTIONS_DIR / f"optionm_calibration_quotes_{year}.parquet"


def _forward_price_year_path(year: int) -> Path:
    return RAW_OPTIONS_DIR / f"optionm_forward_prices_{year}.parquet"


def _beta_year_path(year: int) -> Path:
    return CALIBRATED_SURFACE_PANEL_PATH.parent / f"calibrated_surface_betas_{year}.parquet"


def _split_by_date(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    train = frame[frame["trade_date"] <= pd.Timestamp(TRAIN_END_DATE)].reset_index(drop=True)
    validation = frame[
        (frame["trade_date"] > pd.Timestamp(TRAIN_END_DATE)) & (frame["trade_date"] <= pd.Timestamp(VALIDATION_END_DATE))
    ].reset_index(drop=True)
    test = frame[(frame["trade_date"] > pd.Timestamp(VALIDATION_END_DATE)) & (frame["trade_date"] <= pd.Timestamp(PROJECT_END_DATE))].reset_index(drop=True)
    if train.empty or validation.empty or test.empty:
        raise RuntimeError("One of the chronological calibrated-surface splits is empty.")
    return {"train": train, "validation": validation, "test": test}


def _load_security_link() -> pd.DataFrame:
    if not OPTION_SECURITY_LINK_PATH.exists():
        extract_option_data()
    frame = pd.read_parquet(OPTION_SECURITY_LINK_PATH)
    return frame[["secid", "permno", "universe_ticker"]].drop_duplicates()


def _option_price_table_ref_by_year(connection) -> dict[int, str]:
    libraries = set(connection.list_libraries())
    refs: dict[int, str] = {}
    for library_name in ("optionm", "optionm_all"):
        if library_name not in libraries:
            continue
        tables = {str(name) for name in connection.list_tables(library=library_name)}
        for year in _years():
            table_name = f"opprcd{year}"
            if year not in refs and table_name in tables:
                refs[year] = f"{library_name}.{table_name}"
    return refs


def _forward_table_ref_by_year(connection) -> dict[int, str]:
    libraries = set(connection.list_libraries())
    refs: dict[int, str] = {}
    for library_name in ("optionm", "optionm_all"):
        if library_name not in libraries:
            continue
        tables = {str(name) for name in connection.list_tables(library=library_name)}
        for year in _years():
            table_name = f"fwdprd{year}"
            if year not in refs and table_name in tables:
                refs[year] = f"{library_name}.{table_name}"
    return refs


def _fetch_calibration_quotes_for_year(connection, table_ref: str, secid_list_sql: str, year: int) -> pd.DataFrame:
    sql_candidates = [
        f"""
        select
            secid,
            date::date as trade_date,
            exdate::date as exdate,
            cp_flag,
            strike_price,
            best_bid,
            best_offer,
            volume,
            open_interest,
            impl_volatility,
            delta,
            forward_price
        from {table_ref}
        where secid in ({secid_list_sql})
          and date between '{year}-01-01' and '{year}-12-31'
        """,
        f"""
        select
            secid,
            date::date as trade_date,
            exdate::date as exdate,
            cp_flag,
            strike_price,
            bid as best_bid,
            offer as best_offer,
            volume,
            open_interest,
            impl_volatility,
            delta,
            forward_price
        from {table_ref}
        where secid in ({secid_list_sql})
          and date between '{year}-01-01' and '{year}-12-31'
        """,
    ]
    return execute_candidate_queries(connection, sql_candidates)


def _fetch_forward_prices_for_year(connection, table_ref: str, secid_list_sql: str, year: int) -> pd.DataFrame:
    sql_candidates = [
        f"""
        select
            secid,
            date::date as trade_date,
            expiration::date as exdate,
            amsettlement as am_settlement,
            forwardprice
        from {table_ref}
        where secid in ({secid_list_sql})
          and date between '{year}-01-01' and '{year}-12-31'
        """,
        f"""
        select
            secid,
            date::date as trade_date,
            expiration::date as exdate,
            am_settlement,
            forwardprice
        from {table_ref}
        where secid in ({secid_list_sql})
          and date between '{year}-01-01' and '{year}-12-31'
        """,
    ]
    return execute_candidate_queries(connection, sql_candidates)


def extract_calibrated_surface_inputs() -> dict[str, object]:
    security_link = _load_security_link()
    quote_paths = [_calibration_quote_year_path(year) for year in _years()]
    forward_paths = [_forward_price_year_path(year) for year in _years()]

    if all(path.exists() for path in quote_paths) and all(path.exists() for path in forward_paths):
        quote_rows_by_year = {str(year): int(len(pd.read_parquet(_calibration_quote_year_path(year), columns=["secid"]))) for year in _years()}
        forward_rows_by_year = {str(year): int(len(pd.read_parquet(_forward_price_year_path(year), columns=["secid"]))) for year in _years()}
        summary = {
            "source": "cached_local_files",
            "secid_count": int(security_link["secid"].nunique()),
            "permno_count": int(security_link["permno"].nunique()),
            "quote_rows_by_year": quote_rows_by_year,
            "forward_rows_by_year": forward_rows_by_year,
            "quote_files": [path.name for path in quote_paths],
            "forward_files": [path.name for path in forward_paths],
        }
        write_json(EXTRACT_CALIBRATED_SURFACE_INPUTS_SUMMARY_PATH, summary)
        return summary

    connection = connect_wrds()
    price_refs = _option_price_table_ref_by_year(connection)
    forward_refs = _forward_table_ref_by_year(connection)
    secid_list_sql = ", ".join(str(int(value)) for value in sorted(security_link["secid"].dropna().astype(int).unique()))

    quote_rows_by_year: dict[str, int] = {}
    forward_rows_by_year: dict[str, int] = {}
    quote_files: list[str] = []
    forward_files: list[str] = []

    for year in _years():
        quote_path = _calibration_quote_year_path(year)
        if quote_path.exists():
            quote_frame = pd.read_parquet(quote_path)
        else:
            table_ref = price_refs.get(year)
            if table_ref is None:
                raise RuntimeError(f"Could not find an OptionMetrics option-price table for {year}.")
            print(f"Querying calibration quotes from {table_ref} for {year}...")
            quote_frame = _fetch_calibration_quotes_for_year(connection, table_ref, secid_list_sql, year)
            quote_frame["trade_date"] = pd.to_datetime(quote_frame["trade_date"])
            quote_frame["exdate"] = pd.to_datetime(quote_frame["exdate"])
            quote_frame.to_parquet(quote_path, index=False)
        quote_rows_by_year[str(year)] = int(len(quote_frame))
        quote_files.append(quote_path.name)

        forward_path = _forward_price_year_path(year)
        if forward_path.exists():
            forward_frame = pd.read_parquet(forward_path)
        else:
            table_ref = forward_refs.get(year)
            if table_ref is None:
                raise RuntimeError(f"Could not find an OptionMetrics forward-price table for {year}.")
            print(f"Querying forward prices from {table_ref} for {year}...")
            forward_frame = _fetch_forward_prices_for_year(connection, table_ref, secid_list_sql, year)
            forward_frame["trade_date"] = pd.to_datetime(forward_frame["trade_date"])
            forward_frame["exdate"] = pd.to_datetime(forward_frame["exdate"])
            forward_frame.to_parquet(forward_path, index=False)
        forward_rows_by_year[str(year)] = int(len(forward_frame))
        forward_files.append(forward_path.name)

    summary = {
        "source": "wrds_query",
        "secid_count": int(security_link["secid"].nunique()),
        "permno_count": int(security_link["permno"].nunique()),
        "quote_rows_by_year": quote_rows_by_year,
        "forward_rows_by_year": forward_rows_by_year,
        "quote_files": quote_files,
        "forward_files": forward_files,
    }
    write_json(EXTRACT_CALIBRATED_SURFACE_INPUTS_SUMMARY_PATH, summary)
    return summary


def _skew_basis(moneyness: np.ndarray) -> np.ndarray:
    return np.where(moneyness >= 0.0, moneyness, np.tanh(moneyness))


def _surface_design_matrix(moneyness: np.ndarray, tau: np.ndarray) -> np.ndarray:
    tau_clip = np.clip(tau, 1e-6, None)
    curvature = 1.0 - np.exp(-(moneyness**2))
    term_decay = np.exp(-np.sqrt(tau_clip / 0.25))
    interaction = curvature * np.log(np.clip(tau_clip / SURFACE_T_MAX, 1e-6, None))
    return np.column_stack(
        [
            np.ones_like(moneyness),
            term_decay,
            _skew_basis(moneyness),
            curvature,
            interaction,
        ]
    )


def _maybe_rescale_price_series(series: pd.Series) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce")
    non_null = clean.dropna()
    if non_null.empty:
        return clean
    if float(non_null.median()) > 1000:
        return clean / 1000.0
    return clean


def _fit_group_surface(group: pd.DataFrame) -> dict[str, object] | None:
    if len(group) < MIN_CALIBRATION_QUOTES:
        return None
    expiry_count = int(group["exdate"].nunique())
    if expiry_count < MIN_CALIBRATION_EXPIRIES:
        return None

    moneyness = group["M"].to_numpy(dtype=float)
    tau = group["tau"].to_numpy(dtype=float)
    sigma = group["impl_volatility"].to_numpy(dtype=float)
    mask = np.isfinite(moneyness) & np.isfinite(tau) & np.isfinite(sigma)
    if int(mask.sum()) < MIN_CALIBRATION_QUOTES:
        return None

    moneyness = moneyness[mask]
    tau = tau[mask]
    sigma = sigma[mask]
    design = _surface_design_matrix(moneyness, tau)
    if np.linalg.matrix_rank(design) < len(CALIBRATED_SURFACE_FEATURES):
        return None

    betas, _, _, _ = np.linalg.lstsq(design, sigma, rcond=None)
    fitted = design @ betas
    ss_res = float(np.sum((sigma - fitted) ** 2))
    ss_tot = float(np.sum((sigma - sigma.mean()) ** 2))
    r_squared = float(1.0 - (ss_res / (ss_tot + 1e-12)))
    rmse = float(np.sqrt(np.mean((sigma - fitted) ** 2)))

    return {
        "permno": int(group["permno"].iloc[0]),
        "universe_ticker": group["universe_ticker"].iloc[0],
        "trade_date": pd.Timestamp(group["trade_date"].iloc[0]),
        "surface_beta_1_raw": float(betas[0]),
        "surface_beta_2_raw": float(betas[1]),
        "surface_beta_3_raw": float(betas[2]),
        "surface_beta_4_raw": float(betas[3]),
        "surface_beta_5_raw": float(betas[4]),
        "surface_beta_r2": r_squared,
        "surface_beta_rmse": rmse,
        "surface_beta_quote_count": int(mask.sum()),
        "surface_beta_expiry_count": expiry_count,
    }


def build_calibrated_surface() -> dict[str, object]:
    quote_paths = [_calibration_quote_year_path(year) for year in _years()]
    forward_paths = [_forward_price_year_path(year) for year in _years()]
    if not all(path.exists() for path in quote_paths) or not all(path.exists() for path in forward_paths):
        extract_calibrated_surface_inputs()

    security_link = _load_security_link()
    per_year_panels: list[pd.DataFrame] = []
    yearly_rows: list[dict[str, object]] = []
    total_quote_rows = 0
    total_filtered_rows = 0
    total_otm_rows = 0

    for year in _years():
        beta_year_path = _beta_year_path(year)
        if beta_year_path.exists():
            year_panel = pd.read_parquet(beta_year_path)
            year_panel["trade_date"] = pd.to_datetime(year_panel["trade_date"])
            per_year_panels.append(year_panel)
            yearly_rows.append(
                {
                    "year": year,
                    "raw_quote_rows": None,
                    "filtered_quote_rows": None,
                    "otm_rows": None,
                    "beta_rows": int(len(year_panel)),
                }
            )
            continue

        print(f"Building calibrated-surface betas for {year}...")
        quotes = pd.read_parquet(_calibration_quote_year_path(year))
        forwards = pd.read_parquet(_forward_price_year_path(year))
        quotes = quotes.merge(security_link, on="secid", how="inner")
        quotes["trade_date"] = pd.to_datetime(quotes["trade_date"])
        quotes["exdate"] = pd.to_datetime(quotes["exdate"])
        forwards["trade_date"] = pd.to_datetime(forwards["trade_date"])
        forwards["exdate"] = pd.to_datetime(forwards["exdate"])
        total_quote_rows += int(len(quotes))

        for column in ("strike_price", "best_bid", "best_offer", "volume", "open_interest", "impl_volatility", "delta", "forward_price"):
            if column in quotes.columns:
                quotes[column] = pd.to_numeric(quotes[column], errors="coerce")
        forwards["forwardprice"] = pd.to_numeric(forwards["forwardprice"], errors="coerce")

        quotes["cp_flag"] = quotes["cp_flag"].astype(str).str.upper().str[0]
        quotes["strike"] = _maybe_rescale_price_series(quotes["strike_price"])
        if "forward_price" in quotes.columns:
            quotes["forward_price"] = _maybe_rescale_price_series(quotes["forward_price"])
        forwards["forwardprice"] = _maybe_rescale_price_series(forwards["forwardprice"])

        merged = quotes.merge(
            forwards[["secid", "trade_date", "exdate", "forwardprice"]],
            on=["secid", "trade_date", "exdate"],
            how="left",
        )
        merged["forward_price_used"] = merged["forwardprice"].fillna(merged.get("forward_price"))
        merged["dte"] = (merged["exdate"] - merged["trade_date"]).dt.days
        merged["tau"] = merged["dte"] / 365.0
        merged["M"] = np.log(merged["forward_price_used"] / merged["strike"]) / np.sqrt(np.clip(merged["tau"], 1e-6, None))

        filtered = merged[
            merged["impl_volatility"].gt(0)
            & merged["open_interest"].gt(0)
            & merged["best_bid"].gt(0)
            & merged["best_offer"].gt(0)
            & merged["best_bid"].le(merged["best_offer"])
            & merged["delta"].notna()
            & merged["strike"].gt(0)
            & merged["forward_price_used"].gt(0)
            & merged["dte"].between(MIN_DTE, MAX_DTE)
            & merged["cp_flag"].isin(["C", "P"])
            & merged["M"].abs().le(MAX_ABS_MONEYNESS)
        ].copy()
        total_filtered_rows += int(len(filtered))

        otm = filtered[
            ((filtered["cp_flag"] == "P") & (filtered["strike"] < filtered["forward_price_used"]))
            | ((filtered["cp_flag"] == "C") & (filtered["strike"] >= filtered["forward_price_used"]))
        ].copy()
        total_otm_rows += int(len(otm))

        rows = []
        for _, group in otm.groupby(["permno", "universe_ticker", "trade_date"], sort=False):
            payload = _fit_group_surface(group)
            if payload is not None:
                rows.append(payload)

        year_panel = pd.DataFrame(rows)
        if not year_panel.empty:
            year_panel["trade_date"] = pd.to_datetime(year_panel["trade_date"])
            year_panel.to_parquet(beta_year_path, index=False)
            per_year_panels.append(year_panel)

        yearly_rows.append(
            {
                "year": year,
                "raw_quote_rows": int(len(quotes)),
                "filtered_quote_rows": int(len(filtered)),
                "otm_rows": int(len(otm)),
                "beta_rows": int(len(year_panel)),
            }
        )

    if not per_year_panels:
        raise RuntimeError("No calibrated-surface beta rows were created.")

    calibrated_panel = pd.concat(per_year_panels, ignore_index=True)
    calibrated_panel["trade_date"] = pd.to_datetime(calibrated_panel["trade_date"])
    calibrated_panel = calibrated_panel.sort_values(["permno", "trade_date"]).reset_index(drop=True)

    for raw_name, smooth_name in zip(CALIBRATED_SURFACE_FEATURES_RAW, CALIBRATED_SURFACE_FEATURES):
        calibrated_panel[smooth_name] = (
            calibrated_panel.groupby("permno", sort=False)[raw_name]
            .transform(lambda series: series.astype(float).rolling(window=SMOOTHING_WINDOW, min_periods=1).mean())
        )

    calibrated_panel.to_parquet(CALIBRATED_SURFACE_PANEL_PATH, index=False)

    summary = {
        "source": "cached_or_wrds_calibration_inputs",
        "beta_panel_rows": int(len(calibrated_panel)),
        "permno_count": int(calibrated_panel["permno"].nunique()),
        "date_range": {
            "min_date": str(calibrated_panel["trade_date"].min().date()),
            "max_date": str(calibrated_panel["trade_date"].max().date()),
        },
        "total_quote_rows_seen": int(total_quote_rows) if total_quote_rows else None,
        "total_filtered_rows": int(total_filtered_rows) if total_filtered_rows else None,
        "total_otm_rows": int(total_otm_rows) if total_otm_rows else None,
        "feature_coverage_raw": {feature: float(calibrated_panel[feature].notna().mean()) for feature in CALIBRATED_SURFACE_FEATURES_RAW},
        "feature_coverage_smoothed": {feature: float(calibrated_panel[feature].notna().mean()) for feature in CALIBRATED_SURFACE_FEATURES},
        "surface_beta_r2_summary": {
            "median": float(calibrated_panel["surface_beta_r2"].median()),
            "p10": float(calibrated_panel["surface_beta_r2"].quantile(0.10)),
            "p90": float(calibrated_panel["surface_beta_r2"].quantile(0.90)),
        },
        "quote_count_summary": {
            "median": float(calibrated_panel["surface_beta_quote_count"].median()),
            "p10": float(calibrated_panel["surface_beta_quote_count"].quantile(0.10)),
            "p90": float(calibrated_panel["surface_beta_quote_count"].quantile(0.90)),
        },
        "yearly_summary": yearly_rows,
    }
    write_json(BUILD_CALIBRATED_SURFACE_SUMMARY_PATH, summary)
    return summary


def _load_stock_panel() -> pd.DataFrame:
    frame = pd.read_parquet(STOCK_PANEL_PATH)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    return frame


def _load_option_feature_panel() -> pd.DataFrame:
    frame = pd.read_parquet(OPTION_FEATURE_PANEL_PATH)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    return frame


def _load_surface_factor_panel() -> pd.DataFrame:
    frame = pd.read_parquet(SURFACE_FACTOR_PANEL_PATH)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    return frame


def _load_calibrated_surface_panel() -> pd.DataFrame:
    frame = pd.read_parquet(CALIBRATED_SURFACE_PANEL_PATH)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    return frame


def train_calibrated_surface_extension() -> dict[str, object]:
    if not CALIBRATED_SURFACE_PANEL_PATH.exists():
        build_calibrated_surface()

    stock_panel = _load_stock_panel()
    option_panel = _load_option_feature_panel()
    surface_panel = _load_surface_factor_panel()
    beta_panel = _load_calibrated_surface_panel()

    merged = stock_panel.merge(option_panel, on=["permno", "trade_date"], how="left")
    merged = merged.merge(surface_panel[["permno", "trade_date"] + SURFACE_FEATURES], on=["permno", "trade_date"], how="left")
    merged = merged.merge(
        beta_panel[["permno", "trade_date"] + CALIBRATED_SURFACE_FEATURES + ["surface_beta_r2", "surface_beta_quote_count", "surface_beta_expiry_count"]],
        on=["permno", "trade_date"],
        how="left",
    )
    common_panel = merged.dropna(subset=STOCK_FEATURES + OPTION_FEATURES + SURFACE_FEATURES + CALIBRATED_SURFACE_FEATURES + ["high_rv_regime"]).reset_index(drop=True)
    if common_panel.empty:
        raise RuntimeError("The calibrated-surface common panel is empty.")
    common_panel.to_parquet(CALIBRATED_SURFACE_EXTENSION_PANEL_PATH, index=False)

    splits = _split_by_date(common_panel)

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

    model_specs = [
        ("stock_only_logreg_calibrated_common", STOCK_FEATURES),
        ("option_only_logreg_calibrated_common", OPTION_FEATURES),
        ("surface_only_logreg_calibrated_common", SURFACE_FEATURES),
        ("beta_only_logreg", CALIBRATED_SURFACE_FEATURES),
        ("option_beta_logreg", OPTION_FEATURES + CALIBRATED_SURFACE_FEATURES),
        ("all_extensions_logreg", STOCK_FEATURES + OPTION_FEATURES + SURFACE_FEATURES + CALIBRATED_SURFACE_FEATURES),
    ]

    trained_models: dict[str, Pipeline] = {}
    model_summaries: dict[str, dict[str, object]] = {}
    for model_name, feature_columns in model_specs:
        trained_models[model_name], model_summaries[model_name] = train_logreg(model_name, feature_columns)

    joblib.dump(
        {"model_name": "beta_only_logreg", "pipeline": trained_models["beta_only_logreg"], "feature_columns": CALIBRATED_SURFACE_FEATURES},
        CALIBRATED_BETA_ONLY_MODEL_PATH,
    )
    joblib.dump(
        {"model_name": "option_beta_logreg", "pipeline": trained_models["option_beta_logreg"], "feature_columns": OPTION_FEATURES + CALIBRATED_SURFACE_FEATURES},
        OPTION_BETA_MODEL_PATH,
    )
    joblib.dump(
        {
            "model_name": "all_extensions_logreg",
            "pipeline": trained_models["all_extensions_logreg"],
            "feature_columns": STOCK_FEATURES + OPTION_FEATURES + SURFACE_FEATURES + CALIBRATED_SURFACE_FEATURES,
        },
        ALL_EXTENSIONS_MODEL_PATH,
    )

    y_test = splits["test"]["high_rv_regime"].astype(int).tolist()
    save_confusion_matrix_figure(
        y_test,
        trained_models["beta_only_logreg"].predict(splits["test"][CALIBRATED_SURFACE_FEATURES].to_numpy()).tolist(),
        PART1_LABEL_NAMES,
        CALIBRATED_BETA_ONLY_FIGURE,
        "Calibrated Beta-only (Test)",
    )
    save_confusion_matrix_figure(
        y_test,
        trained_models["option_beta_logreg"].predict(splits["test"][OPTION_FEATURES + CALIBRATED_SURFACE_FEATURES].to_numpy()).tolist(),
        PART1_LABEL_NAMES,
        OPTION_BETA_FIGURE,
        "Option + Calibrated Beta (Test)",
    )
    save_confusion_matrix_figure(
        y_test,
        trained_models["all_extensions_logreg"].predict(splits["test"][STOCK_FEATURES + OPTION_FEATURES + SURFACE_FEATURES + CALIBRATED_SURFACE_FEATURES].to_numpy()).tolist(),
        PART1_LABEL_NAMES,
        ALL_EXTENSIONS_FIGURE,
        "All Extensions (Test)",
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
        "stock_only_prob": ("stock_only_logreg_calibrated_common", STOCK_FEATURES),
        "option_only_prob": ("option_only_logreg_calibrated_common", OPTION_FEATURES),
        "surface_only_prob": ("surface_only_logreg_calibrated_common", SURFACE_FEATURES),
        "beta_only_prob": ("beta_only_logreg", CALIBRATED_SURFACE_FEATURES),
        "option_beta_prob": ("option_beta_logreg", OPTION_FEATURES + CALIBRATED_SURFACE_FEATURES),
        "all_extensions_prob": ("all_extensions_logreg", STOCK_FEATURES + OPTION_FEATURES + SURFACE_FEATURES + CALIBRATED_SURFACE_FEATURES),
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

    write_rows_csv(CALIBRATED_SURFACE_EXTENSION_METRICS_CSV, metric_rows)
    write_rows_csv(CALIBRATED_SURFACE_EXTENSION_PREDICTIONS_CSV, prediction_rows)

    beta_classifier = trained_models["beta_only_logreg"].named_steps["classifier"]
    option_beta_classifier = trained_models["option_beta_logreg"].named_steps["classifier"]
    all_extensions_classifier = trained_models["all_extensions_logreg"].named_steps["classifier"]
    summary = {
        "split_sizes": {split_name: int(len(split_frame)) for split_name, split_frame in splits.items()},
        "models": model_summaries,
        "beta_only_coefficients": [
            {"feature": feature, "coefficient": float(coefficient)}
            for feature, coefficient in zip(CALIBRATED_SURFACE_FEATURES, beta_classifier.coef_[0])
        ],
        "option_beta_coefficients": [
            {"feature": feature, "coefficient": float(coefficient)}
            for feature, coefficient in zip(OPTION_FEATURES + CALIBRATED_SURFACE_FEATURES, option_beta_classifier.coef_[0])
        ],
        "all_extensions_coefficients": [
            {"feature": feature, "coefficient": float(coefficient)}
            for feature, coefficient in zip(STOCK_FEATURES + OPTION_FEATURES + SURFACE_FEATURES + CALIBRATED_SURFACE_FEATURES, all_extensions_classifier.coef_[0])
        ],
        "rows": metric_rows,
    }
    write_json(CALIBRATED_SURFACE_EXTENSION_METRICS_JSON, summary)
    return summary
