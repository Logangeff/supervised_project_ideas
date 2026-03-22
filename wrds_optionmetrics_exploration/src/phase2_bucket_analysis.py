from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import (
    CALIBRATED_SURFACE_EXTENSION_PREDICTIONS_CSV,
    PHASE2_BUCKET_DIAGNOSTICS_CSV,
    PHASE2_BUCKET_EXTENSION_DIAGNOSTICS_CSV,
    PHASE2_BUCKET_EXTENSION_METRICS_CSV,
    PHASE2_BUCKET_EXTENSION_METRICS_JSON,
    PHASE2_BUCKET_FUTURE_RV_FIGURE,
    PHASE2_BUCKET_HIT_RATE_FIGURE,
    PHASE2_BUCKET_METRICS_CSV,
    PHASE2_BUCKET_METRICS_JSON,
    PHASE2_BUCKET_RANK_IC_FIGURE,
    STOCK_PANEL_PATH,
    SURFACE_EXTENSION_PREDICTIONS_CSV,
)
from .utils import write_json, write_rows_csv


BUCKET_COUNT = 5

CORE_SIGNALS = [
    "stock_only_prob",
    "option_only_prob",
    "all_features_prob",
    "stock_only_xgb_prob",
    "option_only_xgb_prob",
    "all_features_xgb_prob",
    "stock_only_trail2y_prob",
    "stock_only_trail5y_prob",
    "option_only_trail2y_prob",
    "option_only_trail5y_prob",
    "all_features_trail2y_prob",
    "all_features_trail5y_prob",
]
CORE_SIGNALS_ORIGINAL = [
    "stock_only_prob",
    "option_only_prob",
    "all_features_prob",
]

EXTENSION_SIGNALS = [
    "beta_only_prob",
    "option_beta_prob",
    "all_extensions_prob",
]

SIGNAL_LABELS = {
    "stock_only_prob": "Stock-only probability",
    "option_only_prob": "Option-only probability",
    "all_features_prob": "All-features probability",
    "stock_only_xgb_prob": "Stock-only probability (XGBoost)",
    "option_only_xgb_prob": "Option-only probability (XGBoost)",
    "all_features_xgb_prob": "All-features probability (XGBoost)",
    "stock_only_trail2y_prob": "Stock-only probability (2y trailing logreg)",
    "stock_only_trail5y_prob": "Stock-only probability (5y trailing logreg)",
    "option_only_trail2y_prob": "Option-only probability (2y trailing logreg)",
    "option_only_trail5y_prob": "Option-only probability (5y trailing logreg)",
    "all_features_trail2y_prob": "All-features probability (2y trailing logreg)",
    "all_features_trail5y_prob": "All-features probability (5y trailing logreg)",
    "beta_only_prob": "Beta-only probability",
    "option_beta_prob": "Option+beta probability",
    "all_extensions_prob": "All-extensions probability",
}

SIGNAL_COLORS = {
    "stock_only_prob": "#d95f02",
    "option_only_prob": "#1b9e77",
    "all_features_prob": "#7570b3",
    "stock_only_xgb_prob": "#a6761d",
    "option_only_xgb_prob": "#1f78b4",
    "all_features_xgb_prob": "#e7298a",
    "stock_only_trail2y_prob": "#8c564b",
    "stock_only_trail5y_prob": "#c49c94",
    "option_only_trail2y_prob": "#17becf",
    "option_only_trail5y_prob": "#9edae5",
    "all_features_trail2y_prob": "#9467bd",
    "all_features_trail5y_prob": "#c5b0d5",
    "beta_only_prob": "#e6ab02",
    "option_beta_prob": "#66a61e",
    "all_extensions_prob": "#1f78b4",
}


def _load_stock_panel() -> pd.DataFrame:
    pickle_path = STOCK_PANEL_PATH.with_suffix(".pkl")
    if pickle_path.exists():
        frame = pd.read_pickle(pickle_path)
    elif STOCK_PANEL_PATH.exists():
        frame = pd.read_parquet(STOCK_PANEL_PATH)
    else:
        raise RuntimeError("Stock panel is missing. Run build_stock_panel first.")
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    return frame


def _load_predictions(path: Path, signal_columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise RuntimeError(f"Prediction file is missing: {path.name}")
    required_columns = ["permno", "universe_ticker", "trade_date", "split", "label"] + signal_columns
    frame = pd.read_csv(path, parse_dates=["trade_date"])
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise RuntimeError(f"Prediction file {path.name} is missing required columns: {missing}")
    frame = frame[frame["split"].isin(["validation", "test"])].copy()
    if frame.empty:
        raise RuntimeError(f"Prediction file {path.name} has no validation/test rows.")
    return frame


def _prepare_bucket_panel(predictions: pd.DataFrame) -> pd.DataFrame:
    stock_panel = _load_stock_panel()[["permno", "trade_date", "future_rv_20d", "high_rv_regime", "ret"]].copy()
    stock_panel = stock_panel.sort_values(["permno", "trade_date"]).reset_index(drop=True)
    stock_panel["ret_next_1d"] = stock_panel.groupby("permno", sort=False)["ret"].shift(-1)
    stock_panel["abs_ret_next_1d"] = stock_panel["ret_next_1d"].abs()

    panel = predictions.merge(
        stock_panel[["permno", "trade_date", "future_rv_20d", "high_rv_regime", "ret_next_1d", "abs_ret_next_1d"]],
        on=["permno", "trade_date"],
        how="left",
    )
    panel = panel.dropna(subset=["future_rv_20d", "high_rv_regime", "ret_next_1d", "abs_ret_next_1d"]).reset_index(drop=True)
    return panel


def _assign_bucket(series: pd.Series) -> pd.Series:
    ranks = series.rank(method="first", pct=True)
    buckets = np.ceil(ranks * BUCKET_COUNT).astype(int)
    return pd.Series(np.clip(buckets, 1, BUCKET_COUNT), index=series.index, dtype=int)


def _date_rank_correlation(frame: pd.DataFrame, left: str, right: str) -> float | None:
    if frame[left].nunique() <= 1 or frame[right].nunique() <= 1:
        return None
    value = frame[[left, right]].corr(method="spearman").iloc[0, 1]
    return float(value) if pd.notna(value) else None


def _analyze_signal(
    panel: pd.DataFrame,
    signal_column: str,
    comparison_set: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    signal_panel = panel[["permno", "trade_date", "split", signal_column, "future_rv_20d", "high_rv_regime", "ret_next_1d", "abs_ret_next_1d"]].copy()
    signal_panel[signal_column] = pd.to_numeric(signal_panel[signal_column], errors="coerce")
    signal_panel = signal_panel.dropna(subset=[signal_column]).reset_index(drop=True)
    signal_panel["bucket"] = signal_panel.groupby(["split", "trade_date"], sort=False)[signal_column].transform(_assign_bucket)

    bucket_frame = (
        signal_panel.groupby(["split", "bucket"], sort=True)
        .agg(
            observation_count=("permno", "count"),
            average_signal_probability=(signal_column, "mean"),
            average_future_rv_20d=("future_rv_20d", "mean"),
            high_rv_hit_rate=("high_rv_regime", "mean"),
            average_ret_next_1d=("ret_next_1d", "mean"),
            average_abs_ret_next_1d=("abs_ret_next_1d", "mean"),
        )
        .reset_index()
    )
    bucket_rows = [
        {
            "comparison_set": comparison_set,
            "signal_column": signal_column,
            **row._asdict(),
        }
        for row in bucket_frame.itertuples(index=False)
    ]

    diagnostics_rows: list[dict[str, object]] = []
    split_summary: dict[str, object] = {}

    for split_name, split_frame in signal_panel.groupby("split", sort=False):
        daily_rank_ic_future_rv: list[float] = []
        daily_rank_ic_label: list[float] = []
        daily_future_rv_spreads: list[float] = []
        daily_hit_rate_spreads: list[float] = []
        daily_abs_return_spreads: list[float] = []
        monotonic_future_rv_flags: list[bool] = []
        monotonic_hit_rate_flags: list[bool] = []

        for trade_date, date_frame in split_frame.groupby("trade_date", sort=True):
            grouped = (
                date_frame.groupby("bucket", sort=True)
                .agg(
                    average_future_rv_20d=("future_rv_20d", "mean"),
                    high_rv_hit_rate=("high_rv_regime", "mean"),
                    average_abs_ret_next_1d=("abs_ret_next_1d", "mean"),
                )
                .reindex(range(1, BUCKET_COUNT + 1))
            )
            top_bucket = grouped.iloc[-1]
            bottom_bucket = grouped.iloc[0]

            future_rv_ic = _date_rank_correlation(date_frame, signal_column, "future_rv_20d")
            label_ic = _date_rank_correlation(date_frame, signal_column, "high_rv_regime")
            if future_rv_ic is not None:
                daily_rank_ic_future_rv.append(future_rv_ic)
            if label_ic is not None:
                daily_rank_ic_label.append(label_ic)

            future_rv_spread = float(top_bucket["average_future_rv_20d"] - bottom_bucket["average_future_rv_20d"])
            hit_rate_spread = float(top_bucket["high_rv_hit_rate"] - bottom_bucket["high_rv_hit_rate"])
            abs_return_spread = float(top_bucket["average_abs_ret_next_1d"] - bottom_bucket["average_abs_ret_next_1d"])
            monotonic_future_rv = bool((grouped["average_future_rv_20d"].diff().dropna() >= -1e-12).all())
            monotonic_hit_rate = bool((grouped["high_rv_hit_rate"].diff().dropna() >= -1e-12).all())

            daily_future_rv_spreads.append(future_rv_spread)
            daily_hit_rate_spreads.append(hit_rate_spread)
            daily_abs_return_spreads.append(abs_return_spread)
            monotonic_future_rv_flags.append(monotonic_future_rv)
            monotonic_hit_rate_flags.append(monotonic_hit_rate)

            diagnostics_rows.append(
                {
                    "comparison_set": comparison_set,
                    "signal_column": signal_column,
                    "split": split_name,
                    "trade_date": trade_date.date().isoformat(),
                    "daily_rank_ic_future_rv": future_rv_ic,
                    "daily_rank_ic_label": label_ic,
                    "top_bottom_future_rv_spread": future_rv_spread,
                    "top_bottom_hit_rate_spread": hit_rate_spread,
                    "top_bottom_abs_return_spread": abs_return_spread,
                    "monotonic_future_rv": monotonic_future_rv,
                    "monotonic_hit_rate": monotonic_hit_rate,
                }
            )

        split_summary[split_name] = {
            "top_bottom_future_rv_spread": float(np.mean(daily_future_rv_spreads)),
            "top_bottom_hit_rate_spread": float(np.mean(daily_hit_rate_spreads)),
            "top_bottom_abs_return_spread": float(np.mean(daily_abs_return_spreads)),
            "average_daily_rank_ic_future_rv": float(np.mean(daily_rank_ic_future_rv)),
            "average_daily_rank_ic_label": float(np.mean(daily_rank_ic_label)),
            "monotonic_future_rv_share": float(np.mean(monotonic_future_rv_flags)),
            "monotonic_hit_rate_share": float(np.mean(monotonic_hit_rate_flags)),
            "bucket_rows": [
                row
                for row in bucket_rows
                if row["split"] == split_name
            ],
        }

    signal_summary = {
        "signal_column": signal_column,
        "comparison_set": comparison_set,
        "splits": split_summary,
    }
    return bucket_rows, diagnostics_rows, signal_summary


def _run_bucket_set(
    predictions_path: Path,
    signal_columns: list[str],
    comparison_set: str,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    predictions = _load_predictions(predictions_path, signal_columns)
    panel = _prepare_bucket_panel(predictions)
    if panel.empty:
        raise RuntimeError(f"No usable bucket-analysis rows were available after merging {predictions_path.name}.")

    bucket_rows: list[dict[str, object]] = []
    diagnostics_rows: list[dict[str, object]] = []
    signal_summaries: dict[str, object] = {}

    for signal_column in signal_columns:
        signal_bucket_rows, signal_diagnostics_rows, signal_summary = _analyze_signal(
            panel,
            signal_column=signal_column,
            comparison_set=comparison_set,
        )
        bucket_rows.extend(signal_bucket_rows)
        diagnostics_rows.extend(signal_diagnostics_rows)
        signal_summaries[signal_column] = signal_summary

    summary = {
        "comparison_set": comparison_set,
        "prediction_file": predictions_path.name,
        "bucket_count": BUCKET_COUNT,
        "usable_rows": int(len(panel)),
        "split_row_counts": {
            split_name: int(len(panel[panel["split"] == split_name])) for split_name in ("validation", "test")
        },
        "signals": signal_summaries,
        "rows": bucket_rows,
    }
    return summary, bucket_rows, diagnostics_rows


def _plot_bucket_curves(
    bucket_rows: list[dict[str, object]],
    value_column: str,
    output_path: Path,
    title: str,
    ylabel: str,
    signal_subset: list[str],
) -> None:
    frame = pd.DataFrame(bucket_rows)
    if frame.empty:
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for axis, split_name in zip(axes, ["validation", "test"]):
        split_frame = frame[(frame["comparison_set"] == "surface_common") & (frame["split"] == split_name)]
        for signal_column in signal_subset:
            signal_frame = split_frame[split_frame["signal_column"] == signal_column].sort_values("bucket")
            if signal_frame.empty:
                continue
            axis.plot(
                signal_frame["bucket"],
                signal_frame[value_column],
                marker="o",
                linewidth=2.2,
                color=SIGNAL_COLORS[signal_column],
                label=SIGNAL_LABELS[signal_column],
            )
        axis.set_title(split_name.capitalize())
        axis.set_xlabel("Bucket (1 = lowest predicted risk, 5 = highest)")
        axis.set_ylabel(ylabel)
        axis.set_xticks(range(1, BUCKET_COUNT + 1))
        axis.legend(frameon=True, fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_rank_ic(diagnostics_rows: list[dict[str, object]], output_path: Path, signal_subset: list[str]) -> None:
    frame = pd.DataFrame(diagnostics_rows)
    if frame.empty:
        return
    summary = (
        frame[(frame["comparison_set"] == "surface_common") & (frame["signal_column"].isin(signal_subset))]
        .groupby(["split", "signal_column"], sort=True)["daily_rank_ic_future_rv"]
        .mean()
        .reset_index()
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for axis, split_name in zip(axes, ["validation", "test"]):
        split_frame = summary[summary["split"] == split_name]
        colors = [SIGNAL_COLORS[signal] for signal in split_frame["signal_column"]]
        labels = [SIGNAL_LABELS[signal] for signal in split_frame["signal_column"]]
        axis.bar(labels, split_frame["daily_rank_ic_future_rv"], color=colors)
        axis.set_title(split_name.capitalize())
        axis.set_ylabel("Average daily Spearman rank IC")
        axis.tick_params(axis="x", rotation=15)
    fig.suptitle("Phase 2 bucket analysis: average daily rank correlation with future RV")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_phase2_bucket_analysis() -> dict[str, object]:
    core_summary, core_bucket_rows, core_diagnostics_rows = _run_bucket_set(
        SURFACE_EXTENSION_PREDICTIONS_CSV,
        signal_columns=CORE_SIGNALS,
        comparison_set="surface_common",
    )
    write_json(PHASE2_BUCKET_METRICS_JSON, core_summary)
    write_rows_csv(PHASE2_BUCKET_METRICS_CSV, core_bucket_rows)
    write_rows_csv(PHASE2_BUCKET_DIAGNOSTICS_CSV, core_diagnostics_rows)

    extension_summary = None
    extension_bucket_rows: list[dict[str, object]] = []
    extension_diagnostics_rows: list[dict[str, object]] = []
    if CALIBRATED_SURFACE_EXTENSION_PREDICTIONS_CSV.exists():
        extension_summary, extension_bucket_rows, extension_diagnostics_rows = _run_bucket_set(
            CALIBRATED_SURFACE_EXTENSION_PREDICTIONS_CSV,
            signal_columns=EXTENSION_SIGNALS,
            comparison_set="calibrated_common",
        )
        write_json(PHASE2_BUCKET_EXTENSION_METRICS_JSON, extension_summary)
        write_rows_csv(PHASE2_BUCKET_EXTENSION_METRICS_CSV, extension_bucket_rows)
        write_rows_csv(PHASE2_BUCKET_EXTENSION_DIAGNOSTICS_CSV, extension_diagnostics_rows)

    all_bucket_rows = core_bucket_rows + extension_bucket_rows
    all_diagnostics_rows = core_diagnostics_rows + extension_diagnostics_rows
    _plot_bucket_curves(
        all_bucket_rows,
        "average_future_rv_20d",
        PHASE2_BUCKET_FUTURE_RV_FIGURE,
        "Phase 2 bucket analysis: future realized volatility",
        "Average future RV (20d)",
        CORE_SIGNALS_ORIGINAL,
    )
    _plot_bucket_curves(
        all_bucket_rows,
        "high_rv_hit_rate",
        PHASE2_BUCKET_HIT_RATE_FIGURE,
        "Phase 2 bucket analysis: high-vol regime hit rate",
        "High-vol regime hit rate",
        CORE_SIGNALS_ORIGINAL,
    )
    _plot_rank_ic(all_diagnostics_rows, PHASE2_BUCKET_RANK_IC_FIGURE, CORE_SIGNALS_ORIGINAL)

    return {
        "core": core_summary,
        "extension": extension_summary,
        "figures": [
            PHASE2_BUCKET_FUTURE_RV_FIGURE.name,
            PHASE2_BUCKET_HIT_RATE_FIGURE.name,
            PHASE2_BUCKET_RANK_IC_FIGURE.name,
        ],
    }
