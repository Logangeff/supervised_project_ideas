from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parent.parent
METRICS_DIR = ROOT_DIR / "outputs" / "metrics"
FIGURES_DIR = ROOT_DIR / "outputs" / "figures"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
SUMMARIES_DIR = ROOT_DIR / "outputs" / "summaries"

CORE_BUCKET_METRICS_CSV = METRICS_DIR / "phase2_bucket_analysis_metrics.csv"
CORE_BUCKET_DIAGNOSTICS_CSV = METRICS_DIR / "phase2_bucket_analysis_diagnostics.csv"
EXT_BUCKET_METRICS_CSV = METRICS_DIR / "phase2_bucket_analysis_extension_metrics.csv"
EXT_BUCKET_DIAGNOSTICS_CSV = METRICS_DIR / "phase2_bucket_analysis_extension_diagnostics.csv"
SURFACE_EXTENSION_METRICS_CSV = METRICS_DIR / "surface_extension_metrics.csv"
SURFACE_EXTENSION_PREDICTIONS_CSV = METRICS_DIR / "surface_extension_predictions.csv"
SURFACE_EXTENSION_PANEL_PATH = PROCESSED_DIR / "surface_extension_panel.parquet"

PHASE2_BUCKET_FUTURE_RV_FIGURE = FIGURES_DIR / "phase2_bucket_future_rv.png"
PHASE2_BUCKET_HIT_RATE_FIGURE = FIGURES_DIR / "phase2_bucket_hit_rate.png"
PHASE2_BUCKET_RANK_IC_FIGURE = FIGURES_DIR / "phase2_bucket_rank_ic.png"

EXTRACT_STOCK_SUMMARY = SUMMARIES_DIR / "extract_stock_data_summary.json"
BUILD_STOCK_PANEL_SUMMARY = SUMMARIES_DIR / "build_stock_panel_summary.json"
EXTRACT_OPTION_SUMMARY = SUMMARIES_DIR / "extract_option_data_summary.json"
BUILD_OPTION_FEATURES_SUMMARY = SUMMARIES_DIR / "build_option_features_summary.json"
BUILD_SURFACE_FACTORS_SUMMARY = SUMMARIES_DIR / "build_surface_factors_summary.json"
BUILD_CALIBRATED_SURFACE_SUMMARY = SUMMARIES_DIR / "build_calibrated_surface_summary.json"

TEMPLATE = "plotly_white"

ORIGINAL_SIGNAL_LABELS = {
    "stock_only_prob": "Stock Only",
    "option_only_prob": "Option Only",
    "all_features_prob": "Stock + Option + Surface",
}
XGB_SIGNAL_LABELS = {
    "stock_only_xgb_prob": "Stock Only (XGBoost)",
    "option_only_xgb_prob": "Option Only (XGBoost)",
    "all_features_xgb_prob": "Stock + Option + Surface (XGBoost)",
}
TRAILING_SIGNAL_LABELS = {
    "stock_only_trail2y_prob": "Stock Only (2Y Trailing)",
    "stock_only_trail5y_prob": "Stock Only (5Y Trailing)",
    "option_only_trail2y_prob": "Option Only (2Y Trailing)",
    "option_only_trail5y_prob": "Option Only (5Y Trailing)",
    "all_features_trail2y_prob": "Stock + Option + Surface (2Y Trailing)",
    "all_features_trail5y_prob": "Stock + Option + Surface (5Y Trailing)",
}
BENCHMARK_SIGNAL_LABELS = {
    **ORIGINAL_SIGNAL_LABELS,
    **XGB_SIGNAL_LABELS,
    **TRAILING_SIGNAL_LABELS,
}
EXT_SIGNAL_LABELS = {
    "beta_only_prob": "Beta Only",
    "option_beta_prob": "Option + Betas",
    "all_extensions_prob": "All Extensions",
}

SIGNAL_COLORS = {
    "Stock Only": "#6c757d",
    "Option Only": "#1f77b4",
    "Stock + Option + Surface": "#d62728",
    "Stock Only (XGBoost)": "#9aa1a8",
    "Option Only (XGBoost)": "#4aa4df",
    "Stock + Option + Surface (XGBoost)": "#ff6b6b",
    "Stock Only (2Y Trailing)": "#8c564b",
    "Stock Only (5Y Trailing)": "#c49c94",
    "Option Only (2Y Trailing)": "#17becf",
    "Option Only (5Y Trailing)": "#9adbe8",
    "Stock + Option + Surface (2Y Trailing)": "#9467bd",
    "Stock + Option + Surface (5Y Trailing)": "#c5b0d5",
    "Beta Only": "#9467bd",
    "Option + Betas": "#2ca02c",
    "All Extensions": "#ff7f0e",
}

MODEL_NAME_BY_SIGNAL = {
    "stock_only_prob": "stock_only_logreg_surface_common",
    "option_only_prob": "option_only_logreg_surface_common",
    "all_features_prob": "all_features_logreg",
    "stock_only_xgb_prob": "stock_only_xgb_surface_common",
    "option_only_xgb_prob": "option_only_xgb_surface_common",
    "all_features_xgb_prob": "all_features_xgb_surface_common",
    "stock_only_trail2y_prob": "stock_only_trail2y_logreg_surface_common",
    "stock_only_trail5y_prob": "stock_only_trail5y_logreg_surface_common",
    "option_only_trail2y_prob": "option_only_trail2y_logreg_surface_common",
    "option_only_trail5y_prob": "option_only_trail5y_logreg_surface_common",
    "all_features_trail2y_prob": "all_features_trail2y_logreg_surface_common",
    "all_features_trail5y_prob": "all_features_trail5y_logreg_surface_common",
}

BENCHMARK_FAMILIES = [
    ("Stock Only", "stock_only_prob", "stock_only_xgb_prob", "stock_only_trail2y_prob", "stock_only_trail5y_prob"),
    ("Option Only", "option_only_prob", "option_only_xgb_prob", "option_only_trail2y_prob", "option_only_trail5y_prob"),
    (
        "Stock + Option + Surface",
        "all_features_prob",
        "all_features_xgb_prob",
        "all_features_trail2y_prob",
        "all_features_trail5y_prob",
    ),
]

DRILLDOWN_SIGNAL_OPTIONS = {
    "Stock Only": "stock_only_prob",
    "Option Only": "option_only_prob",
    "Stock + Option + Surface": "all_features_prob",
    "Stock Only (XGBoost)": "stock_only_xgb_prob",
    "Option Only (XGBoost)": "option_only_xgb_prob",
    "Stock + Option + Surface (XGBoost)": "all_features_xgb_prob",
    "Stock Only (2Y Trailing)": "stock_only_trail2y_prob",
    "Stock Only (5Y Trailing)": "stock_only_trail5y_prob",
    "Option Only (2Y Trailing)": "option_only_trail2y_prob",
    "Option Only (5Y Trailing)": "option_only_trail5y_prob",
    "Stock + Option + Surface (2Y Trailing)": "all_features_trail2y_prob",
    "Stock + Option + Surface (5Y Trailing)": "all_features_trail5y_prob",
}


def _load_csv(path: Path, parse_dates: list[str] | None = None) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=parse_dates)


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _assign_bucket(series: pd.Series) -> pd.Series:
    ranks = series.rank(method="first", pct=True)
    buckets = (ranks * 5).apply(lambda value: min(5, max(1, int(value) if float(value).is_integer() else int(value) + 1)))
    return pd.Series(buckets.values, index=series.index, dtype=int)


def _percent(value: float | int | None, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{100 * float(value):.{digits}f}%"


def _number(value: float | int | None, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):,.{digits}f}"


def _integer(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{int(value):,}"


@st.cache_data(show_spinner=False)
def load_artifacts() -> dict[str, object]:
    return {
        "core_metrics": _load_csv(CORE_BUCKET_METRICS_CSV),
        "core_diag": _load_csv(CORE_BUCKET_DIAGNOSTICS_CSV, parse_dates=["trade_date"]),
        "ext_metrics": _load_csv(EXT_BUCKET_METRICS_CSV),
        "ext_diag": _load_csv(EXT_BUCKET_DIAGNOSTICS_CSV, parse_dates=["trade_date"]),
        "sup_metrics": _load_csv(SURFACE_EXTENSION_METRICS_CSV),
        "extract_stock": _load_json(EXTRACT_STOCK_SUMMARY),
        "build_stock": _load_json(BUILD_STOCK_PANEL_SUMMARY),
        "extract_option": _load_json(EXTRACT_OPTION_SUMMARY),
        "build_option": _load_json(BUILD_OPTION_FEATURES_SUMMARY),
        "build_surface": _load_json(BUILD_SURFACE_FACTORS_SUMMARY),
        "build_calibrated": _load_json(BUILD_CALIBRATED_SURFACE_SUMMARY),
    }


@st.cache_data(show_spinner=False)
def load_core_stock_drilldown_panel() -> pd.DataFrame | None:
    if not SURFACE_EXTENSION_PREDICTIONS_CSV.exists() or not SURFACE_EXTENSION_PANEL_PATH.exists():
        return None

    predictions = pd.read_csv(SURFACE_EXTENSION_PREDICTIONS_CSV, parse_dates=["trade_date"])
    panel_columns = [
        "permno",
        "trade_date",
        "universe_ticker",
        "company_name",
        "sector",
        "prc",
        "ret",
        "rv_20d",
        "future_rv_20d",
    ]
    surface_panel = pd.read_parquet(SURFACE_EXTENSION_PANEL_PATH, columns=panel_columns)
    surface_panel["trade_date"] = pd.to_datetime(surface_panel["trade_date"])

    merged = predictions.merge(
        surface_panel,
        on=["permno", "trade_date", "universe_ticker"],
        how="left",
    )
    merged = merged.sort_values(["universe_ticker", "trade_date"]).reset_index(drop=True)

    for signal_column in DRILLDOWN_SIGNAL_OPTIONS.values():
        if signal_column in merged.columns:
            merged[f"{signal_column}_bucket"] = (
                merged.groupby(["split", "trade_date"], sort=False)[signal_column].transform(_assign_bucket)
            )

    merged["future_high_rv_label"] = merged["label"].astype(int)
    return merged


def _apply_signal_labels(frame: pd.DataFrame, label_map: dict[str, str]) -> pd.DataFrame:
    labeled = frame.copy()
    labeled = labeled[labeled["signal_column"].isin(label_map)].copy()
    labeled["signal_name"] = labeled["signal_column"].map(label_map)
    return labeled


def _summary_from_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["signal_column", "signal_name", "split"], as_index=False)
        .agg(
            avg_rank_ic_future_rv=("daily_rank_ic_future_rv", "mean"),
            avg_rank_ic_label=("daily_rank_ic_label", "mean"),
            avg_top_bottom_future_rv_spread=("top_bottom_future_rv_spread", "mean"),
            avg_top_bottom_hit_rate_spread=("top_bottom_hit_rate_spread", "mean"),
            avg_top_bottom_abs_return_spread=("top_bottom_abs_return_spread", "mean"),
            monotonic_future_rv_share=("monotonic_future_rv", "mean"),
            monotonic_hit_rate_share=("monotonic_hit_rate", "mean"),
            trade_dates=("trade_date", "nunique"),
        )
        .sort_values(["split", "avg_rank_ic_future_rv"], ascending=[True, False])
        .reset_index(drop=True)
    )


def _get_summary_row(summary_df: pd.DataFrame, split: str, signal_column: str) -> pd.Series | None:
    match = summary_df[(summary_df["split"] == split) & (summary_df["signal_column"] == signal_column)]
    if match.empty:
        return None
    return match.iloc[0]


def _get_supervised_row(metrics_df: pd.DataFrame, split: str, signal_column: str) -> pd.Series | None:
    model_name = MODEL_NAME_BY_SIGNAL.get(signal_column)
    if model_name is None:
        return None
    match = metrics_df[(metrics_df["split"] == split) & (metrics_df["model"] == model_name)]
    if match.empty:
        return None
    return match.iloc[0]


def _headline_payload(summary_df: pd.DataFrame, metrics_df: pd.DataFrame, split: str) -> dict[str, object]:
    baseline_signal = "all_features_prob"
    summary_row = _get_summary_row(summary_df, split, baseline_signal)
    supervised_row = _get_supervised_row(metrics_df, split, baseline_signal)
    if summary_row is None or supervised_row is None:
        return {}
    return {
        "signal_name": ORIGINAL_SIGNAL_LABELS[baseline_signal],
        "rank_ic": summary_row["avg_rank_ic_future_rv"],
        "rv_spread": summary_row["avg_top_bottom_future_rv_spread"],
        "hit_spread": summary_row["avg_top_bottom_hit_rate_spread"],
        "pr_auc": supervised_row["pr_auc"],
        "macro_f1": supervised_row["macro_f1"],
    }


def _build_benchmark_table(summary_df: pd.DataFrame, metrics_df: pd.DataFrame, split: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for family_name, baseline_col, xgb_col, trail2_col, trail5_col in BENCHMARK_FAMILIES:
        baseline_bucket = _get_summary_row(summary_df, split, baseline_col)
        baseline_sup = _get_supervised_row(metrics_df, split, baseline_col)
        if baseline_bucket is None or baseline_sup is None:
            continue

        payload: dict[str, object] = {
            "Signal family": family_name,
            "Baseline rank IC": _number(baseline_bucket["avg_rank_ic_future_rv"]),
            "Baseline PR-AUC": _percent(baseline_sup["pr_auc"]),
        }

        for column_name, short_label in (
            (xgb_col, "XGBoost"),
            (trail2_col, "2Y trailing"),
            (trail5_col, "5Y trailing"),
        ):
            bucket_row = _get_summary_row(summary_df, split, column_name)
            sup_row = _get_supervised_row(metrics_df, split, column_name)
            if bucket_row is None or sup_row is None:
                continue
            payload[f"{short_label} rank IC"] = _number(bucket_row["avg_rank_ic_future_rv"])
            payload[f"{short_label} delta rank IC"] = _number(
                bucket_row["avg_rank_ic_future_rv"] - baseline_bucket["avg_rank_ic_future_rv"], digits=3
            )
            payload[f"{short_label} PR-AUC"] = _percent(sup_row["pr_auc"])
            payload[f"{short_label} delta PR-AUC"] = _percent(sup_row["pr_auc"] - baseline_sup["pr_auc"], digits=2)
        rows.append(payload)
    return pd.DataFrame(rows)


def _family_benchmark_callout(summary_df: pd.DataFrame, metrics_df: pd.DataFrame, split: str) -> str:
    lines: list[str] = []
    for family_name, baseline_col, xgb_col, trail2_col, trail5_col in BENCHMARK_FAMILIES:
        baseline_bucket = _get_summary_row(summary_df, split, baseline_col)
        baseline_sup = _get_supervised_row(metrics_df, split, baseline_col)
        if baseline_bucket is None or baseline_sup is None:
            continue

        candidates: list[tuple[str, str, pd.Series, pd.Series]] = []
        for variant_col, variant_label in (
            (xgb_col, "XGBoost"),
            (trail2_col, "2Y trailing"),
            (trail5_col, "5Y trailing"),
        ):
            bucket_row = _get_summary_row(summary_df, split, variant_col)
            sup_row = _get_supervised_row(metrics_df, split, variant_col)
            if bucket_row is not None and sup_row is not None:
                candidates.append((variant_col, variant_label, bucket_row, sup_row))

        if not candidates:
            continue

        best_variant = max(candidates, key=lambda item: item[2]["avg_rank_ic_future_rv"])
        best_delta_rank = best_variant[2]["avg_rank_ic_future_rv"] - baseline_bucket["avg_rank_ic_future_rv"]

        if family_name == "Stock Only" and best_delta_rank > 0.001:
            lines.append(
                f"- `{family_name}`: `{best_variant[1]}` improves rank IC from "
                f"`{baseline_bucket['avg_rank_ic_future_rv']:.3f}` to `{best_variant[2]['avg_rank_ic_future_rv']:.3f}` "
                f"and PR-AUC from `{baseline_sup['pr_auc']:.3f}` to `{best_variant[3]['pr_auc']:.3f}`."
            )
        elif abs(best_delta_rank) <= 0.003:
            lines.append(
                f"- `{family_name}`: the best benchmark is effectively a tie on bucket ranking "
                f"(`{baseline_bucket['avg_rank_ic_future_rv']:.3f}` baseline vs "
                f"`{best_variant[2]['avg_rank_ic_future_rv']:.3f}` for `{best_variant[1]}`), "
                f"so the original fixed-split logistic result stays the clean headline."
            )
        elif best_delta_rank > 0:
            lines.append(
                f"- `{family_name}`: `{best_variant[1]}` raises rank IC slightly "
                f"(`{baseline_bucket['avg_rank_ic_future_rv']:.3f}` to `{best_variant[2]['avg_rank_ic_future_rv']:.3f}`), "
                f"but the gain is small and PR-AUC moves from `{baseline_sup['pr_auc']:.3f}` to `{best_variant[3]['pr_auc']:.3f}`."
            )
        else:
            lines.append(
                f"- `{family_name}`: no benchmark improved the bucket-ranking result relative to the baseline."
            )

    lines.append(
        "- Overall: keep the original fixed-split logistic model as the main professor-demo result. "
        "The benchmarks are useful robustness checks, not replacements."
    )
    return "\n".join(lines)


def _ranking_chart(summary_df: pd.DataFrame, split: str, metric_col: str, title: str, labels_as_percent: bool = False) -> go.Figure:
    plot_df = summary_df[summary_df["split"] == split].sort_values(metric_col, ascending=True)
    text_values = [_percent(value) if labels_as_percent else _number(value) for value in plot_df[metric_col]]
    fig = go.Figure(
        go.Bar(
            x=plot_df[metric_col] * (100 if labels_as_percent else 1),
            y=plot_df["signal_name"],
            orientation="h",
            marker_color=[SIGNAL_COLORS.get(name, "#4c78a8") for name in plot_df["signal_name"]],
            text=text_values,
            textposition="outside",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=title,
        yaxis_title="Signal",
        template=TEMPLATE,
        height=320,
        margin={"l": 10, "r": 20, "t": 50, "b": 10},
    )
    if labels_as_percent:
        fig.update_xaxes(ticksuffix="%")
    return fig


def _bucket_curve_chart(
    frame: pd.DataFrame,
    split: str,
    value_col: str,
    title: str,
    yaxis_title: str,
    labels_as_percent: bool = False,
) -> go.Figure:
    plot_df = frame[frame["split"] == split].copy()
    if labels_as_percent:
        plot_df[value_col] = 100 * plot_df[value_col]
    fig = px.line(
        plot_df,
        x="bucket",
        y=value_col,
        color="signal_name",
        markers=True,
        color_discrete_map=SIGNAL_COLORS,
        template=TEMPLATE,
    )
    fig.update_traces(line={"width": 3}, marker={"size": 9})
    fig.update_layout(
        title=title,
        xaxis_title="Predicted risk bucket",
        yaxis_title=yaxis_title,
        height=360,
        legend_title="Signal",
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )
    if labels_as_percent:
        fig.update_yaxes(ticksuffix="%")
    return fig


def _rolling_rank_ic_chart(frame: pd.DataFrame, split: str, selected_signals: list[str], window: int = 21) -> go.Figure:
    plot_df = frame[(frame["split"] == split) & (frame["signal_name"].isin(selected_signals))].copy()
    plot_df = plot_df.sort_values(["signal_name", "trade_date"])
    plot_df["rolling_rank_ic_future_rv"] = (
        plot_df.groupby("signal_name")["daily_rank_ic_future_rv"].transform(lambda x: x.rolling(window, min_periods=5).mean())
    )
    fig = px.line(
        plot_df,
        x="trade_date",
        y="rolling_rank_ic_future_rv",
        color="signal_name",
        color_discrete_map=SIGNAL_COLORS,
        template=TEMPLATE,
    )
    fig.update_layout(
        title=f"{window}-Day Rolling Rank IC vs Future RV",
        xaxis_title="Trade date",
        yaxis_title="Rolling rank IC",
        height=360,
        legend_title="Signal",
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )
    fig.add_hline(y=0, line_dash="dot", line_color="#999999")
    return fig


def _formatted_bucket_table(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    plot_df = frame[frame["split"] == split].copy()
    table = plot_df[
        [
            "signal_name",
            "bucket",
            "observation_count",
            "average_signal_probability",
            "average_future_rv_20d",
            "high_rv_hit_rate",
            "average_abs_ret_next_1d",
        ]
    ].rename(
        columns={
            "signal_name": "Signal",
            "bucket": "Bucket",
            "observation_count": "Rows",
            "average_signal_probability": "Avg predicted risk",
            "average_future_rv_20d": "Avg future RV (20d)",
            "high_rv_hit_rate": "High-RV hit rate",
            "average_abs_ret_next_1d": "Avg abs next-day return",
        }
    )
    table["Rows"] = table["Rows"].map(_integer)
    table["Avg predicted risk"] = table["Avg predicted risk"].map(_percent)
    table["Avg future RV (20d)"] = table["Avg future RV (20d)"].map(_percent)
    table["High-RV hit rate"] = table["High-RV hit rate"].map(_percent)
    table["Avg abs next-day return"] = table["Avg abs next-day return"].map(_percent)
    return table


def _build_feature_coverage_table(artifacts: dict[str, object]) -> pd.DataFrame:
    build_option = artifacts["build_option"] or {}
    build_surface = artifacts["build_surface"] or {}

    rows: list[dict[str, object]] = []
    for feature, coverage in (build_option.get("feature_coverage") or {}).items():
        rows.append({"Feature": feature, "Coverage": 100 * coverage, "Group": "Option feature"})
    for feature, coverage in (build_surface.get("feature_coverage_smoothed") or {}).items():
        rows.append({"Feature": feature, "Coverage": 100 * coverage, "Group": "Surface feature"})
    return pd.DataFrame(rows)


def _coverage_chart(coverage_df: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        coverage_df,
        x="Coverage",
        y="Feature",
        color="Group",
        orientation="h",
        template=TEMPLATE,
        barmode="group",
        color_discrete_map={"Option feature": "#1f77b4", "Surface feature": "#d62728"},
    )
    fig.update_layout(
        title="Modeling feature coverage",
        xaxis_title="Coverage of usable rows (%)",
        yaxis_title="Feature",
        height=520,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )
    fig.update_xaxes(range=[0, 105], ticksuffix="%")
    return fig


def _retention_funnel_chart(artifacts: dict[str, object]) -> go.Figure:
    build_surface = artifacts["build_surface"] or {}
    fig = go.Figure(
        go.Funnel(
            y=[
                "Raw option quote rows",
                "Eligible quote rows",
                "Surface panel rows",
                "Surface complete rows",
                "Surface-extension rows",
            ],
            x=[
                build_surface.get("total_raw_option_rows_seen", 0),
                build_surface.get("total_eligible_option_rows_seen", 0),
                build_surface.get("surface_panel_rows", 0),
                build_surface.get("surface_complete_rows", 0),
                build_surface.get("surface_extension_rows", 0),
            ],
            textinfo="value+percent initial",
            marker={"color": ["#6c757d", "#1f77b4", "#17becf", "#d62728", "#9467bd"]},
        )
    )
    fig.update_layout(
        title="Data retention funnel for the option-surface branch",
        template=TEMPLATE,
        height=400,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )
    return fig


def _yearly_quote_table(artifacts: dict[str, object]) -> pd.DataFrame:
    build_surface = artifacts["build_surface"] or {}
    yearly = pd.DataFrame(build_surface.get("yearly_quote_summary") or [])
    if yearly.empty:
        return yearly
    rename_map = {
        "year": "Year",
        "raw_rows": "Raw quote rows",
        "eligible_rows": "Eligible rows",
        "surface_rows": "Surface rows",
        "surface_complete_rows": "Surface complete rows",
    }
    table = yearly.rename(columns=rename_map)
    for column in ["Raw quote rows", "Eligible rows", "Surface rows", "Surface complete rows"]:
        if column in table.columns:
            table[column] = table[column].map(_integer)
    return table


def _split_target_table(artifacts: dict[str, object]) -> pd.DataFrame:
    build_stock = artifacts["build_stock"] or {}
    split_sizes = build_stock.get("split_sizes") or {}
    positive_shares = build_stock.get("positive_share_by_split") or {}
    rows = []
    for split in ["train", "validation", "test"]:
        rows.append(
            {
                "Split": split.title(),
                "Rows": _integer(split_sizes.get(split)),
                "High-RV positive share": _percent(positive_shares.get(split)),
            }
        )
    return pd.DataFrame(rows)


def _data_retained_table(artifacts: dict[str, object]) -> pd.DataFrame:
    build_option = artifacts["build_option"] or {}
    build_surface = artifacts["build_surface"] or {}
    rows = [
        {"Stage": "Merged stock + option panel rows", "Rows": _integer(build_option.get("merged_panel_rows"))},
        {"Stage": "Complete-case rows for surface-common training", "Rows": _integer(build_option.get("complete_case_rows"))},
        {"Stage": "Surface-extension rows", "Rows": _integer(build_surface.get("surface_extension_rows"))},
    ]
    return pd.DataFrame(rows)


def _contiguous_true_segments(frame: pd.DataFrame, flag_column: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    flagged = frame.sort_values("trade_date")[["trade_date", flag_column]].copy()
    flagged = flagged[flagged[flag_column].fillna(0).astype(int) == 1].reset_index(drop=True)
    if flagged.empty:
        return []

    segments: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    start_date = flagged.loc[0, "trade_date"]
    prev_date = start_date

    for current_date in flagged["trade_date"].iloc[1:]:
        if (current_date - prev_date).days > 5:
            segments.append((start_date, prev_date + pd.Timedelta(days=1)))
            start_date = current_date
        prev_date = current_date

    segments.append((start_date, prev_date + pd.Timedelta(days=1)))
    return segments


def _stock_timeline_chart(frame: pd.DataFrame, signal_column: str) -> go.Figure:
    signal_label = next((label for label, column in DRILLDOWN_SIGNAL_OPTIONS.items() if column == signal_column), signal_column)
    bucket_column = f"{signal_column}_bucket"

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.28, 0.18, 0.28, 0.26],
        specs=[[{"secondary_y": True}], [{}], [{"secondary_y": True}], [{}]],
    )

    segments = _contiguous_true_segments(frame, "future_high_rv_label")
    for x0, x1 in segments:
        for row_idx in range(1, 5):
            fig.add_vrect(
                x0=x0,
                x1=x1,
                fillcolor="rgba(214, 39, 40, 0.10)",
                line_width=0,
                row=row_idx,
                col=1,
            )

    price_base = frame["prc"].dropna().iloc[0] if not frame["prc"].dropna().empty else 1.0
    normalized_price = 100 * frame["prc"] / price_base if price_base else frame["prc"]

    fig.add_trace(
        go.Scatter(
            x=frame["trade_date"],
            y=normalized_price,
            name="Normalized price",
            line={"color": "#1f77b4", "width": 2.5},
        ),
        row=1,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=frame["trade_date"],
            y=100 * frame[signal_column],
            name=f"{signal_label} probability",
            line={"color": "#d62728", "width": 2, "dash": "dot"},
        ),
        row=1,
        col=1,
        secondary_y=True,
    )

    if bucket_column in frame.columns:
        fig.add_trace(
            go.Scatter(
                x=frame["trade_date"],
                y=frame[bucket_column],
                name="Daily bucket",
                line={"color": "#9467bd", "width": 2, "shape": "hv"},
            ),
            row=2,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=frame["trade_date"],
            y=100 * frame["rv_20d"],
            name="Current RV (20d)",
            line={"color": "#2ca02c", "width": 2.2},
        ),
        row=3,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=frame["trade_date"],
            y=100 * frame["future_rv_20d"],
            name="Future RV (20d)",
            line={"color": "#ff7f0e", "width": 2.2},
        ),
        row=3,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=frame["trade_date"],
            y=100 * frame[signal_column],
            name="Predicted risk",
            line={"color": "#d62728", "width": 1.8, "dash": "dot"},
            showlegend=False,
        ),
        row=3,
        col=1,
        secondary_y=True,
    )

    fig.add_trace(
        go.Bar(
            x=frame["trade_date"],
            y=100 * frame["ret"],
            name="Daily return",
            marker_color=["#2ca02c" if value >= 0 else "#d62728" for value in frame["ret"].fillna(0.0)],
            opacity=0.75,
        ),
        row=4,
        col=1,
    )

    fig.update_yaxes(title_text="Price index", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Pred risk (%)", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Bucket", row=2, col=1, range=[0.5, 5.5], tickmode="linear", dtick=1)
    fig.update_yaxes(title_text="Volatility (%)", row=3, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Pred risk (%)", row=3, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Return (%)", row=4, col=1)
    fig.update_xaxes(rangeslider={"visible": True}, row=4, col=1, title_text="Trade date")
    fig.update_layout(
        title="Stock timeline: predicted risk vs realized path",
        template=TEMPLATE,
        height=960,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
    )
    return fig


def _stock_summary_cards(frame: pd.DataFrame, signal_column: str) -> None:
    latest = frame.sort_values("trade_date").iloc[-1]
    bucket_column = f"{signal_column}_bucket"
    avg_bucket = frame[bucket_column].mean() if bucket_column in frame.columns else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest predicted risk", _percent(latest[signal_column]))
    c2.metric("Latest future-RV label", "High-RV" if int(latest["future_high_rv_label"]) == 1 else "Normal-RV")
    c3.metric("Average future RV (20d)", _percent(frame["future_rv_20d"].mean()))
    c4.metric("Average daily bucket", _number(avg_bucket, digits=2))


def _stock_summary_text(frame: pd.DataFrame, signal_column: str) -> str:
    latest = frame.sort_values("trade_date").iloc[-1]
    bucket_column = f"{signal_column}_bucket"
    avg_bucket = frame[bucket_column].mean() if bucket_column in frame.columns else float("nan")
    high_bucket_share = (frame[bucket_column] >= 4).mean() if bucket_column in frame.columns else float("nan")
    rho = frame[[signal_column, "future_rv_20d"]].corr(method="spearman").iloc[0, 1]
    latest_label = "High-RV" if int(latest["future_high_rv_label"]) == 1 else "Normal-RV"
    return (
        f"The selected signal currently assigns `{_percent(latest[signal_column])}` risk to this stock, and the latest "
        f"realized outcome is `{latest_label}`. Over the displayed period, the stock sits in bucket "
        f"`{_number(avg_bucket, 2)}` on average, spends `{_percent(high_bucket_share)}` of days in buckets 4-5, "
        f"and the within-stock Spearman correlation between predicted risk and future RV is `{_number(rho, 3)}`."
    )


def _stock_detail_table(frame: pd.DataFrame, signal_column: str) -> pd.DataFrame:
    bucket_column = f"{signal_column}_bucket"
    keep_columns = [
        "trade_date",
        signal_column,
        bucket_column,
        "future_high_rv_label",
        "prc",
        "ret",
        "rv_20d",
        "future_rv_20d",
    ]
    table = frame[[column for column in keep_columns if column in frame.columns]].copy()
    rename_map = {
        "trade_date": "Date",
        signal_column: "Predicted risk",
        bucket_column: "Bucket",
        "future_high_rv_label": "Actual future high-RV label",
        "prc": "Price",
        "ret": "Daily return",
        "rv_20d": "Current RV (20d)",
        "future_rv_20d": "Future RV (20d)",
    }
    table = table.rename(columns=rename_map).sort_values("Date", ascending=False)
    for column in ["Predicted risk", "Daily return", "Current RV (20d)", "Future RV (20d)"]:
        if column in table.columns:
            table[column] = table[column].map(_percent)
    if "Price" in table.columns:
        table["Price"] = table["Price"].map(lambda x: f"{x:,.2f}" if pd.notna(x) else "N/A")
    return table


def _render_overview_tab(summary_df: pd.DataFrame, metrics_df: pd.DataFrame) -> None:
    st.header("Overview")
    split = st.radio("Evaluation split", ["test", "validation"], horizontal=True, key="overview_split")
    headline = _headline_payload(summary_df, metrics_df, split)
    if not headline:
        st.warning("Missing headline metrics for the overview page.")
        return

    st.markdown(
        """
        This project predicts whether a large-cap U.S. stock will enter a **high-volatility regime over the next 20 trading days**.
        The key result is not just classification accuracy. It is that **option-implied information materially improves the model's
        ability to rank stocks by future risk**, which makes the probabilities useful for cross-sectional screening and risk triage.
        """
    )
    st.info(
        "How to read the demo: start with the original fixed-split logistic result, then check whether the high-risk buckets "
        "actually realize higher future volatility. After that, treat XGBoost and trailing-window retrains as robustness checks."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Headline signal", headline["signal_name"])
    c2.metric("PR-AUC", _percent(headline["pr_auc"]))
    c3.metric("Avg daily rank IC", _number(headline["rank_ic"]))
    c4.metric("Top-bottom hit-rate spread", _percent(headline["hit_spread"]))
    c5, c6 = st.columns(2)
    c5.metric("Top-bottom future RV spread", _number(headline["rv_spread"]))
    c6.metric("Macro-F1", _number(headline["macro_f1"]))

    original_summary = summary_df[(summary_df["split"] == split) & (summary_df["signal_column"].isin(ORIGINAL_SIGNAL_LABELS))].copy()
    original_summary["sort_rank"] = original_summary["signal_column"].map(
        {"stock_only_prob": 0, "option_only_prob": 1, "all_features_prob": 2}
    )
    original_summary = original_summary.sort_values("sort_rank")

    fig = go.Figure(
        go.Bar(
            x=original_summary["signal_name"],
            y=original_summary["avg_rank_ic_future_rv"],
            marker_color=[SIGNAL_COLORS.get(name, "#4c78a8") for name in original_summary["signal_name"]],
            text=[_number(value) for value in original_summary["avg_rank_ic_future_rv"]],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Original project: option-aware signals dominate stock-only ranking",
        xaxis_title="Signal family",
        yaxis_title="Average daily rank IC",
        template=TEMPLATE,
        height=360,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "The main demo message is immediate: stock-only has real signal, but option-only and stock + option + surface "
        "produce a much stronger out-of-sample risk ranking."
    )


def _render_core_result_tab(metrics_df: pd.DataFrame, summary_df: pd.DataFrame, diag_df: pd.DataFrame) -> None:
    st.header("Core Result")
    split = st.radio("Evaluation split", ["test", "validation"], horizontal=True, key="core_split")
    core_metrics = metrics_df[metrics_df["signal_column"].isin(ORIGINAL_SIGNAL_LABELS)].copy()
    core_diag = summary_df[summary_df["signal_column"].isin(ORIGINAL_SIGNAL_LABELS)].copy()
    selected_signals = list(ORIGINAL_SIGNAL_LABELS.values())

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            _ranking_chart(core_diag, split, "avg_rank_ic_future_rv", "Average daily rank IC vs future RV"),
            use_container_width=True,
        )
        st.caption(
            "Higher is better. The option-aware signals rank future stock-level risk much more accurately than stock-only."
        )
    with right:
        st.plotly_chart(
            _ranking_chart(
                core_diag,
                split,
                "avg_top_bottom_hit_rate_spread",
                "Top-bottom high-volatility hit-rate spread",
                labels_as_percent=True,
            ),
            use_container_width=True,
        )
        st.caption(
            "This measures how much more often Bucket 5 enters the high-volatility regime than Bucket 1. "
            "A large spread means the model cleanly separates dangerous names from calmer names."
        )

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            _bucket_curve_chart(
                core_metrics,
                split,
                "average_future_rv_20d",
                "Future 20-day realized volatility by bucket",
                "Average future RV (20d)",
                labels_as_percent=True,
            ),
            use_container_width=True,
        )
        st.caption(
            "Success means the lines rise from Bucket 1 to Bucket 5. That is the visual proof that the probabilities sort stocks by future volatility."
        )
    with c2:
        st.plotly_chart(
            _bucket_curve_chart(
                core_metrics,
                split,
                "high_rv_hit_rate",
                "High-volatility regime hit rate by bucket",
                "Hit rate",
                labels_as_percent=True,
            ),
            use_container_width=True,
        )
        st.caption(
            "The same sorting pattern holds on the binary high-volatility outcome, not only on continuous realized volatility."
        )

    st.plotly_chart(_rolling_rank_ic_chart(diag_df, split, selected_signals), use_container_width=True)
    st.caption("The option-driven signals stay strong through time rather than working only in one short lucky window.")

    st.subheader("Bucket detail table")
    st.dataframe(_formatted_bucket_table(core_metrics, split), use_container_width=True, hide_index=True)


def _render_benchmark_tab(summary_df: pd.DataFrame, metrics_df: pd.DataFrame) -> None:
    st.header("Benchmark Comparison")
    split = st.radio("Evaluation split", ["test", "validation"], horizontal=True, key="benchmark_split")
    st.markdown(
        """
        The benchmark question is narrow: does a more complex model or a refreshed training window **improve the useful
        cross-sectional risk ranking**, not just one classification metric?
        """
    )
    st.markdown(_family_benchmark_callout(summary_df, metrics_df, split))

    st.subheader("Baseline vs benchmark deltas")
    st.dataframe(_build_benchmark_table(summary_df, metrics_df, split), use_container_width=True, hide_index=True)

    family = st.selectbox(
        "Signal family",
        options=["Stock Only", "Option Only", "Stock + Option + Surface"],
        index=2,
        key="benchmark_family",
    )
    family_lookup = {entry[0]: entry[1:] for entry in BENCHMARK_FAMILIES}
    chosen_columns = list(family_lookup[family])
    family_summary = summary_df[(summary_df["split"] == split) & (summary_df["signal_column"].isin(chosen_columns))].copy()
    family_summary = family_summary.sort_values("avg_rank_ic_future_rv", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=family_summary["avg_rank_ic_future_rv"],
            y=family_summary["signal_name"],
            orientation="h",
            marker_color=[SIGNAL_COLORS.get(name, "#4c78a8") for name in family_summary["signal_name"]],
            text=[_number(value) for value in family_summary["avg_rank_ic_future_rv"]],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=f"{family}: bucket-ranking comparison",
        xaxis_title="Average daily rank IC",
        yaxis_title="Variant",
        template=TEMPLATE,
        height=340,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Use this family view to see whether the benchmark actually beats the original baseline on the Phase 2 risk-sorting objective.")

    rows = []
    for signal_column in chosen_columns:
        bucket_row = _get_summary_row(summary_df, split, signal_column)
        sup_row = _get_supervised_row(metrics_df, split, signal_column)
        if bucket_row is None or sup_row is None:
            continue
        rows.append(
            {
                "Variant": BENCHMARK_SIGNAL_LABELS.get(signal_column, signal_column),
                "Rank IC": _number(bucket_row["avg_rank_ic_future_rv"]),
                "Top-bottom RV spread": _number(bucket_row["avg_top_bottom_future_rv_spread"]),
                "Top-bottom hit-rate spread": _percent(bucket_row["avg_top_bottom_hit_rate_spread"]),
                "Macro-F1": _number(sup_row["macro_f1"]),
                "PR-AUC": _percent(sup_row["pr_auc"]),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_single_stock_tab() -> None:
    st.header("Single-Stock Drill-Down")
    panel = load_core_stock_drilldown_panel()
    if panel is None or panel.empty:
        st.warning("Missing stock-level prediction artifacts for the drill-down page.")
        return

    control_col1, control_col2, control_col3 = st.columns([1.1, 1.2, 1.6])
    with control_col1:
        split = st.selectbox("Evaluation split", ["test", "validation"], index=0)
    filtered = panel[panel["split"] == split].copy()
    if filtered.empty:
        st.info(f"No {split} stock rows available.")
        return

    ticker_options = sorted(filtered["universe_ticker"].dropna().unique().tolist())
    default_ticker = "AAPL" if "AAPL" in ticker_options else ticker_options[0]
    with control_col2:
        ticker = st.selectbox("Ticker", ticker_options, index=ticker_options.index(default_ticker))
    with control_col3:
        signal_label = st.selectbox(
            "Signal view",
            list(DRILLDOWN_SIGNAL_OPTIONS.keys()),
            index=list(DRILLDOWN_SIGNAL_OPTIONS.keys()).index("Stock + Option + Surface"),
        )
    signal_column = DRILLDOWN_SIGNAL_OPTIONS[signal_label]

    stock_frame = filtered[filtered["universe_ticker"] == ticker].copy().sort_values("trade_date")
    if stock_frame.empty:
        st.info(f"No rows available for {ticker} on {split}.")
        return

    latest_meta = stock_frame.iloc[-1]
    st.markdown(
        f"**{ticker}**  \n"
        f"Company: `{latest_meta.get('company_name', 'N/A')}`  \n"
        f"Sector: `{latest_meta.get('sector', 'N/A')}`  \n"
        f"Dates: `{stock_frame['trade_date'].min().date()}` to `{stock_frame['trade_date'].max().date()}`"
    )

    _stock_summary_cards(stock_frame, signal_column)
    st.info(_stock_summary_text(stock_frame, signal_column))
    st.plotly_chart(_stock_timeline_chart(stock_frame, signal_column), use_container_width=True)
    st.caption(
        "Red background bands mark dates where the stock later entered the high-volatility regime. "
        "The dotted red line is the model probability, the purple step line is the cross-sectional bucket, "
        "and the orange line is the realized future 20-day volatility the model is trying to anticipate."
    )

    st.subheader("Stock detail table")
    st.dataframe(_stock_detail_table(stock_frame, signal_column), use_container_width=True, hide_index=True)


def _render_data_quality_tab(artifacts: dict[str, object]) -> None:
    st.header("Data Quality")
    extract_stock = artifacts["extract_stock"] or {}
    build_option = artifacts["build_option"] or {}
    extract_option = artifacts["extract_option"] or {}
    build_surface = artifacts["build_surface"] or {}

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Universe size", _integer(extract_stock.get("universe_size")))
    c2.metric("Resolved tickers", _integer(extract_stock.get("resolved_permnos")))
    c3.metric("Unresolved tickers", _integer(len(extract_stock.get("unresolved_tickers") or [])))
    c4.metric("Merged panel rows", _integer(build_option.get("merged_panel_rows")))

    c5, c6, c7, c8 = st.columns(4)
    stock_range = extract_stock.get("date_range") or {}
    option_range = extract_option.get("date_range") or {}
    c5.metric("Stock date coverage", f"{stock_range.get('min_date', 'N/A')} to {stock_range.get('max_date', 'N/A')}")
    c6.metric("Option date coverage", f"{option_range.get('min_date', 'N/A')} to {option_range.get('max_date', 'N/A')}")
    c7.metric("Complete-case rows", _integer(build_option.get("complete_case_rows")))
    c8.metric("Surface-extension rows", _integer(build_surface.get("surface_extension_rows")))

    st.markdown(
        """
        **No-lookahead design**

        - The target is whether a stock enters a high-volatility regime over the **next 20 trading days**.
        - The high-volatility threshold is fixed from the **training set only**.
        - Train, validation, and test splits are **strictly chronological**.
        - The bucket analysis uses saved out-of-sample probabilities, not in-sample fits.
        """
    )

    coverage_df = _build_feature_coverage_table(artifacts)
    left, right = st.columns([1.1, 0.9])
    with left:
        st.plotly_chart(_coverage_chart(coverage_df), use_container_width=True)
    with right:
        st.plotly_chart(_retention_funnel_chart(artifacts), use_container_width=True)
    st.caption(
        "The coverage chart shows that the main option and surface variables are observed for most usable rows. "
        "The funnel shows how raw option quotes become the final modeling panel after eligibility and surface construction."
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Split sizes and target prevalence")
        st.dataframe(_split_target_table(artifacts), use_container_width=True, hide_index=True)
    with right:
        st.subheader("Data retained after filtering")
        st.dataframe(_data_retained_table(artifacts), use_container_width=True, hide_index=True)

    quote_summary = build_surface.get("surface_quote_count_summary") or {}
    st.subheader("Surface quote-count quality")
    qc1, qc2, qc3 = st.columns(3)
    qc1.metric("Median quotes per surface date", _number(quote_summary.get("median"), digits=0))
    qc2.metric("10th percentile", _number(quote_summary.get("p10"), digits=0))
    qc3.metric("90th percentile", _number(quote_summary.get("p90"), digits=0))

    st.subheader("Yearly surface-construction summary")
    st.dataframe(_yearly_quote_table(artifacts), use_container_width=True, hide_index=True)


def _render_appendix_tab(ext_metrics_df: pd.DataFrame, ext_diag_df: pd.DataFrame) -> None:
    st.header("Appendix")
    split = st.radio("Evaluation split", ["test", "validation"], horizontal=True, key="appendix_split")

    st.subheader("Calibrated extension")
    ext_summary = _summary_from_diagnostics(ext_diag_df)
    ext_split = ext_summary[ext_summary["split"] == split].sort_values("avg_rank_ic_future_rv", ascending=False)
    if not ext_split.empty:
        best = ext_split.iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Best calibrated signal", best["signal_name"])
        c2.metric("Avg daily rank IC", _number(best["avg_rank_ic_future_rv"]))
        c3.metric("Top-bottom hit-rate spread", _percent(best["avg_top_bottom_hit_rate_spread"]))
        st.plotly_chart(
            _ranking_chart(ext_summary, split, "avg_rank_ic_future_rv", "Calibrated extension: rank-IC comparison"),
            use_container_width=True,
        )
        st.caption("The calibrated branch shows extra engineering effort, but it is not the main headline result.")
        st.dataframe(_formatted_bucket_table(ext_metrics_df, split), use_container_width=True, hide_index=True)
    else:
        st.info("Missing calibrated extension summary rows.")

    st.subheader("Saved figures")
    cols = st.columns(3)
    figure_paths = [
        (PHASE2_BUCKET_FUTURE_RV_FIGURE, "Future RV by bucket"),
        (PHASE2_BUCKET_HIT_RATE_FIGURE, "High-RV hit rate by bucket"),
        (PHASE2_BUCKET_RANK_IC_FIGURE, "Rank-IC summary"),
    ]
    for col, (path, caption) in zip(cols, figure_paths):
        with col:
            if path.exists():
                st.image(str(path), caption=caption, use_container_width=True)
            else:
                st.info(f"Missing figure: {path.name}")

    st.markdown(
        """
        **De-emphasized branches**

        - The portfolio overlay branch stayed weaker than the risk-ranking result and is therefore not part of the main demo.
        - The Yahoo/news text extension was tested and did **not** improve the core option-driven signal out of sample.
        """
    )


def main() -> None:
    st.set_page_config(page_title="WRDS Option Risk Demo", layout="wide")

    artifacts = load_artifacts()
    core_metrics = artifacts["core_metrics"]
    core_diag = artifacts["core_diag"]
    ext_metrics = artifacts["ext_metrics"]
    ext_diag = artifacts["ext_diag"]
    sup_metrics = artifacts["sup_metrics"]

    if core_metrics is None or core_diag is None or ext_metrics is None or ext_diag is None or sup_metrics is None:
        st.error("Missing required saved artifacts. Run the main WRDS pipeline outputs before launching the dashboard.")
        return

    core_metrics = _apply_signal_labels(core_metrics, BENCHMARK_SIGNAL_LABELS)
    core_diag = _apply_signal_labels(core_diag, BENCHMARK_SIGNAL_LABELS)
    ext_metrics = _apply_signal_labels(ext_metrics, EXT_SIGNAL_LABELS)
    ext_diag = _apply_signal_labels(ext_diag, EXT_SIGNAL_LABELS)
    core_summary = _summary_from_diagnostics(core_diag)

    st.title("WRDS + OptionMetrics Professor Demo")
    st.caption(
        "Professor-facing version of the project. The default story is the original fixed-split logistic result, "
        "with XGBoost and trailing-window retrains shown only as robustness benchmarks."
    )

    overview_tab, core_tab, benchmark_tab, drilldown_tab, quality_tab, appendix_tab = st.tabs(
        ["Overview", "Core Result", "Benchmark Comparison", "Single-Stock Drill-Down", "Data Quality", "Appendix"]
    )

    with overview_tab:
        _render_overview_tab(core_summary, sup_metrics)
    with core_tab:
        _render_core_result_tab(core_metrics, core_summary, core_diag)
    with benchmark_tab:
        _render_benchmark_tab(core_summary, sup_metrics)
    with drilldown_tab:
        _render_single_stock_tab()
    with quality_tab:
        _render_data_quality_tab(artifacts)
    with appendix_tab:
        _render_appendix_tab(ext_metrics, ext_diag)


if __name__ == "__main__":
    main()
