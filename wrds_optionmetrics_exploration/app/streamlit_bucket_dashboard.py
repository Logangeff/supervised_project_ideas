from __future__ import annotations

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

CORE_BUCKET_METRICS_CSV = METRICS_DIR / "phase2_bucket_analysis_metrics.csv"
CORE_BUCKET_DIAGNOSTICS_CSV = METRICS_DIR / "phase2_bucket_analysis_diagnostics.csv"
EXT_BUCKET_METRICS_CSV = METRICS_DIR / "phase2_bucket_analysis_extension_metrics.csv"
EXT_BUCKET_DIAGNOSTICS_CSV = METRICS_DIR / "phase2_bucket_analysis_extension_diagnostics.csv"

PHASE2_BUCKET_FUTURE_RV_FIGURE = FIGURES_DIR / "phase2_bucket_future_rv.png"
PHASE2_BUCKET_HIT_RATE_FIGURE = FIGURES_DIR / "phase2_bucket_hit_rate.png"
PHASE2_BUCKET_RANK_IC_FIGURE = FIGURES_DIR / "phase2_bucket_rank_ic.png"
SURFACE_EXTENSION_PREDICTIONS_CSV = METRICS_DIR / "surface_extension_predictions.csv"
SURFACE_EXTENSION_PANEL_PATH = PROCESSED_DIR / "surface_extension_panel.parquet"

TEMPLATE = "plotly_white"
CORE_SIGNAL_LABELS_ORIGINAL = {
    "stock_only_prob": "Stock Only",
    "option_only_prob": "Option Only",
    "all_features_prob": "Stock + Option + Surface",
}
CORE_SIGNAL_LABELS_XGB = {
    "stock_only_xgb_prob": "Stock Only (XGBoost)",
    "option_only_xgb_prob": "Option Only (XGBoost)",
    "all_features_xgb_prob": "Stock + Option + Surface (XGBoost)",
}
CORE_SIGNAL_LABELS_TRAILING = {
    "stock_only_trail2y_prob": "Stock Only (2Y Trailing)",
    "stock_only_trail5y_prob": "Stock Only (5Y Trailing)",
    "option_only_trail2y_prob": "Option Only (2Y Trailing)",
    "option_only_trail5y_prob": "Option Only (5Y Trailing)",
    "all_features_trail2y_prob": "Stock + Option + Surface (2Y Trailing)",
    "all_features_trail5y_prob": "Stock + Option + Surface (5Y Trailing)",
}
CORE_SIGNAL_LABELS = {**CORE_SIGNAL_LABELS_ORIGINAL, **CORE_SIGNAL_LABELS_XGB, **CORE_SIGNAL_LABELS_TRAILING}
EXT_SIGNAL_LABELS = {
    "beta_only_prob": "Beta Only",
    "option_beta_prob": "Option + Betas",
    "all_extensions_prob": "All Extensions",
}
SIGNAL_COLORS = {
    "Stock Only": "#6c757d",
    "Option Only": "#1f77b4",
    "Stock + Option + Surface": "#d62728",
    "Stock Only (XGBoost)": "#adb5bd",
    "Option Only (XGBoost)": "#17becf",
    "Stock + Option + Surface (XGBoost)": "#e377c2",
    "Stock Only (2Y Trailing)": "#8c564b",
    "Stock Only (5Y Trailing)": "#c49c94",
    "Option Only (2Y Trailing)": "#17a2b8",
    "Option Only (5Y Trailing)": "#9adbe8",
    "Stock + Option + Surface (2Y Trailing)": "#9467bd",
    "Stock + Option + Surface (5Y Trailing)": "#c5b0d5",
    "Beta Only": "#9467bd",
    "Option + Betas": "#2ca02c",
    "All Extensions": "#ff7f0e",
}


def _load_csv(path: Path, parse_dates: list[str] | None = None) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=parse_dates)


def _assign_bucket(series: pd.Series) -> pd.Series:
    ranks = series.rank(method="first", pct=True)
    buckets = (ranks * 5).apply(lambda value: min(5, max(1, int(value) if float(value).is_integer() else int(value) + 1)))
    return pd.Series(buckets.values, index=series.index, dtype=int)


@st.cache_data(show_spinner=False)
def load_bucket_artifacts() -> dict[str, pd.DataFrame | None]:
    return {
        "core_metrics": _load_csv(CORE_BUCKET_METRICS_CSV),
        "core_diag": _load_csv(CORE_BUCKET_DIAGNOSTICS_CSV, parse_dates=["trade_date"]),
        "ext_metrics": _load_csv(EXT_BUCKET_METRICS_CSV),
        "ext_diag": _load_csv(EXT_BUCKET_DIAGNOSTICS_CSV, parse_dates=["trade_date"]),
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

    for signal_column in CORE_SIGNAL_LABELS:
        if signal_column in merged.columns:
            merged[f"{signal_column}_bucket"] = (
                merged.groupby(["split", "trade_date"], sort=False)[signal_column].transform(_assign_bucket)
            )

    merged["future_high_rv_label"] = merged["label"].astype(int)
    return merged


def _apply_labels(frame: pd.DataFrame, label_map: dict[str, str]) -> pd.DataFrame:
    labeled = frame.copy()
    labeled = labeled[labeled["signal_column"].isin(label_map)].copy()
    labeled["signal_name"] = labeled["signal_column"].map(label_map).fillna(labeled["signal_column"])
    return labeled


def _core_label_map(display_mode: str) -> dict[str, str]:
    if display_mode == "Original project only":
        return CORE_SIGNAL_LABELS_ORIGINAL
    if display_mode == "XGBoost only":
        return CORE_SIGNAL_LABELS_XGB
    if display_mode == "Trailing only":
        return CORE_SIGNAL_LABELS_TRAILING
    if display_mode == "Original + XGBoost":
        return {**CORE_SIGNAL_LABELS_ORIGINAL, **CORE_SIGNAL_LABELS_XGB}
    if display_mode == "Original + Trailing":
        return {**CORE_SIGNAL_LABELS_ORIGINAL, **CORE_SIGNAL_LABELS_TRAILING}
    return CORE_SIGNAL_LABELS


def _summary_from_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    summary = (
        frame.groupby(["signal_name", "split"], as_index=False)
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
    return summary


def _metric_card_row(summary: pd.DataFrame, split: str) -> None:
    split_summary = summary[summary["split"] == split].sort_values("avg_rank_ic_future_rv", ascending=False)
    if split_summary.empty:
        st.info(f"No {split} summary available.")
        return
    best = split_summary.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Signal", best["signal_name"])
    c2.metric("Avg Daily Rank IC", f"{best['avg_rank_ic_future_rv']:.3f}")
    c3.metric("Top-Bottom Future RV Spread", f"{best['avg_top_bottom_future_rv_spread']:.3f}")
    c4.metric("Top-Bottom Hit-Rate Spread", f"{100 * best['avg_top_bottom_hit_rate_spread']:.1f}%")


def _ranking_chart(summary: pd.DataFrame, split: str, metric_col: str, metric_title: str) -> go.Figure:
    plot_df = summary[summary["split"] == split].sort_values(metric_col, ascending=True)
    fig = go.Figure(
        go.Bar(
            x=plot_df[metric_col],
            y=plot_df["signal_name"],
            orientation="h",
            marker_color=[SIGNAL_COLORS.get(name, "#4c78a8") for name in plot_df["signal_name"]],
            text=[f"{value:.3f}" for value in plot_df[metric_col]],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=metric_title,
        xaxis_title=metric_title,
        yaxis_title="Signal",
        template=TEMPLATE,
        height=320,
        margin={"l": 10, "r": 20, "t": 50, "b": 10},
    )
    return fig


def _bucket_curve_chart(frame: pd.DataFrame, split: str, value_col: str, title: str, yaxis_title: str) -> go.Figure:
    plot_df = frame[frame["split"] == split].copy()
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
    return fig


def _rolling_rank_ic_chart(frame: pd.DataFrame, split: str, window: int = 21) -> go.Figure:
    plot_df = frame[frame["split"] == split].sort_values(["signal_name", "trade_date"]).copy()
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


def _monotonicity_chart(summary: pd.DataFrame, split: str) -> go.Figure:
    plot_df = summary[summary["split"] == split].copy()
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=plot_df["signal_name"],
            y=100 * plot_df["monotonic_future_rv_share"],
            name="Future RV monotonic share",
            marker_color="#1f77b4",
        )
    )
    fig.add_trace(
        go.Bar(
            x=plot_df["signal_name"],
            y=100 * plot_df["monotonic_hit_rate_share"],
            name="Hit-rate monotonic share",
            marker_color="#ff7f0e",
        )
    )
    fig.update_layout(
        title="Share of Dates With Clean Bucket Ordering",
        xaxis_title="Signal",
        yaxis_title="Percent of trade dates",
        barmode="group",
        template=TEMPLATE,
        height=360,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )
    return fig


def _formatted_bucket_table(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    table = frame[frame["split"] == split].copy()
    table = table[
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
    return table


def _family_comparison_table(summary: pd.DataFrame, split: str) -> pd.DataFrame:
    split_df = summary[summary["split"] == split].copy()
    comparisons = []
    family_specs = [
        ("Stock Only", "Stock Only (XGBoost)", "Stock Only (2Y Trailing)", "Stock Only (5Y Trailing)"),
        ("Option Only", "Option Only (XGBoost)", "Option Only (2Y Trailing)", "Option Only (5Y Trailing)"),
        (
            "Stock + Option + Surface",
            "Stock + Option + Surface (XGBoost)",
            "Stock + Option + Surface (2Y Trailing)",
            "Stock + Option + Surface (5Y Trailing)",
        ),
    ]
    for base_name, xgb_name, trail2_name, trail5_name in family_specs:
        rows = {
            name: split_df[split_df["signal_name"] == name].iloc[0]
            for name in (base_name, xgb_name, trail2_name, trail5_name)
            if not split_df[split_df["signal_name"] == name].empty
        }
        if base_name not in rows:
            continue
        base_row = rows[base_name]
        payload = {
            "Signal family": base_name,
            "Baseline rank IC": base_row["avg_rank_ic_future_rv"],
            "Baseline RV spread": base_row["avg_top_bottom_future_rv_spread"],
            "Baseline hit-rate spread": base_row["avg_top_bottom_hit_rate_spread"],
        }
        for variant_name, short_label in (
            (xgb_name, "XGBoost"),
            (trail2_name, "Trail 2Y"),
            (trail5_name, "Trail 5Y"),
        ):
            if variant_name not in rows:
                continue
            variant_row = rows[variant_name]
            payload[f"{short_label} rank IC"] = variant_row["avg_rank_ic_future_rv"]
            payload[f"{short_label} delta rank IC"] = variant_row["avg_rank_ic_future_rv"] - base_row["avg_rank_ic_future_rv"]
            payload[f"{short_label} RV spread"] = variant_row["avg_top_bottom_future_rv_spread"]
            payload[f"{short_label} delta RV spread"] = (
                variant_row["avg_top_bottom_future_rv_spread"] - base_row["avg_top_bottom_future_rv_spread"]
            )
            payload[f"{short_label} hit-rate spread"] = variant_row["avg_top_bottom_hit_rate_spread"]
            payload[f"{short_label} delta hit-rate spread"] = (
                variant_row["avg_top_bottom_hit_rate_spread"] - base_row["avg_top_bottom_hit_rate_spread"]
            )
        comparisons.append(payload)
    return pd.DataFrame(comparisons)


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
    signal_label = CORE_SIGNAL_LABELS.get(signal_column, signal_column)
    bucket_column = f"{signal_column}_bucket"

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.28, 0.20, 0.28, 0.24],
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

    price_base = frame["prc"].iloc[0] if not frame["prc"].dropna().empty else 1.0
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
            name=f"{signal_label} probability ",
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

    fig.update_layout(
        title="Stock Timeline: prediction vs realized path",
        template=TEMPLATE,
        height=980,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
    )
    fig.update_xaxes(rangeslider={"visible": True}, row=4, col=1, title_text="Trade date")
    return fig


def _stock_summary_cards(frame: pd.DataFrame, signal_column: str) -> None:
    latest = frame.sort_values("trade_date").iloc[-1]
    bucket_column = f"{signal_column}_bucket"
    avg_bucket = frame[bucket_column].mean() if bucket_column in frame.columns else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest predicted risk", f"{100 * latest[signal_column]:.1f}%")
    c2.metric("Latest future-RV label", "High-RV" if int(latest["future_high_rv_label"]) == 1 else "Normal-RV")
    c3.metric("Average future RV (20d)", f"{100 * frame['future_rv_20d'].mean():.1f}%")
    c4.metric("Average daily bucket", f"{avg_bucket:.2f}" if avg_bucket is not None else "N/A")


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
    return table.rename(columns=rename_map).sort_values("Date", ascending=False)


def _render_stock_drilldown(display_mode: str) -> None:
    st.header("Single-Stock Drill-Down")
    st.caption(
        "Use this page to inspect whether the model's predicted risk for one stock lines up with that stock's "
        "actual realized volatility path and future high-volatility regime outcomes."
    )

    panel = load_core_stock_drilldown_panel()
    if panel is None or panel.empty:
        st.warning("Missing stock-level core prediction artifacts for the drill-down page.")
        return

    control_col1, control_col2, control_col3 = st.columns([1.1, 1.1, 1.6])
    with control_col1:
        split = st.selectbox("Evaluation split", ["test", "validation"], index=0)
    filtered = panel[panel["split"] == split].copy()

    if filtered.empty:
        st.info(f"No {split} stock rows available.")
        return

    ticker_options = sorted(filtered["universe_ticker"].dropna().unique().tolist())
    default_ticker = "TSLA" if "TSLA" in ticker_options else ticker_options[0]

    with control_col2:
        ticker = st.selectbox("Ticker", ticker_options, index=ticker_options.index(default_ticker))
    with control_col3:
        active_label_map = _core_label_map(display_mode)
        available_signal_labels = list(active_label_map.values())
        default_label = "Stock + Option + Surface"
        if default_label not in available_signal_labels:
            default_label = available_signal_labels[0]
        signal_label = st.selectbox("Signal view", available_signal_labels, index=available_signal_labels.index(default_label))

    inverse_label_map = {label: column for column, label in active_label_map.items()}
    signal_column = inverse_label_map[signal_label]

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
    st.plotly_chart(_stock_timeline_chart(stock_frame, signal_column), use_container_width=True)
    st.caption(
        "Red background bands mark dates where the stock's actual future 20-day label was high-volatility. "
        "The dotted red line is the selected model probability, the purple step line is the daily cross-sectional bucket, "
        "and the orange line shows the stock's realized future 20-day volatility."
    )

    st.subheader("Stock Detail Table")
    st.dataframe(_stock_detail_table(stock_frame, signal_column), use_container_width=True, hide_index=True)


def _static_figure_panel() -> None:
    st.subheader("Saved Research Figures")
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


def _render_bucket_dashboard(
    title: str,
    explanation: str,
    metrics_df: pd.DataFrame | None,
    diagnostics_df: pd.DataFrame | None,
    label_map: dict[str, str],
    show_benchmark_comparison: bool = False,
) -> None:
    if metrics_df is None or diagnostics_df is None:
        st.warning("Missing saved bucket-analysis files for this section.")
        return

    metrics_df = _apply_labels(metrics_df, label_map)
    diagnostics_df = _apply_labels(diagnostics_df, label_map)
    summary_df = _summary_from_diagnostics(diagnostics_df)

    st.header(title)
    st.caption(explanation)

    split = st.radio(
        "Evaluation split",
        options=["test", "validation"],
        horizontal=True,
        key=f"{title}_split",
    )

    _metric_card_row(summary_df, split)

    left, right = st.columns([1.15, 1.0])
    with left:
        st.plotly_chart(
            _ranking_chart(summary_df, split, "avg_rank_ic_future_rv", "Average Daily Rank IC vs Future RV"),
            use_container_width=True,
        )
    with right:
        st.plotly_chart(
            _ranking_chart(summary_df, split, "avg_top_bottom_hit_rate_spread", "Top-Bottom High-RV Hit-Rate Spread"),
            use_container_width=True,
        )

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            _bucket_curve_chart(metrics_df, split, "average_future_rv_20d", "Future 20-Day Realized Volatility by Bucket", "Avg future RV (20d)"),
            use_container_width=True,
        )
    with c2:
        st.plotly_chart(
            _bucket_curve_chart(metrics_df, split, "high_rv_hit_rate", "High-Volatility Regime Hit Rate by Bucket", "Hit rate"),
            use_container_width=True,
        )

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(
            _bucket_curve_chart(metrics_df, split, "average_abs_ret_next_1d", "Absolute Next-Day Return by Bucket", "Avg |next-day return|"),
            use_container_width=True,
        )
    with c4:
        st.plotly_chart(_monotonicity_chart(summary_df, split), use_container_width=True)

    if show_benchmark_comparison:
        comparison_table = _family_comparison_table(summary_df, split)
        if not comparison_table.empty:
            st.subheader("Benchmark Comparison vs Original Logistic")
            st.dataframe(comparison_table, use_container_width=True, hide_index=True)

    st.plotly_chart(_rolling_rank_ic_chart(diagnostics_df, split), use_container_width=True)

    st.subheader("Bucket Detail Table")
    st.dataframe(_formatted_bucket_table(metrics_df, split), use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="WRDS Bucket Analysis Dashboard", layout="wide")
    st.title("WRDS + OptionMetrics Bucket Analysis")
    st.caption(
        "Visual dashboard for the successful Phase 2 bucket-analysis result. "
        "Higher predicted-risk buckets should map to higher future realized volatility and higher high-volatility hit rates."
    )

    artifacts = load_bucket_artifacts()

    overview_col1, overview_col2 = st.columns(2)
    with overview_col1:
        st.markdown(
            """
            **What this dashboard is testing**

            - Sort stocks each day into 5 buckets by predicted high-volatility probability.
            - Check whether higher buckets actually realize higher future 20-day volatility.
            - Check whether higher buckets also have higher high-volatility-regime hit rates.
            """
        )
    with overview_col2:
        st.markdown(
            """
            **How to read success**

            - Bucket 5 should sit above Bucket 1 on future RV and hit-rate charts.
            - Daily rank IC should stay positive on average.
            - Monotonic bucket ordering should happen often, especially on validation.
            """
        )

    display_mode = "Original project only"

    core_tab, drilldown_tab, ext_tab, figures_tab = st.tabs(
        ["Core Surface-Common Result", "Single-Stock Drill-Down", "Calibrated Extension", "Saved Figures"]
    )

    with core_tab:
        display_mode = st.radio(
            "Core comparison view",
            options=[
                "Original project only",
                "XGBoost only",
                "Trailing only",
                "Original + XGBoost",
                "Original + Trailing",
                "All benchmarks",
            ],
            horizontal=True,
            index=0,
        )
        _render_bucket_dashboard(
            title="Core Bucket Analysis",
            explanation=(
                "This is the main result for the project: stock-only vs option-only vs combined surface-common signals "
                "on the identical complete-case panel. You can keep the original logistic-regression project view, "
                "switch to XGBoost or trailing-window benchmarks, or compare them side by side."
            ),
            metrics_df=artifacts["core_metrics"],
            diagnostics_df=artifacts["core_diag"],
            label_map=_core_label_map(display_mode),
            show_benchmark_comparison=(display_mode in {"Original + XGBoost", "Original + Trailing", "All benchmarks"}),
        )

    with drilldown_tab:
        _render_stock_drilldown(display_mode)

    with ext_tab:
        _render_bucket_dashboard(
            title="Calibrated-Surface Extension",
            explanation=(
                "This is the calibrated beta-factor extension. It is useful as a comparison branch, "
                "but the core option-aware signal remains the strongest main story."
            ),
            metrics_df=artifacts["ext_metrics"],
            diagnostics_df=artifacts["ext_diag"],
            label_map=EXT_SIGNAL_LABELS,
        )

    with figures_tab:
        _static_figure_panel()


if __name__ == "__main__":
    main()
