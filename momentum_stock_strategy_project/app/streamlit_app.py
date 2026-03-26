from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT_DIR / "outputs" / "reports"
METRICS_DIR = ROOT_DIR / "outputs" / "metrics"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
RAW_DIR = ROOT_DIR / "data" / "raw"
WATCHLIST_CONFIG_PATH = ROOT_DIR / "config" / "watchlist.csv"


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_history(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"])
    return frame


def _load_price_panel() -> pd.DataFrame:
    path = RAW_DIR / "price_panel.parquet"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"])
    return frame


def _sanitize_ticker(ticker: object) -> str:
    return str(ticker).strip().upper().replace(".", "-")


def _load_watchlist_config() -> pd.DataFrame:
    if not WATCHLIST_CONFIG_PATH.exists():
        return pd.DataFrame({"ticker": []})
    frame = pd.read_csv(WATCHLIST_CONFIG_PATH)
    if "ticker" not in frame.columns:
        return pd.DataFrame({"ticker": []})
    clean = [_sanitize_ticker(value) for value in frame["ticker"].dropna().tolist()]
    clean = [value for value in clean if value]
    return pd.DataFrame({"ticker": sorted(dict.fromkeys(clean))})


def _save_watchlist_config(frame: pd.DataFrame) -> int:
    tickers = [_sanitize_ticker(value) for value in frame.get("ticker", pd.Series(dtype=object)).tolist()]
    tickers = [value for value in tickers if value]
    output = pd.DataFrame({"ticker": sorted(dict.fromkeys(tickers))})
    output.to_csv(WATCHLIST_CONFIG_PATH, index=False)
    return int(len(output))


def _metric_delta(value: float, baseline: float) -> str | None:
    delta = value - baseline
    return f"{delta:+.3f}" if abs(delta) > 1e-9 else None


def _styled_rank_table(frame: pd.DataFrame, rank_col: str, score_col: str, pct_col: str) -> "pd.io.formats.style.Styler":
    display = frame.copy()
    cols = [
        col
        for col in [
            "ticker",
            "sector",
            rank_col,
            pct_col,
            score_col,
            "mom_6_1",
            "mom_12_1",
            "prox_52w_high",
            "trend_filter",
            "short_term_action",
            "position_action",
            "action_note",
        ]
        if col in display.columns
    ]
    display = display[cols]
    styled = (
        display.style.format(
            {
                rank_col: "{:.0f}",
                pct_col: "{:.1%}",
                score_col: "{:.2f}",
                "mom_6_1": "{:.1%}",
                "mom_12_1": "{:.1%}",
                "prox_52w_high": "{:.1%}",
            },
            na_rep="",
        )
        .background_gradient(subset=[score_col], cmap="Greens")
        .background_gradient(subset=[pct_col], cmap="Greens")
        .background_gradient(subset=["mom_6_1", "mom_12_1"], cmap="RdYlGn")
        .background_gradient(subset=["prox_52w_high"], cmap="YlGn")
    )
    if "position_action" in display.columns:
        action_colors = {
            "Strong Buy": "background-color: #0f9d58; color: white; font-weight: 600;",
            "Buy": "background-color: #7bd88f; color: black; font-weight: 600;",
            "Watch": "background-color: #f4f4f4; color: black;",
            "Neutral": "background-color: #f4f4f4; color: black;",
            "Reduce": "background-color: #ffd180; color: black; font-weight: 600;",
            "Sell": "background-color: #d93025; color: white; font-weight: 600;",
        }
        styled = styled.map(lambda v: action_colors.get(v, ""), subset=["short_term_action", "position_action"])
    return styled


def _styled_portfolio_table(frame: pd.DataFrame, score_col: str, rank_col: str) -> "pd.io.formats.style.Styler":
    cols = [col for col in ["ticker", "weight", rank_col, score_col] if col in frame.columns]
    display = frame[cols].copy() if not frame.empty else frame
    return (
        display.style.format({"weight": "{:.1%}", rank_col: "{:.0f}", score_col: "{:.2f}"}, na_rep="")
        .background_gradient(subset=["weight"], cmap="Blues")
        .background_gradient(subset=[score_col], cmap="Greens")
    )


def _benchmark_curve(price_panel: pd.DataFrame, ticker: str, dates: pd.Series) -> pd.DataFrame:
    if price_panel.empty:
        return pd.DataFrame()
    bench = price_panel[price_panel["ticker"] == ticker][["date", "adj_close"]].dropna().sort_values("date").copy()
    if bench.empty:
        return pd.DataFrame()
    aligned = pd.DataFrame({"date": pd.to_datetime(dates).sort_values().unique()})
    aligned = aligned.merge(bench, on="date", how="left").sort_values("date")
    aligned["adj_close"] = aligned["adj_close"].ffill()
    aligned = aligned.dropna(subset=["adj_close"]).copy()
    if aligned.empty:
        return pd.DataFrame()
    base = float(aligned["adj_close"].iloc[0])
    if base == 0:
        return pd.DataFrame()
    aligned["equity_curve"] = aligned["adj_close"] / base
    aligned["ticker"] = ticker
    return aligned


def _portfolio_history_chart(history: pd.DataFrame, history_experimental: pd.DataFrame, price_panel: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    if not history.empty:
        figure.add_trace(go.Scatter(x=history["date"], y=history["equity_curve"], name="Baseline equity", line={"color": "black"}))
        figure.add_trace(
            go.Scatter(
                x=history["date"],
                y=history["drawdown"],
                name="Baseline drawdown",
                line={"color": "#C62828", "dash": "dot"},
                yaxis="y2",
            )
        )
    if not history_experimental.empty:
        figure.add_trace(
            go.Scatter(
                x=history_experimental["date"],
                y=history_experimental["equity_curve"],
                name="Experimental equity",
                line={"color": "#1976D2"},
            )
        )
        spy_curve = _benchmark_curve(price_panel, "SPY", history["date"])
        if not spy_curve.empty:
            figure.add_trace(
                go.Scatter(
                    x=spy_curve["date"],
                    y=spy_curve["equity_curve"],
                    name="SPY benchmark",
                    line={"color": "#14833b", "dash": "dash"},
                )
            )
        mtum_curve = _benchmark_curve(price_panel, "MTUM", history["date"])
        if not mtum_curve.empty:
            figure.add_trace(
                go.Scatter(
                    x=mtum_curve["date"],
                    y=mtum_curve["equity_curve"],
                    name="MTUM benchmark",
                    line={"color": "#7b1fa2", "dash": "dash"},
                )
            )
    figure.update_layout(
        title="Portfolio history",
        yaxis_title="Growth of $1",
        yaxis2={"overlaying": "y", "side": "right", "title": "Drawdown"},
        legend={"orientation": "h"},
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
    )
    return figure


def _signal_decomposition_chart(selected_row: pd.Series, score_type: str) -> go.Figure:
    if score_type == "experimental":
        signals = [
            "mom_6_1",
            "mom_12_1",
            "mom_3_1w",
            "risk_adj_mom",
            "prox_52w_high",
            "sector_rel_strength",
            "macd_hist",
            "adx_14",
            "rsi_14",
        ]
    else:
        signals = ["mom_6_1", "mom_12_1", "mom_3_1w", "risk_adj_mom", "prox_52w_high", "sector_rel_strength"]
    decomposition = pd.DataFrame(
        {
            "signal": signals,
            "z_score": [selected_row.get(f"{signal}_z", 0.0) for signal in signals],
        }
    )
    return px.bar(decomposition, x="signal", y="z_score", color="z_score", color_continuous_scale="RdYlGn")


st.set_page_config(page_title="Momentum Stock Strategy", layout="wide")

core = _load_csv(REPORTS_DIR / "latest_core_rank.csv")
experimental = _load_csv(REPORTS_DIR / "latest_experimental_rank.csv")
early = _load_csv(REPORTS_DIR / "latest_early_rank.csv")
watchlist = _load_csv(REPORTS_DIR / "latest_watchlist_overlay.csv")
portfolio = _load_csv(REPORTS_DIR / "latest_target_portfolio.csv")
experimental_portfolio = _load_csv(REPORTS_DIR / "latest_experimental_target_portfolio.csv")
history = _load_history(PROCESSED_DIR / "portfolio_history.parquet")
history_experimental = _load_history(PROCESSED_DIR / "portfolio_history_experimental.parquet")
score_panel = _load_history(PROCESSED_DIR / "score_panel.parquet")
price_panel = _load_price_panel()
backtest = _load_json(METRICS_DIR / "backtest_summary.json")
experimental_backtest = _load_json(METRICS_DIR / "experimental_backtest_summary.json")
screener = _load_json(METRICS_DIR / "screener_summary.json")
experimental_screener = _load_json(METRICS_DIR / "experimental_screener_summary.json")
signal_comparison = _load_json(METRICS_DIR / "signal_variant_comparison.json")

st.title("Momentum Stock Strategy and Screener")
st.caption("Daily EOD screener, weekly rebalance portfolio, long-only plus cash.")
page = st.sidebar.radio("Dashboard page", ["Page 1: Research", "Page 2: Guided Screener", "Page 3: Watchlist Config"])
latest_signal_date = None
if not core.empty and "date" in core.columns:
    latest_signal_date = pd.to_datetime(core["date"]).max().date()

if page == "Page 1: Research":
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("CAGR", f"{100 * float(backtest.get('cagr', 0.0)):.1f}%")
    c2.metric("Sharpe", f"{float(backtest.get('sharpe', 0.0)):.2f}")
    c3.metric("Max Drawdown", f"{100 * float(backtest.get('max_drawdown', 0.0)):.1f}%")
    c4.metric("Avg Rank IC (21d)", f"{float(screener.get('average_rank_ic_21', 0.0)):.3f}")
    c5.metric("Top Bucket Hit Rate", f"{100 * float(screener.get('average_top20_hit_rate_21', 0.0)):.1f}%")

    st.markdown("### System design")
    st.markdown(
        """
    - `core_rank`: tradable ranking for portfolio selection
    - `early_leader_rank`: watchlist ranking for fast-improving names
    - `experimental_rank`: research ranking that adds `MACD`, `ADX`, and `RSI` from the course PDF
    - daily EOD scoring
    - weekly rebalance
    - long-only plus cash with a market regime cut
    """
    )

    if not history.empty:
        st.plotly_chart(_portfolio_history_chart(history, history_experimental, price_panel), use_container_width=True)

    if screener:
        st.markdown("### Screener validation")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Eval Dates", f"{int(screener.get('evaluation_dates', 0))}")
        s2.metric("Avg Top-Decile vs Median (21d)", f"{100 * float(screener.get('average_top_decile_vs_median_21', 0.0)):.2f}%")
        s3.metric("Avg Top 21d Fwd Return", f"{100 * float(screener.get('average_top20_forward_return_21', 0.0)):.2f}%")
        s4.metric("Avg Top 63d Fwd Return", f"{100 * float(screener.get('average_top20_forward_return_63', 0.0)):.2f}%")

    st.markdown("### Baseline vs experimental signal pack")
    if signal_comparison:
        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric(
            "Sharpe delta",
            f"{float(experimental_backtest.get('sharpe', 0.0)):.2f}",
            _metric_delta(float(experimental_backtest.get("sharpe", 0.0)), float(backtest.get("sharpe", 0.0))),
        )
        rc2.metric(
            "CAGR delta",
            f"{100 * float(experimental_backtest.get('cagr', 0.0)):.1f}%",
            f"{100 * (float(experimental_backtest.get('cagr', 0.0)) - float(backtest.get('cagr', 0.0))):+.2f}%",
        )
        rc3.metric(
            "Rank IC 21 delta",
            f"{float(experimental_screener.get('average_rank_ic_21', 0.0)):.3f}",
            _metric_delta(
                float(experimental_screener.get("average_rank_ic_21", 0.0)),
                float(screener.get("average_rank_ic_21", 0.0)),
            ),
        )
        rc4.metric(
            "Top vs median delta",
            f"{100 * float(experimental_screener.get('average_top_decile_vs_median_21', 0.0)):.2f}%",
            f"{100 * (float(experimental_screener.get('average_top_decile_vs_median_21', 0.0)) - float(screener.get('average_top_decile_vs_median_21', 0.0))):+.2f}%",
        )

    left, right = st.columns(2)
    with left:
        st.markdown("### Latest core ranking")
        st.dataframe(core.head(50), use_container_width=True, hide_index=True)
    with right:
        st.markdown("### Latest experimental ranking")
        st.dataframe(experimental.head(50), use_container_width=True, hide_index=True)

    lower_left, lower_right = st.columns(2)
    with lower_left:
        st.markdown("### Latest early leaders")
        st.dataframe(early.head(50), use_container_width=True, hide_index=True)
    with lower_right:
        st.markdown("### Personal watchlist overlay")
        st.dataframe(watchlist, use_container_width=True, hide_index=True)

    portfolio_left, portfolio_right = st.columns(2)
    with portfolio_left:
        st.markdown("### Latest baseline target portfolio")
        st.dataframe(portfolio, use_container_width=True, hide_index=True)
    with portfolio_right:
        st.markdown("### Latest experimental target portfolio")
        st.dataframe(experimental_portfolio, use_container_width=True, hide_index=True)

    if not core.empty and "sector" in core.columns:
        sector_counts = core.head(25)["sector"].value_counts().reset_index()
        sector_counts.columns = ["sector", "count"]
        sector_chart = px.bar(sector_counts, x="sector", y="count", title="Sector mix of top 25 baseline ranks", color="count")
        st.plotly_chart(sector_chart, use_container_width=True)

    if not score_panel.empty:
        eval_panel = score_panel.dropna(subset=["forward_return_21"]).copy()
        if not eval_panel.empty:
            eval_date = pd.to_datetime(eval_panel["date"]).max()
            eval_cross_section = eval_panel[eval_panel["date"] == eval_date].copy()
            scatter = px.scatter(
                eval_cross_section,
                x="core_score",
                y="forward_return_21",
                color="sector",
                hover_data=["ticker", "core_rank", "core_rank_pct", "prox_52w_high"],
                title=f"Latest fully evaluable cross-section: score vs 21-day forward return ({eval_date.date()})",
            )
            st.plotly_chart(scatter, use_container_width=True)

    if not core.empty:
        st.markdown("### Stock-level signal decomposition")
        selected_ticker = st.selectbox("Ticker", options=core["ticker"].tolist(), index=0)
        score_type = st.radio("Decomposition", ["baseline", "experimental"], horizontal=True)
        source = experimental if score_type == "experimental" and not experimental.empty else core
        selected = source[source["ticker"] == selected_ticker]
        if not selected.empty:
            selected_row = selected.iloc[0]
            bar = _signal_decomposition_chart(selected_row, score_type)
            bar.update_layout(title=f"{selected_ticker} {score_type} signal decomposition")
            st.plotly_chart(bar, use_container_width=True)

    if not core.empty:
        scatter = px.scatter(
            core.head(100),
            x="core_score",
            y="mom_6_1",
            color="sector",
            hover_data=["ticker", "core_rank", "core_rank_pct", "prox_52w_high"],
            title="Latest cross-section: core score vs 6-1 momentum",
        )
        st.plotly_chart(scatter, use_container_width=True)

elif page == "Page 2: Guided Screener":
    st.subheader("Guided Screener")
    if latest_signal_date is not None:
        st.caption(f"Latest signal date: {latest_signal_date}. This page is meant to be your quick morning check.")
    st.markdown(
        """
This page is the simpler version.

Use it when you want quick answers to 3 questions:

1. Which stocks look strongest right now?
2. Why does a given stock rank well or badly?
3. Did the extra course indicators help the screener?
"""
    )

    top1, top2, top3, top4 = st.columns(4)
    top1.metric("Baseline Sharpe", f"{float(backtest.get('sharpe', 0.0)):.2f}")
    top2.metric("Experimental Sharpe", f"{float(experimental_backtest.get('sharpe', 0.0)):.2f}")
    top3.metric("Baseline Rank IC 21d", f"{float(screener.get('average_rank_ic_21', 0.0)):.3f}")
    top4.metric("Experimental Rank IC 21d", f"{float(experimental_screener.get('average_rank_ic_21', 0.0)):.3f}")

    st.markdown("### 1. Portfolio growth")
    st.caption(
        "Black is the original strategy. Blue is the research version with MACD, ADX, and RSI added. "
        "Green dashed is SPY. Purple dashed is MTUM, a well-known momentum ETF. "
        "Use this chart to see whether the strategy beats a passive benchmark you could buy directly."
    )
    if not history.empty:
        st.plotly_chart(_portfolio_history_chart(history, history_experimental, price_panel), use_container_width=True)

    st.markdown("### 2. Best stocks right now")
    st.caption(
        "Green rows are stronger. This is the easiest screener table to read. "
        "Use baseline first, then compare with the experimental version to see if the new indicators reshuffle names."
    )
    gt1, gt2 = st.columns(2)
    with gt1:
        st.markdown("#### Baseline screener")
        if not core.empty:
            st.dataframe(_styled_rank_table(core.head(30), "core_rank", "core_score", "core_rank_pct"), use_container_width=True, hide_index=True)
    with gt2:
        st.markdown("#### Experimental screener")
        if not experimental.empty:
            st.dataframe(
                _styled_rank_table(experimental.head(30), "experimental_rank", "experimental_score", "experimental_rank_pct"),
                use_container_width=True,
                hide_index=True,
            )

    st.markdown("### 3. Your personal watchlist")
    st.caption(
        "This table is the practical one if you mainly care about your own names. "
        "Check the `short_term_action` for a faster 2-6 week read and `position_action` for the main medium-term stance."
    )
    if not watchlist.empty:
        st.dataframe(_styled_rank_table(watchlist, "core_rank", "core_score", "core_rank_pct"), use_container_width=True, hide_index=True)
        st.markdown(
            """
Action labels:

- `Strong Buy`: top-tier setup with strong rank and trend support
- `Buy`: constructive setup, but not the very strongest
- `Neutral`: mixed signals, usually watch rather than act
- `Reduce`: not favored right now; avoid adding and consider trimming
- `Sell`: clear deterioration across multiple dimensions
"""
        )

    st.markdown("### 4. Why a stock is strong or weak")
    st.caption(
        "This bar chart breaks a stock into signal pieces. Positive bars help the stock. Negative bars hurt it. "
        "The experimental view adds MACD, ADX, and RSI from your course PDF."
    )
    if not core.empty:
        chosen = st.selectbox("Choose a stock", options=core["ticker"].tolist(), index=0, key="guided_ticker")
        guided_view = st.radio("View", ["baseline", "experimental"], horizontal=True, key="guided_view")
        source = experimental if guided_view == "experimental" and not experimental.empty else core
        selected = source[source["ticker"] == chosen]
        if not selected.empty:
            row = selected.iloc[0]
            chart = _signal_decomposition_chart(row, guided_view)
            chart.update_layout(title=f"{chosen}: how the score is built")
            st.plotly_chart(chart, use_container_width=True)
            st.markdown(
                """
Signals used here:

- `mom_12_1`: strong medium-term momentum
- `mom_6_1`: shorter medium-term momentum
- `mom_3_1w`: early acceleration
- `risk_adj_mom`: momentum after penalizing volatility
- `prox_52w_high`: whether price is near its 52-week high
- `sector_rel_strength`: strength relative to the sector ETF
- `MACD`: short-vs-long EMA momentum
- `ADX`: trend strength
- `RSI`: fast momentum / overbought-overextended style indicator
"""
            )

    st.markdown("### 5. Current model portfolios")
    st.caption(
        "These are the names the weekly portfolio engine would hold. "
        "Use them as a stricter shortlist than the screener table."
    )
    pt1, pt2 = st.columns(2)
    with pt1:
        st.markdown("#### Baseline portfolio")
        if not portfolio.empty:
            st.dataframe(_styled_portfolio_table(portfolio, "core_score", "core_rank"), use_container_width=True, hide_index=True)
    with pt2:
        st.markdown("#### Experimental portfolio")
        if not experimental_portfolio.empty:
            st.dataframe(
                _styled_portfolio_table(experimental_portfolio, "experimental_score", "experimental_rank"),
                use_container_width=True,
                hide_index=True,
            )

    st.markdown("### 6. Did the added course indicators help?")
    st.caption(
        "This section answers the research question directly. "
        "If the deltas are positive, the extra indicators improved that metric. If not, they added noise."
    )
    comparison_table = pd.DataFrame(
        [
            {
                "metric": "Sharpe",
                "baseline": float(backtest.get("sharpe", 0.0)),
                "experimental": float(experimental_backtest.get("sharpe", 0.0)),
                "delta": float(experimental_backtest.get("sharpe", 0.0)) - float(backtest.get("sharpe", 0.0)),
            },
            {
                "metric": "CAGR",
                "baseline": float(backtest.get("cagr", 0.0)),
                "experimental": float(experimental_backtest.get("cagr", 0.0)),
                "delta": float(experimental_backtest.get("cagr", 0.0)) - float(backtest.get("cagr", 0.0)),
            },
            {
                "metric": "Rank IC 21d",
                "baseline": float(screener.get("average_rank_ic_21", 0.0)),
                "experimental": float(experimental_screener.get("average_rank_ic_21", 0.0)),
                "delta": float(experimental_screener.get("average_rank_ic_21", 0.0))
                - float(screener.get("average_rank_ic_21", 0.0)),
            },
            {
                "metric": "Top vs Median 21d",
                "baseline": float(screener.get("average_top_decile_vs_median_21", 0.0)),
                "experimental": float(experimental_screener.get("average_top_decile_vs_median_21", 0.0)),
                "delta": float(experimental_screener.get("average_top_decile_vs_median_21", 0.0))
                - float(screener.get("average_top_decile_vs_median_21", 0.0)),
            },
        ]
    )
    st.dataframe(
        comparison_table.style.format({"baseline": "{:.3f}", "experimental": "{:.3f}", "delta": "{:+.3f}"}, na_rep="")
        .background_gradient(subset=["delta"], cmap="RdYlGn"),
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "Read this page as a screener and research dashboard, not as blind auto-trading advice. "
        "The cleaner question is whether the ranking quality is improving, not whether every top stock should be bought immediately."
    )

else:
    st.subheader("Watchlist Config")
    st.caption("Edit your personal watchlist here. This page writes directly to `config/watchlist.csv`.")

    if "watchlist_editor_df" not in st.session_state:
        st.session_state["watchlist_editor_df"] = _load_watchlist_config()

    st.markdown(
        """
Use this page to manage the tickers followed in `watchlist` mode.

- add or remove rows
- type tickers like `AAPL`, `TSLA`, `NVDA`
- tickers are normalized to uppercase automatically
- after saving, rerun `run_watchlist.bat` to refresh rankings and portfolio outputs
"""
    )

    edited = st.data_editor(
        st.session_state["watchlist_editor_df"],
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key="watchlist_editor",
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Save Watchlist", type="primary", use_container_width=True):
            count = _save_watchlist_config(pd.DataFrame(edited))
            st.session_state["watchlist_editor_df"] = _load_watchlist_config()
            st.success(f"Saved {count} watchlist tickers to config/watchlist.csv")
    with c2:
        if st.button("Reload From File", use_container_width=True):
            st.session_state["watchlist_editor_df"] = _load_watchlist_config()
            st.rerun()

    current_file = _load_watchlist_config()
    st.markdown("### Current saved watchlist")
    st.dataframe(current_file, use_container_width=True, hide_index=True)

    st.info(
        "Saving this page updates the config file only. "
        "To update the screener outputs, rerun `run_watchlist.bat` or `python -m src.main --phase all --universe-mode watchlist`."
    )
