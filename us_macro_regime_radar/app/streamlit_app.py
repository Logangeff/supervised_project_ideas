from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parent.parent
HISTORY_PAYLOAD = ROOT_DIR / "outputs" / "dashboard" / "history_payload.json"
LATEST_SNAPSHOT = ROOT_DIR / "outputs" / "dashboard" / "latest_snapshot.json"
FORECAST_PAYLOAD = ROOT_DIR / "outputs" / "dashboard" / "forecast_payload.json"

PHASE_COLORS = {
    "Expansion": "#1b9e77",
    "Slowdown": "#d95f02",
    "Contraction": "#d73027",
    "Recovery": "#4575b4",
}


def _load_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _phase_segments(history: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    segments: list[tuple[pd.Timestamp, pd.Timestamp, str]] = []
    if history.empty:
        return segments
    start_idx = 0
    for idx in range(1, history.shape[0]):
        if history.iloc[idx]["current_phase"] != history.iloc[idx - 1]["current_phase"]:
            segments.append((history.iloc[start_idx]["date"], history.iloc[idx - 1]["date"], history.iloc[start_idx]["current_phase"]))
            start_idx = idx
    segments.append((history.iloc[start_idx]["date"], history.iloc[-1]["date"], history.iloc[start_idx]["current_phase"]))
    return segments


def _segment_bounds(start_date: pd.Timestamp, end_date: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    # Convert month-end labels into continuous month coverage with no white gaps.
    x0 = start_date.to_period("M").to_timestamp()
    x1 = (end_date.to_period("M") + 1).to_timestamp()
    return x0, x1


def _add_phase_background(figure: go.Figure, history: pd.DataFrame, opacity: float = 0.18) -> None:
    for start_date, end_date, phase_name in _phase_segments(history):
        x0, x1 = _segment_bounds(pd.Timestamp(start_date), pd.Timestamp(end_date))
        figure.add_vrect(
            x0=x0,
            x1=x1,
            fillcolor=PHASE_COLORS.get(phase_name, "#cccccc"),
            opacity=opacity,
            line_width=0,
            layer="below",
        )


def _phase_strip_chart(history: pd.DataFrame) -> go.Figure:
    strip = go.Figure()
    seen_phases: set[str] = set()
    for start_date, end_date, phase_name in _phase_segments(history):
        x0, x1 = _segment_bounds(pd.Timestamp(start_date), pd.Timestamp(end_date))
        strip.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[0, 0],
                mode="lines",
                line={
                    "width": 18,
                    "color": PHASE_COLORS.get(phase_name, "#cccccc"),
                },
                name=phase_name,
                showlegend=phase_name not in seen_phases,
                hovertemplate=(
                    f"{phase_name}<br>"
                    f"{pd.Timestamp(start_date).strftime('%Y-%m')} to {pd.Timestamp(end_date).strftime('%Y-%m')}"
                    "<extra></extra>"
                ),
            )
        )
        seen_phases.add(phase_name)
    strip.update_layout(
        title="Historical Phase Strip",
        xaxis_title="Date",
        yaxis_title="Phase",
        yaxis={
            "tickmode": "array",
            "tickvals": [0],
            "ticktext": ["Phase"],
            "range": [-0.6, 0.6],
        },
        template="plotly_white",
        height=180,
        legend_title="Phase",
        hovermode="x unified",
    )
    return strip


def main() -> None:
    st.set_page_config(page_title="US Macro Regime Radar", layout="wide")
    latest = _load_json(LATEST_SNAPSHOT)
    history_payload = _load_json(HISTORY_PAYLOAD)
    forecast_payload = _load_json(FORECAST_PAYLOAD)

    if latest is None or history_payload is None or forecast_payload is None:
        st.error("Missing dashboard payloads. Run `python -m src.main --phase all` first.")
        return

    history = pd.DataFrame(history_payload["rows"])
    history["date"] = pd.to_datetime(history["date"])
    plot_history = history[history["recession_risk_probability"].notna()].copy()
    if plot_history.empty:
        plot_history = history.copy()
    first_history_date = plot_history["date"].min()
    latest_history_date = plot_history["date"].max()
    default_start = max(first_history_date, latest_history_date - pd.DateOffset(years=10))
    default_end = latest_history_date + pd.offsets.MonthEnd(6)
    forecast_long = pd.DataFrame(forecast_payload["phase_probabilities_long"])

    st.title("US Macro Regime Radar")
    st.caption("Current business-cycle view plus forward transition-risk probabilities.")

    card1, card2, card3, card4 = st.columns(4)
    card1.metric("Current Phase", latest["current_phase"])
    card2.metric("Recession Within 6M", f"{100 * latest['recession_within_6m_probability']:.1f}%")
    card3.metric("Most Likely Next Phase", latest["most_likely_next_phase"])
    card4.metric("Next-Phase Confidence", f"{100 * latest['next_phase_confidence']:.1f}%")
    st.caption(f"Forecast origin date: {pd.to_datetime(latest['date']).strftime('%Y-%m-%d')}")

    timeline = go.Figure()
    _add_phase_background(timeline, plot_history, opacity=0.20)
    recession_months = plot_history[plot_history["is_recession_month"] == 1]
    if not recession_months.empty:
        for _, row in recession_months.iterrows():
            timeline.add_vrect(
                x0=row["date"],
                x1=row["date"] + pd.offsets.MonthEnd(0),
                fillcolor="#7f0000",
                opacity=0.10,
                line_width=0,
                layer="below",
            )
    timeline.add_trace(
        go.Scatter(
            x=plot_history["date"],
            y=plot_history["recession_risk_probability"],
            mode="lines",
            name="Recession within 6M probability",
            line={"color": "#111111", "width": 2},
            line_shape="spline",
            line_smoothing=0.65,
        )
    )
    forecast_origin = pd.to_datetime(latest["date"])
    future_dates = [forecast_origin] + [forecast_origin + pd.offsets.MonthEnd(h) for h in range(1, 7)]
    recession_line = [latest["recession_within_6m_probability"]] * len(future_dates)
    timeline.add_vrect(
        x0=forecast_origin,
        x1=future_dates[-1],
        fillcolor="#7f0000",
        opacity=0.04,
        line_width=0,
        layer="below",
    )
    timeline.add_vline(
        x=forecast_origin,
        line_dash="dot",
        line_color="#7f0000",
        opacity=0.7,
    )
    timeline.add_trace(
        go.Scatter(
            x=future_dates,
            y=recession_line,
            mode="lines+markers",
            name="Current 6M risk view",
            line={"color": "#7f0000", "dash": "dash", "width": 2},
            line_shape="spline",
            marker={"size": 7, "color": "#7f0000"},
        )
    )
    timeline.update_layout(
        title="Historical recession-within-6m probability with phase background",
        yaxis_title="Probability",
        xaxis_title="Date",
        xaxis={
            "range": [default_start, default_end],
            "rangeslider": {"visible": True},
            "rangeselector": {
                "buttons": [
                    {"count": 5, "label": "5Y", "step": "year", "stepmode": "backward"},
                    {"count": 10, "label": "10Y", "step": "year", "stepmode": "backward"},
                    {"step": "all", "label": "All"},
                ]
            },
        },
        template="plotly_white",
        hovermode="x unified",
        height=520,
    )
    st.plotly_chart(timeline, use_container_width=True)
    st.caption(
        "Background colors show the model's current phase classification for each month. "
        "Dark red recession bars show official recession months. "
        f"The chart starts at the first month where recession-risk probabilities are available ({first_history_date.strftime('%Y-%m')}). "
        "It defaults to the last 10 years, and the bottom range slider lets you widen or narrow the window. "
        "The shaded red block marks the forward 6-month forecast window starting at the forecast origin."
    )

    phase_strip = _phase_strip_chart(plot_history)
    phase_strip.update_traces(line={"width": 28})
    phase_strip.update_layout(height=240)
    phase_strip.update_xaxes(rangeslider={"visible": True})
    st.plotly_chart(phase_strip, use_container_width=True)

    left, right = st.columns([1.1, 1.2])
    with left:
        clock = go.Figure()
        trail = plot_history.dropna(subset=["level_score", "momentum_score"]).copy()
        if not trail.empty:
            recent_trail = trail.tail(60)
            clock.add_trace(
                go.Scatter(
                    x=recent_trail["level_score"],
                    y=recent_trail["momentum_score"],
                    mode="lines+markers",
                    line={"color": "#bbbbbb", "width": 2},
                    marker={"size": 5, "color": "#bbbbbb"},
                    opacity=0.55,
                    name="Recent path",
                    hovertemplate="%{text}<extra></extra>",
                    text=[f"{row.date.strftime('%Y-%m')}<br>{row.current_phase}" for row in recent_trail.itertuples()],
                )
            )
        clock.add_hline(y=0, line_dash="dot", line_color="#999999")
        clock.add_vline(x=0, line_dash="dot", line_color="#999999")
        clock.add_trace(
            go.Scatter(
                x=[latest["level_score"]],
                y=[latest["momentum_score"]],
                mode="markers+text",
                text=[latest["current_phase"]],
                textposition="top center",
                marker={"size": 18, "color": PHASE_COLORS.get(latest["current_phase"], "#111111")},
                name="Latest point",
            )
        )
        max_level = max(0.6, float(trail["level_score"].abs().max()) if not trail.empty else 0.6)
        max_momentum = max(0.08, float(trail["momentum_score"].abs().max()) if not trail.empty else 0.08)
        clock.update_layout(
            title="Business-Cycle Clock",
            xaxis_title="Level score",
            yaxis_title="Momentum score",
            template="plotly_white",
            xaxis={"range": [-max_level * 1.1, max_level * 1.1]},
            yaxis={"range": [-max_momentum * 1.1, max_momentum * 1.1]},
        )
        clock.add_annotation(x=max_level * 0.75, y=max_momentum * 0.75, text="Expansion", showarrow=False, font={"color": PHASE_COLORS["Expansion"]})
        clock.add_annotation(x=max_level * 0.75, y=-max_momentum * 0.75, text="Slowdown", showarrow=False, font={"color": PHASE_COLORS["Slowdown"]})
        clock.add_annotation(x=-max_level * 0.75, y=-max_momentum * 0.75, text="Contraction", showarrow=False, font={"color": PHASE_COLORS["Contraction"]})
        clock.add_annotation(x=-max_level * 0.75, y=max_momentum * 0.75, text="Recovery", showarrow=False, font={"color": PHASE_COLORS["Recovery"]})
        st.plotly_chart(clock, use_container_width=True)

        phase_forecast = go.Figure()
        for phase_name, phase_frame in forecast_long.groupby("phase"):
            phase_forecast.add_trace(
                go.Scatter(
                    x=phase_frame["horizon_months"],
                    y=phase_frame["probability"],
                    mode="lines+markers",
                    name=phase_name,
                    line={"width": 3, "color": PHASE_COLORS.get(phase_name, "#111111")},
                )
            )
        phase_forecast.update_layout(
            title=f"Forward 6-Month Phase Probability Path from {pd.to_datetime(latest['date']).strftime('%Y-%m')}",
            xaxis_title="Forecast horizon (months)",
            yaxis_title="Probability",
            template="plotly_white",
            hovermode="x unified",
        )
        st.plotly_chart(phase_forecast, use_container_width=True)

    with right:
        drivers = pd.DataFrame(latest["driver_rows"])
        if not drivers.empty:
            drivers = drivers.reindex(drivers["importance"].abs().sort_values().index)
            driver_chart = go.Figure(
                go.Bar(
                    x=drivers["importance"],
                    y=drivers["feature"],
                    orientation="h",
                    marker_color="#4c78a8",
                    customdata=drivers[["source_name", "latest_value", "z_score"]],
                    hovertemplate=(
                        "%{y}<br>%{customdata[0]}"
                        "<br>Latest value: %{customdata[1]}"
                        "<br>Z-score: %{customdata[2]:.2f}"
                        "<br>Signed contribution: %{x:.3f}<extra></extra>"
                    ),
                )
            )
            driver_chart.add_vline(x=0, line_dash="dot", line_color="#999999")
            driver_chart.update_layout(
                title=f"Top Latest Recession-Risk Contributors ({pd.to_datetime(latest['date']).strftime('%Y-%m')})",
                xaxis_title="Signed contribution to current recession-risk score",
                yaxis_title="Feature",
                template="plotly_white",
                height=500,
            )
            st.plotly_chart(driver_chart, use_container_width=True)
            st.caption("Negative bars push the current recession-risk score down. Positive bars push it up.")
        st.subheader("Latest Indicator Drivers")
        st.dataframe(drivers[["feature", "source_name", "latest_value", "z_score", "importance"]], use_container_width=True)


if __name__ == "__main__":
    main()
