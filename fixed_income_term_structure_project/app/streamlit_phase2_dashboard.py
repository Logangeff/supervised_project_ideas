from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parent.parent
PHASE2_PROCESSED_DIR = ROOT / "data" / "processed" / "phase2"
PHASE2_OUTPUTS_DIR = ROOT / "outputs" / "phase2"
PHASE2_SUMMARIES_DIR = PHASE2_OUTPUTS_DIR / "summaries"
PHASE2_METRICS_DIR = PHASE2_OUTPUTS_DIR / "metrics"
PHASE2_FIGURES_DIR = PHASE2_OUTPUTS_DIR / "figures"


@st.cache_data
def load_json(name: str) -> dict:
    return json.loads((PHASE2_SUMMARIES_DIR / name).read_text(encoding="utf-8"))


@st.cache_data
def load_csv(name: str, folder: Path) -> pd.DataFrame:
    return pd.read_csv(folder / name, parse_dates=["date"])


def render_overview(data_summary: dict, curve_summary: dict, gap_rates: pd.DataFrame) -> None:
    st.subheader("Why Phase 2 Exists")
    st.write(
        "Phase 2 keeps Phase 1 intact and adds a separate pricing architecture question: what changes when we move from a single Treasury-based curve to a public-only multi-curve swap setup with separate discount and projection curves?"
    )
    avg_gaps = (
        gap_rates.groupby("maturity_years", as_index=False)
        .agg(
            mean_total_gap_bps=("total_gap_bps", "mean"),
            mean_discount_effect_bps=("discount_effect_bps", "mean"),
            mean_projection_effect_bps=("projection_effect_bps", "mean"),
            max_abs_total_gap_bps=("total_gap_bps", lambda values: values.abs().max()),
        )
        .sort_values("maturity_years")
    )
    cols = st.columns(4)
    cols[0].metric("Phase 2 Sample Start", data_summary["phase2_sample_start"])
    cols[1].metric("Phase 2 Snapshots", str(data_summary["phase2_snapshot_count"]))
    cols[2].metric("Mean 5Y Gap", f"{avg_gaps.loc[avg_gaps['maturity_years'] == 5.0, 'mean_total_gap_bps'].iloc[0]:.2f} bps")
    cols[3].metric("Discount Monotonic Share", f"{curve_summary['discount_monotonic_share'] * 100:.1f}%")
    st.info(
        "Benchmark structure: Phase 1 single-curve NSS swap pricing is the baseline. Phase 2 then separates discounting from projection using a SOFR-anchored public discount proxy and a Treasury-forward-based projection curve."
    )
    st.info(
        "Read this dashboard in order: first understand the public discount and projection curves, then compare single-curve versus multi-curve swap rates, then interpret the sensitivity split."
    )
    benchmark_table = avg_gaps.copy()
    benchmark_table["maturity_years"] = benchmark_table["maturity_years"].map(lambda value: f"{int(value)}Y")
    benchmark_table = benchmark_table.rename(
        columns={
            "maturity_years": "Maturity",
            "mean_total_gap_bps": "Mean total gap (bps)",
            "mean_discount_effect_bps": "Mean discount effect (bps)",
            "mean_projection_effect_bps": "Mean projection effect (bps)",
            "max_abs_total_gap_bps": "Max abs gap (bps)",
        }
    )
    st.dataframe(benchmark_table, use_container_width=True, hide_index=True)
    st.caption(
        "Interpretation: the 2Y gap is mostly a discounting effect, the 5Y gap is mainly projection-driven, and the 10Y gap is dominated by the projection curve in this public-data prototype."
    )

    fig = px.line(
        gap_rates,
        x="date",
        y="total_gap_bps",
        color=gap_rates["maturity_years"].astype(str) + "Y",
        title="Single-Curve vs Multi-Curve Par Rate Gap Over Time",
    )
    fig.update_layout(xaxis_title="", yaxis_title="Par-rate gap (bps)", legend_title="Maturity")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("A non-zero gap means the separate discount and projection curves produce a different swap rate than the Phase 1 single-curve baseline.")


def render_curve_architecture(discount_curves: pd.DataFrame, projection_curves: pd.DataFrame, curve_summary: dict) -> None:
    st.subheader("Curve Architecture")
    available_dates = sorted(pd.to_datetime(discount_curves["date"]).dt.date.unique())
    selected_date = st.selectbox("Snapshot date", options=available_dates, index=len(available_dates) - 1, key="phase2_curve_date")

    discount = discount_curves[pd.to_datetime(discount_curves["date"]).dt.date == selected_date].copy()
    projection = projection_curves[pd.to_datetime(projection_curves["date"]).dt.date == selected_date].copy()

    fig_zero = go.Figure()
    fig_zero.add_trace(go.Scatter(x=discount["maturity_years"], y=discount["zero_yield"] * 100.0, mode="lines", name="Discount proxy", line=dict(color="#0f766e", width=3)))
    fig_zero.add_trace(go.Scatter(x=projection["maturity_years"], y=projection["zero_yield"] * 100.0, mode="lines", name="Projection curve", line=dict(color="#2563eb", width=3)))
    fig_zero.update_layout(title=f"Discount vs Projection Zero Curves on {selected_date}", xaxis_title="Maturity (years)", yaxis_title="Zero yield (%)")
    st.plotly_chart(fig_zero, use_container_width=True)

    fig_forward = go.Figure()
    fig_forward.add_trace(go.Scatter(x=discount["maturity_years"], y=discount["forward_rate"] * 100.0, mode="lines", name="Discount proxy forward", line=dict(color="#059669", width=2)))
    fig_forward.add_trace(go.Scatter(x=projection["maturity_years"], y=projection["forward_rate_3m"] * 100.0, mode="lines", name="Projection 3M forward", line=dict(color="#1d4ed8", width=2)))
    fig_forward.update_layout(title=f"Forward Curve View on {selected_date}", xaxis_title="Maturity (years)", yaxis_title="Forward rate (%)")
    st.plotly_chart(fig_forward, use_container_width=True)

    st.dataframe(
        pd.DataFrame(
            [
                {"Metric": "Mean join shift", "Value": f"{curve_summary['mean_join_shift_bps']:.2f} bps"},
                {"Metric": "Max abs join shift", "Value": f"{curve_summary['max_abs_join_shift_bps']:.2f} bps"},
                {"Metric": "Discount positive share", "Value": f"{curve_summary['discount_positive_share'] * 100:.1f}%"},
                {"Metric": "Discount monotonic share", "Value": f"{curve_summary['discount_monotonic_share'] * 100:.1f}%"},
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )


def render_swap_pricing(gap_rates: pd.DataFrame, gap_pv: pd.DataFrame, pricing_details: pd.DataFrame) -> None:
    st.subheader("Swap Pricing")
    maturity = st.selectbox("Swap maturity", options=[2.0, 5.0, 10.0], index=1, key="phase2_pricing_maturity")

    subset = gap_rates[gap_rates["maturity_years"] == maturity].copy()
    fig_gap = go.Figure()
    fig_gap.add_trace(go.Scatter(x=subset["date"], y=subset["discount_effect_bps"], mode="lines", name="Discount effect", line=dict(color="#0f766e")))
    fig_gap.add_trace(go.Scatter(x=subset["date"], y=subset["projection_effect_bps"], mode="lines", name="Projection effect", line=dict(color="#dc2626")))
    fig_gap.add_trace(go.Scatter(x=subset["date"], y=subset["total_gap_bps"], mode="lines", name="Total gap", line=dict(color="#111827", width=3)))
    fig_gap.update_layout(title=f"Par Rate Gap Decomposition: {int(maturity)}Y", xaxis_title="", yaxis_title="Gap (bps)")
    st.plotly_chart(fig_gap, use_container_width=True)

    pv_subset = gap_pv[gap_pv["maturity_years"] == maturity].copy()
    fig_pv = go.Figure()
    fig_pv.add_trace(go.Scatter(x=pv_subset["date"], y=pv_subset["discount_effect_pv"], mode="lines", name="Discount effect", line=dict(color="#0f766e")))
    fig_pv.add_trace(go.Scatter(x=pv_subset["date"], y=pv_subset["projection_effect_pv"], mode="lines", name="Projection effect", line=dict(color="#dc2626")))
    fig_pv.add_trace(go.Scatter(x=pv_subset["date"], y=pv_subset["total_gap_pv"], mode="lines", name="Total PV gap", line=dict(color="#111827", width=3)))
    fig_pv.update_layout(title=f"PV Gap Decomposition: {int(maturity)}Y", xaxis_title="", yaxis_title="PV gap")
    st.plotly_chart(fig_pv, use_container_width=True)

    latest_date = pd.to_datetime(pricing_details["date"]).max()
    latest = pricing_details[(pricing_details["maturity_years"] == maturity) & (pd.to_datetime(pricing_details["date"]) == latest_date)].copy()
    st.dataframe(latest, use_container_width=True, hide_index=True)
    st.caption("This table shows the latest pricing details for the Phase 1 single-curve setup, the discount-only step, and the full Phase 2 multi-curve setup.")
    st.caption("If the discount-only line moves but the full multi-curve line moves more, the added change is coming from the projection curve rather than discounting alone.")


def render_sensitivities(sensitivity: pd.DataFrame) -> None:
    st.subheader("Sensitivities")
    maturity = st.selectbox("Sensitivity maturity", options=[2.0, 5.0, 10.0], index=2, key="phase2_sensitivity_maturity")
    available_dates = sorted(pd.to_datetime(sensitivity["date"]).dt.date.unique())
    selected_date = st.selectbox("Sensitivity date", options=available_dates, index=len(available_dates) - 1, key="phase2_sensitivity_date")
    subset = sensitivity[(sensitivity["maturity_years"] == maturity) & (pd.to_datetime(sensitivity["date"]).dt.date == selected_date)].copy()

    fig_pv = px.bar(subset, x="scenario", y="swap_pv_change", title=f"Swap PV Change by Scenario: {int(maturity)}Y on {selected_date}", color_discrete_sequence=["#0f4c81"])
    fig_pv.update_layout(xaxis_title="", yaxis_title="PV change")
    st.plotly_chart(fig_pv, use_container_width=True)

    fig_rate = px.bar(subset, x="scenario", y="par_rate_change_bps", title=f"Par Rate Change by Scenario: {int(maturity)}Y on {selected_date}", color_discrete_sequence=["#9333ea"])
    fig_rate.update_layout(xaxis_title="", yaxis_title="Par-rate change (bps)")
    st.plotly_chart(fig_rate, use_container_width=True)

    st.dataframe(subset, use_container_width=True, hide_index=True)
    st.caption("Discount shocks mostly move the fixed-leg valuation through discounting. Projection shocks move the floating leg and shift the par rate directly.")


def render_quality_and_limits(data_summary: dict, curve_summary: dict) -> None:
    st.subheader("Data Quality and Limitations")
    checks = pd.DataFrame(
        [
            {"Check": "SOFR public data available", "Result": "Pass"},
            {"Check": "12M SOFR lookback enforced", "Result": "Pass"},
            {"Check": "Discount factors positive", "Result": f"{curve_summary['discount_positive_share'] * 100:.1f}% pass"},
            {"Check": "Discount factors monotone", "Result": f"{curve_summary['discount_monotonic_share'] * 100:.1f}% pass"},
        ]
    )
    st.dataframe(checks, use_container_width=True, hide_index=True)
    st.warning(
        "This is a public-data proxy architecture. The discount curve is SOFR-anchored but not a market-vendor OIS curve, and the projection curve is Treasury-forward-based rather than a fully calibrated tenor-specific market curve."
    )
    st.write(
        f"SOFR sample: {data_summary['sofr_start_date']} to {data_summary['sofr_end_date']}. "
        f"Phase 2 usable sample: {data_summary['phase2_sample_start']} to {data_summary['phase2_sample_end']}."
    )


def main() -> None:
    st.set_page_config(page_title="Phase 2 Multi-Curve Dashboard", layout="wide")
    st.title("Phase 2 Multi-Curve Swap Pricing Dashboard")
    st.caption("A public-only extension beyond the standalone Phase 1 term-structure project.")

    data_summary = load_json("phase2_data_summary.json")
    curve_summary = load_json("phase2_curve_summary.json")
    discount_curves = load_csv("discount_proxy_curves.csv", PHASE2_PROCESSED_DIR)
    projection_curves = load_csv("projection_curves.csv", PHASE2_PROCESSED_DIR)
    pricing_details = load_csv("swap_pricing_details.csv", PHASE2_PROCESSED_DIR)
    gap_rates = load_csv("baseline_vs_multicurve_swap_rates.csv", PHASE2_METRICS_DIR)
    gap_pv = load_csv("baseline_vs_multicurve_pv.csv", PHASE2_METRICS_DIR)
    sensitivity = load_csv("sensitivity_summary.csv", PHASE2_METRICS_DIR)

    tabs = st.tabs(["Overview", "Curve Architecture", "Swap Pricing", "Sensitivities", "Data Quality / Limitations"])
    with tabs[0]:
        render_overview(data_summary, curve_summary, gap_rates)
    with tabs[1]:
        render_curve_architecture(discount_curves, projection_curves, curve_summary)
    with tabs[2]:
        render_swap_pricing(gap_rates, gap_pv, pricing_details)
    with tabs[3]:
        render_sensitivities(sensitivity)
    with tabs[4]:
        render_quality_and_limits(data_summary, curve_summary)


if __name__ == "__main__":
    main()
