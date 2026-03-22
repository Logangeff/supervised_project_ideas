from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parent.parent
METRICS_DIR = ROOT / "outputs" / "metrics"
SUMMARIES_DIR = ROOT / "outputs" / "summaries"
PROCESSED_DIR = ROOT / "data" / "processed"
FIGURES_DIR = ROOT / "outputs" / "figures"

COLOR_MAP = {
    "NSS benchmark": "#111827",
    "Hull-White 1F": "#0f766e",
    "CIR": "#dc2626",
    "Vasicek": "#2563eb",
}


@st.cache_data
def load_json(name: str) -> dict:
    return json.loads((SUMMARIES_DIR / name).read_text(encoding="utf-8"))


@st.cache_data
def load_csv(name: str, parse_dates: list[str] | None = None) -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / name, parse_dates=parse_dates)


@st.cache_data
def load_metric_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(METRICS_DIR / name)


def format_pct(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def format_bps(value: float, digits: int = 2) -> str:
    return f"{value * 10000:.{digits}f} bps"


def format_price(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def build_overview_table(fit_metrics: pd.DataFrame, pricing_metrics: pd.DataFrame, swap_metrics: pd.DataFrame) -> pd.DataFrame:
    fit = fit_metrics.copy()
    pricing = pricing_metrics.groupby("model", as_index=False).agg(pricing_rmse=("pricing_rmse", "mean"))
    swaps = swap_metrics.groupby("model", as_index=False).agg(swap_rate_rmse=("swap_rate_rmse", "mean"))
    merged = fit.merge(pricing, on="model", how="left").merge(swaps, on="model", how="left")
    merged["zero_yield_rmse_bps"] = merged["zero_yield_rmse"] * 10000.0
    merged["pricing_rmse"] = merged["pricing_rmse"].fillna(0.0)
    merged["swap_rate_rmse_bps"] = merged["swap_rate_rmse"].fillna(0.0) * 10000.0
    return merged[
        [
            "model",
            "zero_yield_rmse_bps",
            "zero_yield_mae",
            "pricing_rmse",
            "swap_rate_rmse_bps",
            "forward_roughness",
            "discount_monotonic_share",
        ]
    ].sort_values("zero_yield_rmse_bps")


def render_metric_cards(fetch_summary: dict, bootstrap_summary: dict, observed_metrics: pd.DataFrame) -> None:
    fair_contest = observed_metrics[observed_metrics["model"].isin(["CIR", "Vasicek"])].sort_values("observed_yield_rmse_bps").reset_index(drop=True)
    best_equilibrium = fair_contest.iloc[0]
    cols = st.columns(4)
    cols[0].metric("Sample Range", f"{fetch_summary['start_date']} to {fetch_summary['end_date']}")
    cols[1].metric("Monthly Snapshots", str(bootstrap_summary["monthly_snapshot_count"]))
    cols[2].metric("Best Equilibrium Model", best_equilibrium["model"])
    cols[3].metric("Observed-Curve RMSE", format_bps(best_equilibrium["observed_yield_rmse_bps"] / 10000.0))
    st.caption(
        "Observed Treasury yields are the primary benchmark. NSS is the internal smoothing layer. Hull-White 1F is the exact-fit arbitrage-free anchor, and the fair structural contest is CIR versus Vasicek."
    )


def render_overview_tab(
    fetch_summary: dict,
    bootstrap_summary: dict,
    overview: pd.DataFrame,
    observed_metrics: pd.DataFrame,
    monthly_rates: pd.DataFrame,
    nss_curves: pd.DataFrame,
    model_curves: pd.DataFrame,
) -> None:
    st.subheader("Project Story")
    st.write(
        "This project starts from observed public U.S. Treasury yields. Those observed market quotes are the primary benchmark. We then bootstrap par, zero, discount, and forward curves, smooth the term structure with Nelson-Siegel-Svensson, and compare one-factor models. Hull-White 1F is included as the exact-fit anchor, while the meaningful structural contest is CIR versus Vasicek."
    )
    render_metric_cards(fetch_summary, bootstrap_summary, observed_metrics)
    st.subheader("Decision Snapshot")
    fair_contest = observed_metrics[observed_metrics["model"].isin(["CIR", "Vasicek"])].sort_values("observed_yield_rmse_bps").reset_index(drop=True)
    winner = fair_contest.iloc[0]
    runner_up = fair_contest.iloc[1]
    st.info(
        f"Against the observed Treasury curve, {winner['model']} currently edges {runner_up['model']} within the fair equilibrium-model contest. Hull-White remains the anchored lower bound, and NSS remains the internal smoothing reference rather than the primary benchmark."
    )
    available_dates = sorted(pd.to_datetime(monthly_rates["date"]).dt.date.unique())
    selected_date = st.selectbox("Observed-vs-model snapshot", options=available_dates, index=len(available_dates) - 1, key="overview_date")
    selected_row = monthly_rates[pd.to_datetime(monthly_rates["date"]).dt.date == selected_date].iloc[0]
    observed_points = pd.DataFrame(
        [
            {"maturity_years": 1.0, "label": "1Y", "observed_yield_pct": float(selected_row["DGS1"])},
            {"maturity_years": 2.0, "label": "2Y", "observed_yield_pct": float(selected_row["DGS2"])},
            {"maturity_years": 3.0, "label": "3Y", "observed_yield_pct": float(selected_row["DGS3"])},
            {"maturity_years": 5.0, "label": "5Y", "observed_yield_pct": float(selected_row["DGS5"])},
            {"maturity_years": 7.0, "label": "7Y", "observed_yield_pct": float(selected_row["DGS7"])},
            {"maturity_years": 10.0, "label": "10Y", "observed_yield_pct": float(selected_row["DGS10"])},
            {"maturity_years": 20.0, "label": "20Y", "observed_yield_pct": float(selected_row["DGS20"])},
            {"maturity_years": 30.0, "label": "30Y", "observed_yield_pct": float(selected_row["DGS30"])},
        ]
    )
    benchmark = nss_curves[pd.to_datetime(nss_curves["date"]).dt.date == selected_date].copy()
    benchmark["display_rate"] = benchmark["zero_yield"] * 100.0
    benchmark["model"] = "NSS benchmark"
    models = model_curves[pd.to_datetime(model_curves["date"]).dt.date == selected_date].copy()
    models["display_rate"] = models["zero_yield"] * 100.0
    plot_frame = pd.concat(
        [
            benchmark[["maturity_years", "display_rate", "model"]],
            models[["maturity_years", "display_rate", "model"]],
        ],
        ignore_index=True,
    )
    fig = px.line(
        plot_frame,
        x="maturity_years",
        y="display_rate",
        color="model",
        color_discrete_map=COLOR_MAP,
        title=f"Observed Treasury Curve vs Smoothed / Model Curves on {selected_date}",
    )
    fig.add_trace(
        go.Scatter(
            x=observed_points["maturity_years"],
            y=observed_points["observed_yield_pct"],
            mode="markers",
            marker=dict(color="#f59e0b", size=10, symbol="diamond"),
            name="Observed Treasury yields",
        )
    )
    fig.update_layout(xaxis_title="Maturity (years)", yaxis_title="Yield / zero rate (%)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Orange markers are the actual observed Treasury yields. The lines are the internal smoothed benchmark and the model-implied curves.")
    display = overview.copy()
    display["Zero-Yield RMSE"] = display["zero_yield_rmse_bps"].map(lambda value: f"{value:.2f} bps")
    display["Bond Pricing RMSE"] = display["pricing_rmse"].map(format_price)
    display["Swap RMSE"] = display["swap_rate_rmse_bps"].map(lambda value: f"{value:.2f} bps")
    display["Monotonic Discount Share"] = display["discount_monotonic_share"].map(lambda value: f"{value * 100:.1f}%")
    observed_display = observed_metrics.copy()
    observed_display["Observed Curve RMSE"] = observed_display["observed_yield_rmse_bps"].map(lambda value: f"{value:.2f} bps")
    observed_display["Observed Curve MAE"] = observed_display["observed_yield_mae_bps"].map(lambda value: f"{value:.2f} bps")
    observed_display = observed_display[["model", "Observed Curve RMSE", "Observed Curve MAE", "points"]]
    st.dataframe(observed_display.rename(columns={"model": "Model", "points": "Observed points"}), use_container_width=True, hide_index=True)
    st.caption("This table is the direct market benchmark: each model is compared to the observed Treasury curve at the quoted maturities.")
    st.dataframe(
        display[
            ["model", "Zero-Yield RMSE", "Bond Pricing RMSE", "Swap RMSE", "Monotonic Discount Share"]
        ].rename(columns={"model": "Model"}),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("This second table is the internal comparison layer: each model versus the NSS-smoothed zero-curve benchmark.")


def render_benchmark_tab(
    fit_metrics: pd.DataFrame,
    pricing_metrics: pd.DataFrame,
    swap_metrics: pd.DataFrame,
    observed_metrics: pd.DataFrame,
) -> None:
    st.subheader("Benchmark Logic")
    st.write(
        "Read the benchmark story in two layers. First, compare everything to the observed Treasury curve. Second, compare the model zero curves to the internal NSS smoothing layer."
    )

    observed_plot = observed_metrics.copy()
    observed_plot["Observed RMSE (bps)"] = observed_plot["observed_yield_rmse"] * 10000.0
    fig_observed = px.bar(
        observed_plot,
        x="model",
        y="Observed RMSE (bps)",
        color="model",
        color_discrete_map=COLOR_MAP,
        title="Observed Treasury Curve Fit Error",
        text_auto=".2f",
    )
    fig_observed.update_layout(showlegend=False, xaxis_title="", yaxis_title="RMSE (basis points)")
    st.plotly_chart(fig_observed, use_container_width=True)
    st.caption("This is the primary benchmark: how closely each curve representation lines up with the actual observed Treasury yields.")

    fit_plot = fit_metrics.copy()
    fit_plot["RMSE (bps)"] = fit_plot["zero_yield_rmse"] * 10000.0
    fig_fit = px.bar(
        fit_plot,
        x="model",
        y="RMSE (bps)",
        color="model",
        color_discrete_map=COLOR_MAP,
        title="Zero-Yield Fit Error vs Benchmark Curve",
        text_auto=".2f",
    )
    fig_fit.update_layout(showlegend=False, xaxis_title="", yaxis_title="RMSE (basis points)")
    st.plotly_chart(fig_fit, use_container_width=True)
    st.caption("This is the secondary internal benchmark: model zero curves versus the NSS-smoothed zero curve.")

    maturity = st.selectbox("Pricing maturity", options=[2.0, 5.0, 10.0, 30.0], index=2, key="pricing_maturity")
    pricing_slice = pricing_metrics[pricing_metrics["maturity_years"] == maturity].copy()
    pricing_slice["Pricing RMSE"] = pricing_slice["pricing_rmse"]
    fig_pricing = px.bar(
        pricing_slice,
        x="model",
        y="Pricing RMSE",
        color="model",
        color_discrete_map=COLOR_MAP,
        title=f"Coupon-Bond Pricing RMSE at {int(maturity)}Y",
        text_auto=".3f",
    )
    fig_pricing.update_layout(showlegend=False, xaxis_title="", yaxis_title="Price RMSE")
    st.plotly_chart(fig_pricing, use_container_width=True)
    st.caption("This panel shows how closely each model reprices the benchmark par-coupon bond at the selected maturity.")

    swap_slice = swap_metrics[swap_metrics["maturity_years"] == maturity].copy()
    swap_slice["Swap RMSE (bps)"] = swap_slice["swap_rate_rmse"] * 10000.0
    fig_swap = px.bar(
        swap_slice,
        x="model",
        y="Swap RMSE (bps)",
        color="model",
        color_discrete_map=COLOR_MAP,
        title=f"Par Swap Rate RMSE at {int(maturity)}Y",
        text_auto=".2f",
    )
    fig_swap.update_layout(showlegend=False, xaxis_title="", yaxis_title="RMSE (basis points)")
    st.plotly_chart(fig_swap, use_container_width=True)
    st.caption("This is the pricing-usefulness view: how much each model deviates from the benchmark par swap rate.")

    table = build_overview_table(fit_metrics, pricing_metrics, swap_metrics)
    display = table.copy()
    display["zero_yield_rmse_bps"] = display["zero_yield_rmse_bps"].map(lambda value: f"{value:.2f}")
    display["zero_yield_mae"] = display["zero_yield_mae"].map(lambda value: f"{value * 10000:.2f}")
    display["pricing_rmse"] = display["pricing_rmse"].map(lambda value: f"{value:.3f}")
    display["swap_rate_rmse_bps"] = display["swap_rate_rmse_bps"].map(lambda value: f"{value:.2f}")
    display["forward_roughness"] = display["forward_roughness"].map(lambda value: f"{value:.2e}")
    display["discount_monotonic_share"] = display["discount_monotonic_share"].map(lambda value: f"{value * 100:.1f}%")
    st.dataframe(
        display.rename(
            columns={
                "model": "Model",
                "zero_yield_rmse_bps": "Zero RMSE (bps)",
                "zero_yield_mae": "Zero MAE (bps)",
                "pricing_rmse": "Avg Bond Pricing RMSE",
                "swap_rate_rmse_bps": "Avg Swap RMSE (bps)",
                "forward_roughness": "Forward Roughness",
                "discount_monotonic_share": "Discount Monotonic Share",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


def render_curve_explorer(nss_curves: pd.DataFrame, model_curves: pd.DataFrame, monthly_rates: pd.DataFrame) -> None:
    st.subheader("Curve Explorer")
    available_dates = sorted(pd.to_datetime(nss_curves["date"]).dt.date.unique())
    default_date = available_dates[-1]
    selected_date = st.selectbox("Snapshot date", options=available_dates, index=len(available_dates) - 1)
    metric = st.selectbox(
        "Curve metric",
        options=["zero_yield", "discount_factor", "forward_rate"],
        format_func=lambda value: {
            "zero_yield": "Zero yield",
            "discount_factor": "Discount factor",
            "forward_rate": "Forward rate",
        }[value],
    )
    benchmark = nss_curves[pd.to_datetime(nss_curves["date"]).dt.date == selected_date].copy()
    benchmark["model"] = "NSS benchmark"
    models = model_curves[pd.to_datetime(model_curves["date"]).dt.date == selected_date].copy()
    plot_frame = pd.concat(
        [
            benchmark[["maturity_years", metric, "model"]],
            models[["maturity_years", metric, "model"]],
        ],
        ignore_index=True,
    )
    if metric in {"zero_yield", "forward_rate"}:
        plot_frame["display_value"] = plot_frame[metric] * 100.0
        y_title = "Rate (%)"
    else:
        plot_frame["display_value"] = plot_frame[metric]
        y_title = "Discount factor"
    fig = px.line(
        plot_frame,
        x="maturity_years",
        y="display_value",
        color="model",
        color_discrete_map=COLOR_MAP,
        markers=False,
        title=f"{selected_date} Curve Comparison",
    )
    observed_row = monthly_rates[pd.to_datetime(monthly_rates["date"]).dt.date == selected_date].iloc[0]
    observed_points = pd.DataFrame(
        [
            {"maturity_years": 1.0, "display_value": float(observed_row["DGS1"]), "model": "Observed Treasury yields"},
            {"maturity_years": 2.0, "display_value": float(observed_row["DGS2"]), "model": "Observed Treasury yields"},
            {"maturity_years": 3.0, "display_value": float(observed_row["DGS3"]), "model": "Observed Treasury yields"},
            {"maturity_years": 5.0, "display_value": float(observed_row["DGS5"]), "model": "Observed Treasury yields"},
            {"maturity_years": 7.0, "display_value": float(observed_row["DGS7"]), "model": "Observed Treasury yields"},
            {"maturity_years": 10.0, "display_value": float(observed_row["DGS10"]), "model": "Observed Treasury yields"},
            {"maturity_years": 20.0, "display_value": float(observed_row["DGS20"]), "model": "Observed Treasury yields"},
            {"maturity_years": 30.0, "display_value": float(observed_row["DGS30"]), "model": "Observed Treasury yields"},
        ]
    )
    fig.add_trace(
        go.Scatter(
            x=observed_points["maturity_years"],
            y=observed_points["display_value"],
            mode="markers",
            marker=dict(color="#f59e0b", size=9, symbol="diamond"),
            name="Observed Treasury yields",
        )
    )
    fig.update_layout(xaxis_title="Maturity (years)", yaxis_title=y_title)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Observed market yields are shown as orange markers so you can compare them directly with the smoothed and model-implied curves.")

    eval_grid = [0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
    detail = plot_frame[plot_frame["maturity_years"].isin(eval_grid)].copy()
    pivot = detail.pivot(index="maturity_years", columns="model", values="display_value").reset_index()
    st.dataframe(pivot, use_container_width=True, hide_index=True)


def render_pricing_tab(pricing_details: pd.DataFrame, swap_details: pd.DataFrame) -> None:
    st.subheader("Pricing Usefulness")
    maturity = st.selectbox("Detailed maturity", options=[2.0, 5.0, 10.0, 30.0], index=1, key="detail_maturity")
    metric = st.radio("Series", options=["Bond pricing error", "Par swap rate"], horizontal=True)

    if metric == "Bond pricing error":
        detail = pricing_details[pricing_details["maturity_years"] == maturity].copy()
        detail["abs_pricing_error"] = detail["pricing_error"].abs()
        fig = px.line(
            detail,
            x="date",
            y="abs_pricing_error",
            color="model",
            color_discrete_map=COLOR_MAP,
            title=f"Absolute Bond Pricing Error Over Time at {int(maturity)}Y",
        )
        fig.update_layout(xaxis_title="", yaxis_title="Absolute price error")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("This shows how much each model deviates from the benchmark par-coupon bond price through time.")
    else:
        detail = swap_details[swap_details["maturity_years"] == maturity].copy()
        detail["swap_rate_error_bps"] = detail["swap_rate_error"] * 10000.0
        fig = px.line(
            detail,
            x="date",
            y="swap_rate_error_bps",
            color="model",
            color_discrete_map=COLOR_MAP,
            title=f"Par Swap Rate Error Over Time at {int(maturity)}Y",
        )
        fig.update_layout(xaxis_title="", yaxis_title="Swap rate error (bps)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("A value near zero means the model projects a par swap rate very close to the benchmark curve.")

    latest_date = pd.to_datetime(detail["date"]).max()
    latest = detail[pd.to_datetime(detail["date"]) == latest_date].copy()
    st.dataframe(latest, use_container_width=True, hide_index=True)


def render_data_quality_tab(fetch_summary: dict, bootstrap_summary: dict, fit_summary: dict, monthly_rates: pd.DataFrame) -> None:
    st.subheader("Data Quality and Validation")
    cols = st.columns(4)
    cols[0].metric("Treasury Series", str(fetch_summary["series_count"]))
    cols[1].metric("Raw Daily Rows", str(fetch_summary["rows"]))
    cols[2].metric("Monthly Snapshots", str(bootstrap_summary["monthly_snapshot_count"]))
    cols[3].metric("Toy Bootstrap Check", "Passed" if abs(bootstrap_summary["toy_bootstrap_check"]["bootstrap_equation_residual_2y"]) < 1e-10 else "Review")

    validation = pd.DataFrame(
        [
            {"Check": "Bootstrap discount factors stay positive", "Result": "Pass" if bootstrap_summary["all_discount_positive"] else "Fail"},
            {"Check": "Bootstrap discount factors are monotone", "Result": "Pass" if bootstrap_summary["all_discount_monotonic"] else "Fail"},
            {"Check": "Hull-White exact-fit benchmark reproduced", "Result": "Pass" if fit_summary["hull_white_max_abs_error"] == 0.0 else "Review"},
            {"Check": "Dynamic Hull-White parameters estimated from data", "Result": "Pass" if not fit_summary["hull_white_dynamic_estimate"]["used_default"] else "Defaulted"},
        ]
    )
    st.dataframe(validation, use_container_width=True, hide_index=True)

    missing = pd.DataFrame(
        {
            "series_id": list(fetch_summary["missing_share_by_series"].keys()),
            "missing_share": list(fetch_summary["missing_share_by_series"].values()),
        }
    )
    missing["missing_pct"] = missing["missing_share"] * 100.0
    fig_missing = px.bar(
        missing,
        x="series_id",
        y="missing_pct",
        title="Missing Share by Treasury Series",
        color_discrete_sequence=["#0f4c81"],
    )
    fig_missing.update_layout(xaxis_title="", yaxis_title="Missing share (%)", showlegend=False)
    st.plotly_chart(fig_missing, use_container_width=True)
    st.caption("The monthly snapshot builder drops all-NaN holiday month-end rows and keeps the last valid in-month observation.")

    recent_rates = monthly_rates.tail(24).copy()
    long_rates = recent_rates.melt(id_vars="date", value_vars=["DGS1", "DGS2", "DGS5", "DGS10", "DGS30"], var_name="series_id", value_name="yield_pct")
    fig_rates = px.line(
        long_rates,
        x="date",
        y="yield_pct",
        color="series_id",
        title="Recent Monthly Treasury Levels Used in the Project",
    )
    fig_rates.update_layout(xaxis_title="", yaxis_title="Yield (%)")
    st.plotly_chart(fig_rates, use_container_width=True)


def render_appendix_tab() -> None:
    st.subheader("Appendix")
    st.write("Saved figures from the pipeline are embedded below. The earlier high-scoring course submission kept under `docs/TP1 - Submission` is retained as reference material for later phases, but it is not mixed into the current benchmark results.")
    image_paths = sorted(FIGURES_DIR.glob("*.png"))
    for image_path in image_paths:
        st.image(str(image_path), caption=image_path.name, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Fixed-Income Term Structure Dashboard", layout="wide")
    st.title("Fixed-Income Term Structure Dashboard")
    st.caption("Public U.S. Treasury term-structure construction, NSS smoothing, and one-factor benchmark comparison.")

    fetch_summary = load_json("fetch_public_rates_summary.json")
    bootstrap_summary = load_json("build_bootstrap_curves_summary.json")
    fit_summary = load_json("fit_short_rate_models_summary.json")

    fit_metrics = load_metric_csv("model_fit_metrics.csv")
    pricing_metrics = load_metric_csv("model_pricing_metrics.csv")
    swap_metrics = load_metric_csv("swap_rate_comparison.csv")
    observed_metrics = load_metric_csv("observed_curve_fit_metrics.csv")
    monthly_rates = load_csv("monthly_rates.csv", parse_dates=["date"])
    nss_curves = load_csv("nss_curves.csv", parse_dates=["date"])
    model_curves = load_csv("model_curves.csv", parse_dates=["date"])
    pricing_details = load_csv("pricing_details.csv", parse_dates=["date"])
    swap_details = load_csv("swap_rate_details.csv", parse_dates=["date"])

    overview = build_overview_table(fit_metrics, pricing_metrics, swap_metrics)

    tabs = st.tabs(["Overview", "Benchmark Comparison", "Curve Explorer", "Pricing", "Data Quality", "Appendix"])
    with tabs[0]:
        render_overview_tab(fetch_summary, bootstrap_summary, overview, observed_metrics.assign(observed_yield_rmse_bps=lambda df: df["observed_yield_rmse"] * 10000.0, observed_yield_mae_bps=lambda df: df["observed_yield_mae"] * 10000.0), monthly_rates, nss_curves, model_curves)
    with tabs[1]:
        render_benchmark_tab(fit_metrics, pricing_metrics, swap_metrics, observed_metrics)
    with tabs[2]:
        render_curve_explorer(nss_curves, model_curves, monthly_rates)
    with tabs[3]:
        render_pricing_tab(pricing_details, swap_details)
    with tabs[4]:
        render_data_quality_tab(fetch_summary, bootstrap_summary, fit_summary, monthly_rates)
    with tabs[5]:
        render_appendix_tab()


if __name__ == "__main__":
    main()
