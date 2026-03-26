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
    "Fed published benchmark": "#7c3aed",
    "NSS benchmark": "#111827",
    "Hull-White 1F": "#0f766e",
    "CIR": "#dc2626",
    "Vasicek": "#2563eb",
}

THEMES = {
    "Neo Green": {
        "bg": "#050505",
        "surface": "#111111",
        "surface_alt": "#181818",
        "text": "#f7f7f7",
        "muted": "#b8b8b8",
        "accent": "#8dff2f",
        "accent_soft": "rgba(141,255,47,0.14)",
        "border": "#222222",
        "bar": "#000000",
        "plot_bg": "#111111",
        "paper_bg": "#050505",
    },
    "Light": {
        "bg": "#f5f6f1",
        "surface": "#ffffff",
        "surface_alt": "#edf1ea",
        "text": "#101418",
        "muted": "#55606d",
        "accent": "#2cbf4f",
        "accent_soft": "rgba(44,191,79,0.14)",
        "border": "#d8ddd5",
        "bar": "#111111",
        "plot_bg": "#ffffff",
        "paper_bg": "#f5f6f1",
    },
    "Classic Dark": {
        "bg": "#0f1720",
        "surface": "#16212d",
        "surface_alt": "#1c2a38",
        "text": "#ecf1f7",
        "muted": "#b0becd",
        "accent": "#33d17a",
        "accent_soft": "rgba(51,209,122,0.14)",
        "border": "#263545",
        "bar": "#000000",
        "plot_bg": "#16212d",
        "paper_bg": "#0f1720",
    },
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


def format_bps_from_decimal(value: float) -> str:
    return f"{value * 10000:.2f} bps"


def format_bps(value: float) -> str:
    return f"{value:.2f} bps"


def inject_css(theme_name: str) -> dict:
    theme = THEMES[theme_name]
    st.markdown(
        f"""
        <style>
        :root {{
            --app-bg: {theme["bg"]};
            --card-bg: {theme["surface"]};
            --card-alt: {theme["surface_alt"]};
            --text-main: {theme["text"]};
            --text-muted: {theme["muted"]};
            --accent: {theme["accent"]};
            --accent-soft: {theme["accent_soft"]};
            --border: {theme["border"]};
            --bar: {theme["bar"]};
        }}
        .stApp {{
            background: var(--app-bg);
            color: var(--text-main);
        }}
        [data-testid="stHeader"] {{
            background: transparent;
        }}
        [data-testid="stToolbar"] {{
            right: 1rem;
        }}
        .block-container {{
            padding-top: 1.2rem;
            padding-bottom: 4rem;
            max-width: 1200px;
        }}
        h1, h2, h3, h4, h5, h6, p, li, label, div, span {{
            color: var(--text-main);
        }}
        .muted {{
            color: var(--text-muted);
        }}
        .hero {{
            background: linear-gradient(135deg, var(--card-bg) 0%, var(--card-alt) 100%);
            border: 1px solid var(--border);
            border-radius: 24px;
            padding: 2rem 2rem 1.5rem 2rem;
            margin-bottom: 1.2rem;
            box-shadow: 0 24px 80px rgba(0,0,0,0.18);
        }}
        .eyebrow {{
            display: inline-block;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--accent);
            margin-bottom: 0.75rem;
        }}
        .hero h1 {{
            font-size: 3rem;
            line-height: 1.03;
            margin: 0 0 0.75rem 0;
        }}
        .hero p {{
            max-width: 900px;
            color: var(--text-muted);
            font-size: 1.05rem;
        }}
        .cta-row {{
            display: flex;
            gap: 0.75rem;
            flex-wrap: wrap;
            margin-top: 1rem;
        }}
        .ui-btn {{
            display: inline-block;
            padding: 0.78rem 1rem;
            border-radius: 999px;
            font-weight: 700;
            text-decoration: none;
            border: 1px solid var(--accent);
            color: #081005 !important;
            background: var(--accent);
            box-shadow: 0 0 24px rgba(141,255,47,0.18);
        }}
        .ui-btn.secondary {{
            color: var(--text-main) !important;
            background: transparent;
            border-color: var(--border);
            box-shadow: none;
        }}
        .section-bar {{
            height: 16px;
            border-radius: 999px;
            background: var(--bar);
            margin: 2rem 0 1.25rem 0;
        }}
        .section-copy {{
            margin-bottom: 0.8rem;
            color: var(--text-muted);
        }}
        .card-note {{
            padding: 1rem 1.15rem;
            border-radius: 18px;
            border: 1px solid var(--border);
            background: var(--accent-soft);
            margin-bottom: 1rem;
        }}
        [data-testid="stMetric"] {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 0.75rem 0.9rem;
        }}
        [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {{
            color: var(--text-main);
        }}
        .dataframe, [data-testid="stDataFrame"] {{
            border-radius: 14px;
            overflow: hidden;
        }}
        .stTabs [data-baseweb="tab-list"] {{
            gap: 0.25rem;
        }}
        .stRadio > div {{
            gap: 0.6rem;
        }}
        .stButton > button, .stDownloadButton > button {{
            background: var(--accent);
            color: #081005;
            border: 1px solid var(--accent);
            border-radius: 999px;
            font-weight: 700;
        }}
        .stSelectbox label, .stRadio label {{
            color: var(--text-main) !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    return theme


def apply_plot_theme(fig: go.Figure, theme: dict) -> go.Figure:
    fig.update_layout(
        paper_bgcolor=theme["paper_bg"],
        plot_bgcolor=theme["plot_bg"],
        font_color=theme["text"],
        legend_font_color=theme["text"],
        xaxis=dict(gridcolor=theme["border"], zerolinecolor=theme["border"]),
        yaxis=dict(gridcolor=theme["border"], zerolinecolor=theme["border"]),
    )
    return fig


def section_header(title: str, subtitle: str | None = None) -> None:
    st.markdown('<div class="section-bar"></div>', unsafe_allow_html=True)
    st.header(title)
    if subtitle:
        st.markdown(f'<p class="section-copy">{subtitle}</p>', unsafe_allow_html=True)


def curve_snapshot_figure(selected_date, monthly_rates, fed_curves, nss_curves, model_curves, theme: dict) -> go.Figure:
    selected_row = monthly_rates[pd.to_datetime(monthly_rates["date"]).dt.date == selected_date].iloc[0]
    observed_points = pd.DataFrame(
        [
            {"maturity_years": 1.0, "observed_yield_pct": float(selected_row["DGS1"])},
            {"maturity_years": 2.0, "observed_yield_pct": float(selected_row["DGS2"])},
            {"maturity_years": 3.0, "observed_yield_pct": float(selected_row["DGS3"])},
            {"maturity_years": 5.0, "observed_yield_pct": float(selected_row["DGS5"])},
            {"maturity_years": 7.0, "observed_yield_pct": float(selected_row["DGS7"])},
            {"maturity_years": 10.0, "observed_yield_pct": float(selected_row["DGS10"])},
            {"maturity_years": 20.0, "observed_yield_pct": float(selected_row["DGS20"])},
            {"maturity_years": 30.0, "observed_yield_pct": float(selected_row["DGS30"])},
        ]
    )
    fed_curve = fed_curves[pd.to_datetime(fed_curves["date"]).dt.date == selected_date].copy()
    fed_curve["display_rate"] = fed_curve["zero_yield"] * 100.0
    fed_curve["model"] = "Fed published benchmark"
    nss_curve = nss_curves[pd.to_datetime(nss_curves["date"]).dt.date == selected_date].copy()
    nss_curve["display_rate"] = nss_curve["zero_yield"] * 100.0
    nss_curve["model"] = "NSS benchmark"
    models = model_curves[pd.to_datetime(model_curves["date"]).dt.date == selected_date].copy()
    models["display_rate"] = models["zero_yield"] * 100.0
    plot_frame = pd.concat(
        [fed_curve[["maturity_years", "display_rate", "model"]], nss_curve[["maturity_years", "display_rate", "model"]], models[["maturity_years", "display_rate", "model"]]],
        ignore_index=True,
    )
    fig = px.line(
        plot_frame,
        x="maturity_years",
        y="display_rate",
        color="model",
        color_discrete_map=COLOR_MAP,
        title=f"Observed, Fed, NSS, and Model Curves on {selected_date}",
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
    return apply_plot_theme(fig, theme)


def main() -> None:
    st.set_page_config(page_title="Fixed-Income Term Structure One-Page", layout="wide")

    fetch_summary = load_json("fetch_public_rates_summary.json")
    fed_fetch_summary = load_json("fetch_fed_benchmark_summary.json")
    bootstrap_summary = load_json("build_bootstrap_curves_summary.json")
    fit_summary = load_json("fit_short_rate_models_summary.json")
    fed_summary = load_json("build_fed_benchmark_summary.json")

    observed_metrics = load_metric_csv("observed_curve_fit_metrics.csv")
    fed_metrics = load_metric_csv("fed_curve_fit_metrics.csv")
    fit_metrics = load_metric_csv("model_fit_metrics.csv")
    pricing_metrics = load_metric_csv("model_pricing_metrics.csv")
    swap_metrics = load_metric_csv("swap_rate_comparison.csv")
    monthly_rates = load_csv("monthly_rates.csv", parse_dates=["date"])
    fed_curves = load_csv("fed_benchmark_curves.csv", parse_dates=["date"])
    nss_curves = load_csv("nss_curves.csv", parse_dates=["date"])
    model_curves = load_csv("model_curves.csv", parse_dates=["date"])
    pricing_details = load_csv("pricing_details.csv", parse_dates=["date"])
    swap_details = load_csv("swap_rate_details.csv", parse_dates=["date"])

    theme_name = st.radio("UI Mode", options=list(THEMES.keys()), horizontal=True, index=0, label_visibility="collapsed")
    theme = inject_css(theme_name)

    fair_fed = fed_metrics[fed_metrics["model"].isin(["CIR", "Vasicek"])].sort_values("fed_yield_rmse").reset_index(drop=True)
    fair_observed = observed_metrics[observed_metrics["model"].isin(["CIR", "Vasicek"])].sort_values("observed_yield_rmse").reset_index(drop=True)
    winner_fed = fair_fed.iloc[0]
    winner_observed = fair_observed.iloc[0]
    overall_fed = fed_metrics[~fed_metrics["model"].eq("Fed published benchmark")].sort_values("fed_yield_rmse").reset_index(drop=True).iloc[0]
    overall_observed = observed_metrics.sort_values("observed_yield_rmse").reset_index(drop=True).iloc[0]

    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">Fixed Income / Phase 1</div>
            <h1>Public Treasury Curve Construction With Fed Benchmark Validation</h1>
            <p>
                A one-page professor-facing view of the project: observed U.S. Treasury quotes, our own bootstrap and NSS smoothing,
                the official Fed nominal curve, and the one-factor model comparison layered on top.
            </p>
            <div class="cta-row">
                <a class="ui-btn" href="#benchmark-story">Benchmark Story</a>
                <a class="ui-btn secondary" href="#curve-snapshot">Curve Snapshot</a>
                <a class="ui-btn secondary" href="#data-quality">Data Quality</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(4)
    cols[0].metric("Sample range", f"{fetch_summary['start_date']} to {fetch_summary['end_date']}")
    cols[1].metric("Monthly snapshots", str(bootstrap_summary["monthly_snapshot_count"]))
    cols[2].metric("Best equilibrium vs Fed", f"{winner_fed['model']} ({winner_fed['fed_yield_rmse'] * 10000:.2f} bps)")
    cols[3].metric("Best overall vs observed", f"{overall_observed['model']} ({overall_observed['observed_yield_rmse'] * 10000:.2f} bps)")

    st.markdown(
        """
        <div class="card-note">
            <strong>Reading rule.</strong> Start from observed Treasury quotes, then compare our curve to the Fed published benchmark,
            then compare the short-rate models to those benchmark layers. Hull-White is the anchored exact-fit reference and therefore the overall best-fit model layer; the fair structural contest is CIR versus Vasicek.
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_header(
        "Benchmark Story",
        "The project now uses three benchmark layers: observed Treasury quotes, the official Fed nominal curve, and our internal NSS smoothing layer.",
    )
    benchmark_cols = st.columns(2)
    with benchmark_cols[0]:
        observed_plot = observed_metrics.copy()
        observed_plot["Observed RMSE (bps)"] = observed_plot["observed_yield_rmse"] * 10000.0
        fig_obs = px.bar(
            observed_plot,
            x="model",
            y="Observed RMSE (bps)",
            color="model",
            color_discrete_map=COLOR_MAP,
            title="Observed Treasury Quote Fit Error",
            text_auto=".2f",
        )
        fig_obs.update_layout(showlegend=False, xaxis_title="", yaxis_title="RMSE (bps)")
        st.plotly_chart(apply_plot_theme(fig_obs, theme), use_container_width=True)
    with benchmark_cols[1]:
        fed_plot = fed_metrics.copy()
        fed_plot["Fed RMSE (bps)"] = fed_plot["fed_yield_rmse"] * 10000.0
        fig_fed = px.bar(
            fed_plot,
            x="model",
            y="Fed RMSE (bps)",
            color="model",
            color_discrete_map=COLOR_MAP,
            title="Official Fed Curve Fit Error",
            text_auto=".2f",
        )
        fig_fed.update_layout(showlegend=False, xaxis_title="", yaxis_title="RMSE (bps)")
        st.plotly_chart(apply_plot_theme(fig_fed, theme), use_container_width=True)

    scoreboard = (
        observed_metrics.rename(columns={"observed_yield_rmse": "observed_rmse", "observed_yield_mae": "observed_mae"})
        .merge(fed_metrics.rename(columns={"fed_yield_rmse": "fed_rmse", "fed_yield_mae": "fed_mae"}), on=["model", "points"], how="outer")
    )
    scoreboard["Observed RMSE (bps)"] = scoreboard["observed_rmse"].map(lambda value: f"{value * 10000:.2f}" if pd.notna(value) else "")
    scoreboard["Fed RMSE (bps)"] = scoreboard["fed_rmse"].map(lambda value: f"{value * 10000:.2f}" if pd.notna(value) else "")
    st.dataframe(
        scoreboard[["model", "Observed RMSE (bps)", "Fed RMSE (bps)", "points"]].rename(columns={"model": "Model", "points": "Points"}),
        use_container_width=True,
        hide_index=True,
    )

    section_header(
        "Curve Snapshot",
        "A single date view that makes the benchmark hierarchy visually obvious: orange markers for observed quotes, purple for the Fed curve, black for our NSS smoothing, then the model curves.",
    )
    available_dates = sorted(pd.to_datetime(monthly_rates["date"]).dt.date.unique())
    selected_date = st.selectbox("Snapshot date", options=available_dates, index=len(available_dates) - 1, key="onepage_date")
    st.plotly_chart(curve_snapshot_figure(selected_date, monthly_rates, fed_curves, nss_curves, model_curves, theme), use_container_width=True)

    section_header(
        "Internal Structural Contest",
        "Once the benchmark layers are established, the real model contest is CIR versus Vasicek, with Hull-White treated separately as the exact-fit anchor.",
    )
    internal_cols = st.columns(2)
    with internal_cols[0]:
        fit_plot = fit_metrics.copy()
        fit_plot["Zero RMSE (bps)"] = fit_plot["zero_yield_rmse"] * 10000.0
        fig_fit = px.bar(
            fit_plot,
            x="model",
            y="Zero RMSE (bps)",
            color="model",
            color_discrete_map=COLOR_MAP,
            title="Internal NSS Benchmark Fit Error",
            text_auto=".2f",
        )
        fig_fit.update_layout(showlegend=False, xaxis_title="", yaxis_title="RMSE (bps)")
        st.plotly_chart(apply_plot_theme(fig_fit, theme), use_container_width=True)
    with internal_cols[1]:
        maturity = st.selectbox("Pricing maturity", options=[2.0, 5.0, 10.0, 30.0], index=2, key="onepage_maturity")
        swap_slice = swap_metrics[swap_metrics["maturity_years"] == maturity].copy()
        swap_slice["Swap RMSE (bps)"] = swap_slice["swap_rate_rmse"] * 10000.0
        fig_swap = px.bar(
            swap_slice,
            x="model",
            y="Swap RMSE (bps)",
            color="model",
            color_discrete_map=COLOR_MAP,
            title=f"Par Swap RMSE at {int(maturity)}Y",
            text_auto=".2f",
        )
        fig_swap.update_layout(showlegend=False, xaxis_title="", yaxis_title="RMSE (bps)")
        st.plotly_chart(apply_plot_theme(fig_swap, theme), use_container_width=True)

    section_header(
        "Pricing and Stability",
        "Phase 1 is not just a curve-fitting exercise. It also checks bond pricing usefulness and the stability of the calibrated parameters.",
    )
    pricing_cols = st.columns(2)
    with pricing_cols[0]:
        detail = pricing_details[pricing_details["maturity_years"] == maturity].copy()
        detail["abs_pricing_error"] = detail["pricing_error"].abs()
        fig_price = px.line(
            detail,
            x="date",
            y="abs_pricing_error",
            color="model",
            color_discrete_map=COLOR_MAP,
            title=f"Absolute Bond Pricing Error Through Time at {int(maturity)}Y",
        )
        fig_price.update_layout(xaxis_title="", yaxis_title="Absolute price error")
        st.plotly_chart(apply_plot_theme(fig_price, theme), use_container_width=True)
    with pricing_cols[1]:
        stability = load_metric_csv("parameter_stability_metrics.csv")
        stability_view = stability[stability["parameter"].isin(["kappa", "theta", "sigma", "rmse"])].copy()
        st.dataframe(stability_view, use_container_width=True, hide_index=True)

    section_header(
        "Data Quality",
        "The benchmark implementation is only useful if the underlying curve construction and Fed reconstruction are validated explicitly.",
    )
    dq_cols = st.columns(4)
    dq_cols[0].metric("Toy bootstrap check", "Passed" if abs(bootstrap_summary["toy_bootstrap_check"]["bootstrap_equation_residual_2y"]) < 1e-10 else "Review")
    dq_cols[1].metric("Fed reconstruction", f"{fed_summary['reconstruction_rmse_bps']:.4f} bps")
    dq_cols[2].metric("TP1 overlap dates", str(fed_summary["tp1_validation"].get("overlap_dates", 0)))
    dq_cols[3].metric("TP1 validation", f"{fed_summary['tp1_validation'].get('yield_rmse_bps', 0.0):.4f} bps")
    st.markdown(
        """
        <div class="card-note">
            <strong>Validation chain.</strong> Public Treasury quotes -> bootstrap -> NSS smoothing -> official Fed curve -> short-rate models.
            The TP1 coefficient file is used only as a small overlap consistency check, not as the main benchmark source.
        </div>
        """,
        unsafe_allow_html=True,
    )
    dq_table = pd.DataFrame(
        [
            {"Check": "Bootstrap discount factors positive", "Result": "Pass" if bootstrap_summary["all_discount_positive"] else "Fail"},
            {"Check": "Bootstrap discount factors monotone", "Result": "Pass" if bootstrap_summary["all_discount_monotonic"] else "Fail"},
            {"Check": "Hull-White anchored fit reproduced", "Result": "Pass" if fit_summary["hull_white_max_abs_error"] == 0.0 else "Review"},
            {"Check": "Fed coefficient reconstruction", "Result": "Pass" if fed_summary["reconstruction_rmse_bps"] < 0.1 else "Review"},
            {"Check": "Fed data coverage", "Result": f"{fed_fetch_summary['start_date']} to {fed_fetch_summary['end_date']}"},
        ]
    )
    st.dataframe(dq_table, use_container_width=True, hide_index=True)

    section_header(
        "Saved Figures",
        "The one-page app is presentation-first, but the original saved pipeline figures are still available below.",
    )
    for image_path in sorted(FIGURES_DIR.glob("*.png")):
        st.image(str(image_path), caption=image_path.name, use_container_width=True)


if __name__ == "__main__":
    main()
