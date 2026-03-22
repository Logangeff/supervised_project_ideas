from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import (
    CALIBRATED_SURFACE_EXTENSION_PREDICTIONS_CSV,
    PHASE2_CUMULATIVE_VALUE_FIGURE,
    PHASE2_DECISION_DAILY_RETURNS_CSV,
    PHASE2_DECISION_EXTENSION_METRICS_CSV,
    PHASE2_DECISION_EXTENSION_METRICS_JSON,
    PHASE2_DECISION_METRICS_CSV,
    PHASE2_DECISION_METRICS_JSON,
    PHASE2_DRAWDOWN_FIGURE,
    PHASE2_EXPOSURE_FIGURE,
    STOCK_PANEL_PATH,
    SURFACE_EXTENSION_PREDICTIONS_CSV,
    TRADING_DAYS_PER_YEAR,
)
from .utils import write_json, write_rows_csv


CORE_STRATEGIES: list[dict[str, object]] = [
    {"strategy_name": "equal_weight_full", "signal_column": None, "weight_method": "equal_weight", "parameter": None},
    {"strategy_name": "stock_only_scaled", "signal_column": "stock_only_prob", "weight_method": "cash_scaled", "parameter": None},
    {"strategy_name": "option_only_scaled", "signal_column": "option_only_prob", "weight_method": "cash_scaled", "parameter": None},
    {"strategy_name": "all_features_scaled", "signal_column": "all_features_prob", "weight_method": "cash_scaled", "parameter": None},
    {"strategy_name": "stock_only_full_linear", "signal_column": "stock_only_prob", "weight_method": "fully_invested_linear", "parameter": None},
    {"strategy_name": "stock_only_full_exp2", "signal_column": "stock_only_prob", "weight_method": "fully_invested_exponential", "parameter": 2.0},
    {"strategy_name": "stock_only_top10_exclusion", "signal_column": "stock_only_prob", "weight_method": "top_quantile_exclusion", "parameter": 0.90},
    {"strategy_name": "option_only_full_linear", "signal_column": "option_only_prob", "weight_method": "fully_invested_linear", "parameter": None},
    {"strategy_name": "option_only_full_exp2", "signal_column": "option_only_prob", "weight_method": "fully_invested_exponential", "parameter": 2.0},
    {"strategy_name": "option_only_top10_exclusion", "signal_column": "option_only_prob", "weight_method": "top_quantile_exclusion", "parameter": 0.90},
    {"strategy_name": "all_features_full_linear", "signal_column": "all_features_prob", "weight_method": "fully_invested_linear", "parameter": None},
    {"strategy_name": "all_features_full_exp2", "signal_column": "all_features_prob", "weight_method": "fully_invested_exponential", "parameter": 2.0},
    {"strategy_name": "all_features_top10_exclusion", "signal_column": "all_features_prob", "weight_method": "top_quantile_exclusion", "parameter": 0.90},
]

CALIBRATED_EXTENSION_STRATEGIES: list[dict[str, object]] = [
    {"strategy_name": "equal_weight_full_calibrated", "signal_column": None, "weight_method": "equal_weight", "parameter": None},
    {"strategy_name": "beta_only_scaled", "signal_column": "beta_only_prob", "weight_method": "cash_scaled", "parameter": None},
    {"strategy_name": "option_beta_scaled", "signal_column": "option_beta_prob", "weight_method": "cash_scaled", "parameter": None},
    {"strategy_name": "all_extensions_scaled", "signal_column": "all_extensions_prob", "weight_method": "cash_scaled", "parameter": None},
    {"strategy_name": "beta_only_full_linear", "signal_column": "beta_only_prob", "weight_method": "fully_invested_linear", "parameter": None},
    {"strategy_name": "option_beta_full_linear", "signal_column": "option_beta_prob", "weight_method": "fully_invested_linear", "parameter": None},
    {"strategy_name": "all_extensions_full_linear", "signal_column": "all_extensions_prob", "weight_method": "fully_invested_linear", "parameter": None},
    {"strategy_name": "all_extensions_full_exp2", "signal_column": "all_extensions_prob", "weight_method": "fully_invested_exponential", "parameter": 2.0},
    {"strategy_name": "all_extensions_top10_exclusion", "signal_column": "all_extensions_prob", "weight_method": "top_quantile_exclusion", "parameter": 0.90},
]

CORE_PLOT_STRATEGIES = [
    "equal_weight_full",
    "stock_only_scaled",
    "stock_only_full_linear",
    "all_features_scaled",
    "all_features_full_linear",
    "all_features_top10_exclusion",
]

PLOT_COLORS = {
    "equal_weight_full": "#1f1f1f",
    "stock_only_scaled": "#d95f02",
    "stock_only_full_linear": "#e6ab02",
    "all_features_scaled": "#7570b3",
    "all_features_full_linear": "#1b9e77",
    "all_features_top10_exclusion": "#66a61e",
}

PLOT_LABELS = {
    "equal_weight_full": "Equal-weight full",
    "stock_only_scaled": "Stock-only cash scaled",
    "stock_only_full_linear": "Stock-only full linear",
    "all_features_scaled": "All-features cash scaled",
    "all_features_full_linear": "All-features full linear",
    "all_features_top10_exclusion": "All-features top-10% exclusion",
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


def _load_predictions(path: Path, required_columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise RuntimeError(f"Prediction file is missing: {path.name}")
    frame = pd.read_csv(path, parse_dates=["trade_date"])
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise RuntimeError(f"Prediction file {path.name} is missing required columns: {missing}")
    frame = frame[frame["split"].isin(["validation", "test"])].copy()
    if frame.empty:
        raise RuntimeError(f"Prediction file {path.name} has no validation/test rows.")
    return frame


def _prepare_signal_panel(predictions: pd.DataFrame) -> pd.DataFrame:
    stock_panel = _load_stock_panel()[["permno", "trade_date", "ret"]].copy()
    stock_panel = stock_panel.sort_values(["permno", "trade_date"]).reset_index(drop=True)
    stock_panel["ret_next_1d"] = stock_panel.groupby("permno", sort=False)["ret"].shift(-1)
    stock_panel["return_date"] = stock_panel.groupby("permno", sort=False)["trade_date"].shift(-1)

    panel = predictions.merge(stock_panel[["permno", "trade_date", "ret_next_1d", "return_date"]], on=["permno", "trade_date"], how="left")
    panel = panel.dropna(subset=["ret_next_1d", "return_date"]).reset_index(drop=True)
    panel["return_date"] = pd.to_datetime(panel["return_date"])
    panel["ret_next_1d"] = pd.to_numeric(panel["ret_next_1d"], errors="coerce")
    panel = panel.dropna(subset=["ret_next_1d"]).reset_index(drop=True)
    return panel


def _apply_weight_rule(
    strategy_panel: pd.DataFrame,
    weight_method: str,
    parameter: float | None,
) -> tuple[pd.Series, pd.Series]:
    base_weight = 1.0 / strategy_panel.groupby(["split", "trade_date"])["permno"].transform("count")
    signal_probability = strategy_panel["predicted_high_rv_probability"].astype(float).clip(lower=0.0, upper=1.0)

    if weight_method == "equal_weight":
        return base_weight, pd.Series(np.ones(len(strategy_panel)), index=strategy_panel.index)

    if weight_method == "cash_scaled":
        multiplier = 1.0 - signal_probability
        return base_weight * multiplier, multiplier

    if weight_method == "fully_invested_linear":
        raw_scores = (1.0 - signal_probability).clip(lower=1e-6)
    elif weight_method == "fully_invested_exponential":
        strength = float(parameter if parameter is not None else 2.0)
        raw_scores = pd.Series(np.exp(-strength * signal_probability.to_numpy(dtype=float)), index=strategy_panel.index)
    elif weight_method == "top_quantile_exclusion":
        keep_quantile = float(parameter if parameter is not None else 0.90)
        thresholds = strategy_panel.groupby(["split", "trade_date"])["predicted_high_rv_probability"].transform(
            lambda series: float(series.quantile(keep_quantile))
        )
        raw_scores = (signal_probability <= thresholds).astype(float)
    else:
        raise RuntimeError(f"Unknown Phase 2 weight method: {weight_method}")

    denominator = strategy_panel.assign(raw_score=raw_scores).groupby(["split", "trade_date"])["raw_score"].transform("sum")
    fallback = base_weight.copy()
    fully_invested_weight = pd.Series(np.where(denominator > 0, raw_scores / denominator, fallback), index=strategy_panel.index, dtype=float)
    return fully_invested_weight, pd.Series(np.ones(len(strategy_panel)), index=strategy_panel.index)


def _compute_turnover(weight_frame: pd.DataFrame) -> pd.DataFrame:
    turnover_rows: list[dict[str, object]] = []
    for split_name, split_frame in weight_frame.groupby("split", sort=False):
        previous_weights: pd.Series | None = None
        for trade_date, date_frame in split_frame.groupby("trade_date", sort=True):
            current_weights = date_frame.set_index("permno")["weight"].astype(float)
            if previous_weights is None:
                turnover = 0.0
            else:
                aligned = pd.concat(
                    [previous_weights.rename("previous_weight"), current_weights.rename("current_weight")],
                    axis=1,
                ).fillna(0.0)
                turnover = 0.5 * float((aligned["current_weight"] - aligned["previous_weight"]).abs().sum())
            turnover_rows.append({"split": split_name, "trade_date": trade_date, "turnover": turnover})
            previous_weights = current_weights
    return pd.DataFrame(turnover_rows)


def _daily_weight_concentration(weight_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (split_name, trade_date), date_frame in weight_frame.groupby(["split", "trade_date"], sort=True):
        weights = date_frame["weight"].astype(float)
        rows.append(
            {
                "split": split_name,
                "trade_date": trade_date,
                "top10_weight_share": float(weights.nlargest(min(10, len(weights))).sum()),
                "effective_n": float(1.0 / np.square(weights).sum()),
            }
        )
    return pd.DataFrame(rows)


def _compute_strategy_outputs(
    panel: pd.DataFrame,
    strategy_config: dict[str, object],
    comparison_set: str,
) -> tuple[pd.DataFrame, list[dict[str, object]], dict[str, object]]:
    strategy_name = str(strategy_config["strategy_name"])
    signal_column = strategy_config["signal_column"]
    weight_method = str(strategy_config["weight_method"])
    parameter = strategy_config.get("parameter")

    strategy_panel = panel.copy()
    if signal_column is None:
        strategy_panel["predicted_high_rv_probability"] = 0.0
    else:
        strategy_panel["predicted_high_rv_probability"] = (
            pd.to_numeric(strategy_panel[str(signal_column)], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
        )

    strategy_panel["weight"], strategy_panel["exposure_multiplier"] = _apply_weight_rule(strategy_panel, weight_method, parameter)
    strategy_panel["weighted_return"] = strategy_panel["weight"] * strategy_panel["ret_next_1d"]

    daily = (
        strategy_panel.groupby(["split", "trade_date"], sort=True)
        .agg(
            return_date=("return_date", "max"),
            stock_count=("permno", "count"),
            portfolio_return=("weighted_return", "sum"),
            equity_exposure=("weight", "sum"),
            average_signal_probability=("predicted_high_rv_probability", "mean"),
            max_signal_probability=("predicted_high_rv_probability", "max"),
        )
        .reset_index()
    )
    daily["cash_weight"] = 1.0 - daily["equity_exposure"]

    turnover_frame = _compute_turnover(strategy_panel[["split", "trade_date", "permno", "weight"]])
    concentration_frame = _daily_weight_concentration(strategy_panel[["split", "trade_date", "permno", "weight"]])
    daily = daily.merge(turnover_frame, on=["split", "trade_date"], how="left")
    daily = daily.merge(concentration_frame, on=["split", "trade_date"], how="left")
    daily["turnover"] = daily["turnover"].fillna(0.0)

    daily_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    split_summary: dict[str, object] = {}

    for split_name, split_frame in daily.groupby("split", sort=False):
        split_frame = split_frame.sort_values("trade_date").reset_index(drop=True)
        split_frame["cumulative_value"] = (1.0 + split_frame["portfolio_return"]).cumprod()
        running_peak = split_frame["cumulative_value"].cummax()
        split_frame["drawdown"] = split_frame["cumulative_value"] / running_peak - 1.0
        split_frame["strategy"] = strategy_name
        split_frame["comparison_set"] = comparison_set
        split_frame["signal_column"] = signal_column or "none"
        split_frame["weight_method"] = weight_method
        split_frame["parameter"] = parameter if parameter is not None else ""

        for row in split_frame.itertuples(index=False):
            daily_rows.append(
                {
                    "comparison_set": comparison_set,
                    "strategy": strategy_name,
                    "signal_column": signal_column or "none",
                    "weight_method": weight_method,
                    "parameter": parameter if parameter is not None else "",
                    "split": split_name,
                    "signal_date": pd.Timestamp(row.trade_date).date().isoformat(),
                    "return_date": pd.Timestamp(row.return_date).date().isoformat(),
                    "portfolio_return": float(row.portfolio_return),
                    "cumulative_value": float(row.cumulative_value),
                    "drawdown": float(row.drawdown),
                    "equity_exposure": float(row.equity_exposure),
                    "cash_weight": float(row.cash_weight),
                    "turnover": float(row.turnover),
                    "stock_count": int(row.stock_count),
                    "average_signal_probability": float(row.average_signal_probability),
                    "max_signal_probability": float(row.max_signal_probability),
                    "top10_weight_share": float(row.top10_weight_share),
                    "effective_n": float(row.effective_n),
                }
            )

        returns = split_frame["portfolio_return"].astype(float)
        trading_days = int(len(split_frame))
        cumulative_terminal = float(split_frame["cumulative_value"].iloc[-1])
        annualized_return = float(cumulative_terminal ** (TRADING_DAYS_PER_YEAR / trading_days) - 1.0)
        volatility = float(returns.std(ddof=0))
        annualized_volatility = float(volatility * np.sqrt(TRADING_DAYS_PER_YEAR))
        sharpe_ratio = float(returns.mean() / volatility * np.sqrt(TRADING_DAYS_PER_YEAR)) if volatility > 0 else 0.0
        cvar_cutoff = float(returns.quantile(0.05))
        tail_returns = returns[returns <= cvar_cutoff]
        cvar_95 = float(tail_returns.mean()) if not tail_returns.empty else float(cvar_cutoff)

        split_summary[split_name] = {
            "annualized_return": annualized_return,
            "annualized_volatility": annualized_volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": float(split_frame["drawdown"].min()),
            "cvar_95": cvar_95,
            "average_equity_exposure": float(split_frame["equity_exposure"].mean()),
            "average_cash_weight": float(split_frame["cash_weight"].mean()),
            "annualized_turnover": float(split_frame["turnover"].mean() * TRADING_DAYS_PER_YEAR),
            "total_return": float(cumulative_terminal - 1.0),
            "trading_days": trading_days,
            "mean_daily_return": float(returns.mean()),
            "daily_return_std": volatility,
            "average_signal_probability": float(split_frame["average_signal_probability"].mean()),
            "max_signal_probability": float(split_frame["max_signal_probability"].max()),
            "average_stock_count": float(split_frame["stock_count"].mean()),
            "average_top10_weight_share": float(split_frame["top10_weight_share"].mean()),
            "average_effective_n": float(split_frame["effective_n"].mean()),
        }
        metric_rows.append(
            {
                "comparison_set": comparison_set,
                "strategy": strategy_name,
                "signal_column": signal_column or "none",
                "weight_method": weight_method,
                "parameter": parameter if parameter is not None else "",
                "split": split_name,
                **split_summary[split_name],
            }
        )

    return pd.DataFrame(daily_rows), metric_rows, split_summary


def _run_strategy_set(
    predictions_path: Path,
    strategies: list[dict[str, object]],
    comparison_set: str,
) -> tuple[dict[str, object], list[dict[str, object]], pd.DataFrame]:
    required_columns = ["permno", "universe_ticker", "trade_date", "split", "label"] + [
        str(strategy["signal_column"])
        for strategy in strategies
        if strategy["signal_column"] is not None
    ]
    predictions = _load_predictions(predictions_path, required_columns)
    signal_panel = _prepare_signal_panel(predictions)
    if signal_panel.empty:
        raise RuntimeError(f"No usable next-day return rows were available after merging {predictions_path.name}.")

    daily_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    strategy_summary: dict[str, object] = {}

    for strategy_config in strategies:
        strategy_daily, strategy_metric_rows, split_summary = _compute_strategy_outputs(
            signal_panel,
            strategy_config=strategy_config,
            comparison_set=comparison_set,
        )
        daily_frames.append(strategy_daily)
        metric_rows.extend(strategy_metric_rows)
        strategy_summary[str(strategy_config["strategy_name"])] = {
            "signal_column": strategy_config["signal_column"],
            "weight_method": strategy_config["weight_method"],
            "parameter": strategy_config.get("parameter"),
            "splits": split_summary,
        }

    daily_output = pd.concat(daily_frames, ignore_index=True).sort_values(["comparison_set", "strategy", "split", "signal_date"])
    summary = {
        "comparison_set": comparison_set,
        "prediction_file": predictions_path.name,
        "usable_signal_rows": int(len(signal_panel)),
        "split_row_counts": {
            split_name: int(len(signal_panel[signal_panel["split"] == split_name])) for split_name in ("validation", "test")
        },
        "split_date_ranges": {
            split_name: {
                "min_signal_date": str(signal_panel.loc[signal_panel["split"] == split_name, "trade_date"].min().date()),
                "max_signal_date": str(signal_panel.loc[signal_panel["split"] == split_name, "trade_date"].max().date()),
            }
            for split_name in ("validation", "test")
        },
        "strategies": strategy_summary,
        "rows": metric_rows,
    }
    return summary, metric_rows, daily_output


def _plot_core_phase2_figures(daily_returns: pd.DataFrame) -> None:
    core_daily = daily_returns[daily_returns["comparison_set"] == "surface_common"].copy()
    if core_daily.empty:
        return

    plt.style.use("seaborn-v0_8-whitegrid")

    def plot_metric(metric_name: str, ylabel: str, output_path: Path, title: str) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
        for axis, split_name in zip(axes, ["validation", "test"]):
            split_frame = core_daily[core_daily["split"] == split_name]
            for strategy_name in CORE_PLOT_STRATEGIES:
                strategy_frame = split_frame[split_frame["strategy"] == strategy_name].sort_values("signal_date")
                if strategy_frame.empty:
                    continue
                axis.plot(
                    pd.to_datetime(strategy_frame["signal_date"]),
                    strategy_frame[metric_name],
                    label=PLOT_LABELS[strategy_name],
                    linewidth=2.2,
                    color=PLOT_COLORS[strategy_name],
                )
            axis.set_title(split_name.capitalize())
            axis.set_xlabel("Decision date")
            axis.set_ylabel(ylabel)
            axis.legend(frameon=True, fontsize=8)
        fig.suptitle(title)
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

    plot_metric("cumulative_value", "Cumulative portfolio value", PHASE2_CUMULATIVE_VALUE_FIGURE, "Phase 2 cumulative value")
    plot_metric("drawdown", "Drawdown", PHASE2_DRAWDOWN_FIGURE, "Phase 2 drawdown")
    plot_metric("equity_exposure", "Average equity exposure", PHASE2_EXPOSURE_FIGURE, "Phase 2 equity exposure")


def run_phase2_decision() -> dict[str, object]:
    core_summary, core_rows, core_daily = _run_strategy_set(
        SURFACE_EXTENSION_PREDICTIONS_CSV,
        strategies=CORE_STRATEGIES,
        comparison_set="surface_common",
    )

    extension_summary = None
    extension_rows: list[dict[str, object]] = []
    extension_daily = pd.DataFrame()
    if CALIBRATED_SURFACE_EXTENSION_PREDICTIONS_CSV.exists():
        extension_summary, extension_rows, extension_daily = _run_strategy_set(
            CALIBRATED_SURFACE_EXTENSION_PREDICTIONS_CSV,
            strategies=CALIBRATED_EXTENSION_STRATEGIES,
            comparison_set="calibrated_common",
        )
        write_json(PHASE2_DECISION_EXTENSION_METRICS_JSON, extension_summary)
        write_rows_csv(PHASE2_DECISION_EXTENSION_METRICS_CSV, extension_rows)

    all_daily = pd.concat([core_daily, extension_daily], ignore_index=True) if not extension_daily.empty else core_daily.copy()
    write_rows_csv(PHASE2_DECISION_DAILY_RETURNS_CSV, all_daily.to_dict(orient="records"))
    write_json(PHASE2_DECISION_METRICS_JSON, core_summary)
    write_rows_csv(PHASE2_DECISION_METRICS_CSV, core_rows)
    _plot_core_phase2_figures(all_daily)

    return {
        "core": core_summary,
        "extension": extension_summary,
        "daily_returns_path": str(PHASE2_DECISION_DAILY_RETURNS_CSV),
        "figures": [
            PHASE2_CUMULATIVE_VALUE_FIGURE.name,
            PHASE2_DRAWDOWN_FIGURE.name,
            PHASE2_EXPOSURE_FIGURE.name,
        ],
    }
