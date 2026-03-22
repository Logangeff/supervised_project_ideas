from __future__ import annotations

import json
from pathlib import Path

from .config import (
    BUILD_CALIBRATED_SURFACE_SUMMARY_PATH,
    BUILD_TEXT_NEWS_PANEL_SUMMARY_PATH,
    BUILD_OPTION_FEATURES_SUMMARY_PATH,
    BUILD_SURFACE_FACTORS_SUMMARY_PATH,
    BUILD_STOCK_PANEL_SUMMARY_PATH,
    CALIBRATED_SURFACE_EXTENSION_METRICS_JSON,
    EXTRACT_OPTION_SUMMARY_PATH,
    EXTRACT_CALIBRATED_SURFACE_INPUTS_SUMMARY_PATH,
    EXTRACT_STOCK_SUMMARY_PATH,
    PART1_METRICS_JSON,
    PART2_METRICS_JSON,
    PHASE2_BUCKET_EXTENSION_METRICS_JSON,
    PHASE2_BUCKET_METRICS_JSON,
    PHASE2_DECISION_EXTENSION_METRICS_JSON,
    PHASE2_DECISION_METRICS_JSON,
    SMOKE_SUMMARY_PATH,
    SURFACE_EXTENSION_METRICS_JSON,
    TEXT_NEWS_EXTENSION_METRICS_JSON,
)


def _load_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def print_saved_results_overview() -> str:
    sections: list[str] = []
    stock_extract = _load_json(EXTRACT_STOCK_SUMMARY_PATH)
    if stock_extract:
        sections.append(
            "Stock Extract\n"
            f"  Source: {stock_extract.get('source')}\n"
            f"  Resolved PERMNOs: {stock_extract.get('resolved_permnos')}\n"
            f"  Daily rows: {stock_extract.get('stock_daily_rows')}"
        )
    stock_panel = _load_json(BUILD_STOCK_PANEL_SUMMARY_PATH)
    if stock_panel:
        sections.append(
            "Stock Panel\n"
            f"  Rows: {stock_panel.get('panel_rows')}\n"
            f"  PERMNO count: {stock_panel.get('permno_count')}\n"
            f"  Train threshold: {stock_panel.get('train_threshold_future_rv_20d'):.6f}"
        )
    part1 = _load_json(PART1_METRICS_JSON)
    if part1:
        lines = []
        for model_name, model_summary in part1.get("models", {}).items():
            test_metrics = model_summary.get("test", {})
            lines.append(f"  {model_name}: test macro_f1={test_metrics.get('macro_f1'):.4f}, pr_auc={test_metrics.get('pr_auc'):.4f}")
        sections.append("Part 1\n" + "\n".join(lines))
    option_extract = _load_json(EXTRACT_OPTION_SUMMARY_PATH)
    if option_extract:
        sections.append(
            "Option Extract\n"
            f"  SECIDs: {option_extract.get('secid_count')}\n"
            f"  Option rows: {option_extract.get('option_price_rows')}"
        )
    option_build = _load_json(BUILD_OPTION_FEATURES_SUMMARY_PATH)
    if option_build:
        sections.append(
            "Option Features\n"
            f"  Feature rows: {option_build.get('option_feature_rows')}\n"
            f"  Complete-case rows: {option_build.get('complete_case_rows')}"
        )
    surface_build = _load_json(BUILD_SURFACE_FACTORS_SUMMARY_PATH)
    if surface_build:
        sections.append(
            "Surface Factors\n"
            f"  Surface rows: {surface_build.get('surface_panel_rows')}\n"
            f"  Surface extension rows: {surface_build.get('surface_extension_rows')}"
        )
    calibrated_input_extract = _load_json(EXTRACT_CALIBRATED_SURFACE_INPUTS_SUMMARY_PATH)
    if calibrated_input_extract:
        sections.append(
            "Calibrated Surface Inputs\n"
            f"  SECIDs: {calibrated_input_extract.get('secid_count')}\n"
            f"  Quote files: {len(calibrated_input_extract.get('quote_files', []))}\n"
            f"  Forward files: {len(calibrated_input_extract.get('forward_files', []))}"
        )
    calibrated_build = _load_json(BUILD_CALIBRATED_SURFACE_SUMMARY_PATH)
    if calibrated_build:
        sections.append(
            "Calibrated Surface\n"
            f"  Beta rows: {calibrated_build.get('beta_panel_rows')}\n"
            f"  PERMNO count: {calibrated_build.get('permno_count')}\n"
            f"  Median R2: {calibrated_build.get('surface_beta_r2_summary', {}).get('median'):.4f}"
        )
    text_news_build = _load_json(BUILD_TEXT_NEWS_PANEL_SUMMARY_PATH)
    if text_news_build:
        sections.append(
            "Text News Panel\n"
            f"  Covered tickers: {text_news_build.get('covered_ticker_count')}\n"
            f"  Article-ticker rows: {text_news_build.get('article_ticker_rows')}\n"
            f"  Extension rows: {text_news_build.get('extension_panel_rows')}"
        )
    part2 = _load_json(PART2_METRICS_JSON)
    if part2:
        lines = []
        for model_name, model_summary in part2.get("models", {}).items():
            test_metrics = model_summary.get("test", {})
            lines.append(f"  {model_name}: test macro_f1={test_metrics.get('macro_f1'):.4f}, pr_auc={test_metrics.get('pr_auc'):.4f}")
        sections.append("Part 2\n" + "\n".join(lines))
    surface_extension = _load_json(SURFACE_EXTENSION_METRICS_JSON)
    if surface_extension:
        lines = []
        for model_name, model_summary in surface_extension.get("models", {}).items():
            test_metrics = model_summary.get("test", {})
            lines.append(f"  {model_name}: test macro_f1={test_metrics.get('macro_f1'):.4f}, pr_auc={test_metrics.get('pr_auc'):.4f}")
        sections.append("Surface Extension\n" + "\n".join(lines))
    calibrated_surface_extension = _load_json(CALIBRATED_SURFACE_EXTENSION_METRICS_JSON)
    if calibrated_surface_extension:
        lines = []
        for model_name, model_summary in calibrated_surface_extension.get("models", {}).items():
            test_metrics = model_summary.get("test", {})
            lines.append(f"  {model_name}: test macro_f1={test_metrics.get('macro_f1'):.4f}, pr_auc={test_metrics.get('pr_auc'):.4f}")
        sections.append("Calibrated Surface Extension\n" + "\n".join(lines))
    text_news_extension = _load_json(TEXT_NEWS_EXTENSION_METRICS_JSON)
    if text_news_extension:
        lines = []
        for section_name, section_label in (
            ("full_panel_models", "Full"),
            ("event_panel_models", "Event"),
            ("models", "Legacy"),
        ):
            model_block = text_news_extension.get(section_name, {})
            for model_name, model_summary in model_block.items():
                test_metrics = model_summary.get("test", {})
                lines.append(
                    f"  [{section_label}] {model_name}: test macro_f1={test_metrics.get('macro_f1'):.4f}, "
                    f"pr_auc={test_metrics.get('pr_auc'):.4f}"
                )
        sections.append("Text News Extension\n" + "\n".join(lines))
    phase2_core = _load_json(PHASE2_DECISION_METRICS_JSON)
    if phase2_core:
        ranked = sorted(
            phase2_core.get("strategies", {}).items(),
            key=lambda item: (
                float(item[1].get("splits", {}).get("test", {}).get("sharpe_ratio", float("-inf"))),
                float(item[1].get("splits", {}).get("test", {}).get("annualized_return", float("-inf"))),
            ),
            reverse=True,
        )
        lines = []
        for strategy_name, strategy_summary in ranked[:5]:
            test_metrics = strategy_summary.get("splits", {}).get("test", {})
            lines.append(
                "  "
                f"{strategy_name}: test ann_return={test_metrics.get('annualized_return'):.4f}, "
                f"ann_vol={test_metrics.get('annualized_volatility'):.4f}, "
                f"sharpe={test_metrics.get('sharpe_ratio'):.4f}, "
                f"max_drawdown={test_metrics.get('max_drawdown'):.4f}"
            )
        sections.append("Phase 2 Decision\n" + "\n".join(lines))
    phase2_extension = _load_json(PHASE2_DECISION_EXTENSION_METRICS_JSON)
    if phase2_extension:
        ranked = sorted(
            phase2_extension.get("strategies", {}).items(),
            key=lambda item: (
                float(item[1].get("splits", {}).get("test", {}).get("sharpe_ratio", float("-inf"))),
                float(item[1].get("splits", {}).get("test", {}).get("annualized_return", float("-inf"))),
            ),
            reverse=True,
        )
        lines = []
        for strategy_name, strategy_summary in ranked[:5]:
            test_metrics = strategy_summary.get("splits", {}).get("test", {})
            lines.append(
                "  "
                f"{strategy_name}: test ann_return={test_metrics.get('annualized_return'):.4f}, "
                f"ann_vol={test_metrics.get('annualized_volatility'):.4f}, "
                f"sharpe={test_metrics.get('sharpe_ratio'):.4f}, "
                f"max_drawdown={test_metrics.get('max_drawdown'):.4f}"
            )
        sections.append("Phase 2 Extension\n" + "\n".join(lines))
    phase2_bucket = _load_json(PHASE2_BUCKET_METRICS_JSON)
    if phase2_bucket:
        ranked = sorted(
            phase2_bucket.get("signals", {}).items(),
            key=lambda item: (
                float(item[1].get("splits", {}).get("test", {}).get("average_daily_rank_ic_future_rv", float("-inf"))),
                float(item[1].get("splits", {}).get("test", {}).get("top_bottom_future_rv_spread", float("-inf"))),
            ),
            reverse=True,
        )
        lines = []
        for signal_name, signal_summary in ranked:
            test_metrics = signal_summary.get("splits", {}).get("test", {})
            lines.append(
                "  "
                f"{signal_name}: test rank_ic={test_metrics.get('average_daily_rank_ic_future_rv'):.4f}, "
                f"rv_spread={test_metrics.get('top_bottom_future_rv_spread'):.4f}, "
                f"hit_spread={test_metrics.get('top_bottom_hit_rate_spread'):.4f}, "
                f"monotonic_share={test_metrics.get('monotonic_future_rv_share'):.4f}"
            )
        sections.append("Phase 2 Bucket Analysis\n" + "\n".join(lines))
    phase2_bucket_extension = _load_json(PHASE2_BUCKET_EXTENSION_METRICS_JSON)
    if phase2_bucket_extension:
        ranked = sorted(
            phase2_bucket_extension.get("signals", {}).items(),
            key=lambda item: (
                float(item[1].get("splits", {}).get("test", {}).get("average_daily_rank_ic_future_rv", float("-inf"))),
                float(item[1].get("splits", {}).get("test", {}).get("top_bottom_future_rv_spread", float("-inf"))),
            ),
            reverse=True,
        )
        lines = []
        for signal_name, signal_summary in ranked:
            test_metrics = signal_summary.get("splits", {}).get("test", {})
            lines.append(
                "  "
                f"{signal_name}: test rank_ic={test_metrics.get('average_daily_rank_ic_future_rv'):.4f}, "
                f"rv_spread={test_metrics.get('top_bottom_future_rv_spread'):.4f}, "
                f"hit_spread={test_metrics.get('top_bottom_hit_rate_spread'):.4f}, "
                f"monotonic_share={test_metrics.get('monotonic_future_rv_share'):.4f}"
            )
        sections.append("Phase 2 Bucket Extension\n" + "\n".join(lines))
    smoke = _load_json(SMOKE_SUMMARY_PATH)
    if smoke:
        sections.append(
            "Smoke\n"
            f"  Stock panel rows: {smoke.get('stock_panel_rows')}\n"
            f"  Complete-case rows: {smoke.get('complete_case_rows')}\n"
            f"  Part 1 test macro_f1: {smoke.get('part1_test_macro_f1'):.4f}"
        )
    return "\n\n".join(sections) if sections else "No saved results yet."
