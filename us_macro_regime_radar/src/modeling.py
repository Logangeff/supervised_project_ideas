from __future__ import annotations

from dataclasses import dataclass

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import IsotonicRegression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import (
    CYCLE_LABEL_PANEL_PATH,
    FIXED_BINARY_THRESHOLDS,
    MODELS_DIR,
    PHASE_CLASSES,
    PHASE_CLASS_TO_INT,
    PHASE_CONFUSION_DIR,
    PHASE_FORECAST_METRICS_CSV,
    PHASE_FORECAST_METRICS_JSON,
    PHASE_FORECAST_PREDICTIONS_CSV,
    PHASE_HORIZONS,
    PHASE_MODEL_PATH,
    RECESSION_MODEL_PATH,
    RECESSION_PR_CURVE_FIGURE,
    RECESSION_RISK_METRICS_CSV,
    RECESSION_RISK_METRICS_JSON,
    RECESSION_RISK_PREDICTIONS_CSV,
    RECESSION_ROC_CURVE_FIGURE,
    SEED,
    VALIDATION_END,
    WARMUP_TRAIN_END,
)
from .utils import save_json

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - runtime dependency
    XGBClassifier = None


PHASE_TARGET_COLUMNS = [f"phase_t_plus_{h}m_int" for h in PHASE_HORIZONS]


@dataclass
class ExpandingPredictionResult:
    predictions: pd.DataFrame
    metrics_by_split: dict[str, dict[str, float]]


def _load_labeled_panel() -> pd.DataFrame:
    panel = pd.read_parquet(CYCLE_LABEL_PANEL_PATH)
    panel["date"] = pd.to_datetime(panel["date"])
    return panel.sort_values("date").reset_index(drop=True)


def _feature_columns(panel: pd.DataFrame) -> list[str]:
    excluded_prefixes = ("USRECDM_", "phase_t_plus_")
    excluded_columns = {
        "date",
        "split",
        "current_phase",
        "current_phase_int",
        "recession_within_6m",
        "recession_start",
    }
    feature_cols = []
    for column in panel.columns:
        if column in excluded_columns:
            continue
        if column.startswith(excluded_prefixes):
            continue
        if column.endswith("_z"):
            continue
        if column in {"unrate_component"}:
            continue
        if pd.api.types.is_numeric_dtype(panel[column]):
            feature_cols.append(column)
    for column in ("activity_score", "level_score", "momentum_score"):
        if column in panel.columns and column not in feature_cols:
            feature_cols.append(column)
    return sorted(dict.fromkeys(feature_cols))


def _make_multiclass_model(model_name: str):
    if model_name == "phase_logreg":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        C=1.0,
                        random_state=SEED,
                    ),
                ),
            ]
        )
    if model_name == "phase_xgboost":
        if XGBClassifier is None:
            raise RuntimeError("xgboost is not installed. Install requirements.txt to train XGBoost models.")
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBClassifier(
                        objective="multi:softprob",
                        num_class=len(PHASE_CLASSES),
                        n_estimators=250,
                        max_depth=3,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        reg_lambda=1.0,
                        random_state=SEED,
                        eval_metric="mlogloss",
                    ),
                ),
            ]
        )
    raise RuntimeError(f"Unsupported phase model: {model_name}")


def _make_binary_model(model_name: str):
    if model_name in {"yield_curve_probit", "recession_logreg"}:
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        penalty="elasticnet" if model_name == "recession_logreg" else "l2",
                        solver="saga",
                        l1_ratio=0.5 if model_name == "recession_logreg" else None,
                        max_iter=2000,
                        C=0.5 if model_name == "recession_logreg" else 1.0,
                        random_state=SEED,
                    ),
                ),
            ]
        )
    if model_name == "recession_xgboost":
        if XGBClassifier is None:
            raise RuntimeError("xgboost is not installed. Install requirements.txt to train XGBoost models.")
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBClassifier(
                        objective="binary:logistic",
                        n_estimators=250,
                        max_depth=3,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        reg_lambda=1.0,
                        random_state=SEED,
                        eval_metric="logloss",
                    ),
                ),
            ]
        )
    raise RuntimeError(f"Unsupported recession model: {model_name}")


def _phase_metrics(y_true: list[int], y_pred: list[int], probabilities: np.ndarray) -> dict[str, float]:
    if not y_true:
        return {"accuracy": float("nan"), "macro_f1": float("nan"), "balanced_accuracy": float("nan")}
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def _binary_metrics(y_true: list[int], probabilities: list[float]) -> dict[str, float]:
    if not y_true:
        metrics = {
            "pr_auc": float("nan"),
            "auroc": float("nan"),
            "brier_score": float("nan"),
            "accuracy": float("nan"),
            "macro_f1": float("nan"),
            "balanced_accuracy": float("nan"),
        }
        for threshold in FIXED_BINARY_THRESHOLDS:
            metrics[f"precision_at_{threshold:.2f}"] = float("nan")
            metrics[f"recall_at_{threshold:.2f}"] = float("nan")
        return metrics
    scores = np.asarray(probabilities, dtype=float)
    labels = np.asarray(y_true, dtype=int)
    predicted = (scores >= 0.5).astype(int)
    metrics = {
        "pr_auc": float(average_precision_score(labels, scores)),
        "auroc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) > 1 else float("nan"),
        "brier_score": float(brier_score_loss(labels, scores)),
        "accuracy": float(accuracy_score(labels, predicted)),
        "macro_f1": float(f1_score(labels, predicted, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
    }
    for threshold in FIXED_BINARY_THRESHOLDS:
        hard = (scores >= threshold).astype(int)
        metrics[f"precision_at_{threshold:.2f}"] = float(precision_score(labels, hard, zero_division=0))
        metrics[f"recall_at_{threshold:.2f}"] = float(recall_score(labels, hard, zero_division=0))
    return metrics


def _expanding_prediction_rows(panel: pd.DataFrame, split_name: str) -> pd.DataFrame:
    if split_name == "validation":
        return panel[(panel["date"] > pd.Timestamp(WARMUP_TRAIN_END)) & (panel["date"] <= pd.Timestamp(VALIDATION_END))].copy()
    return panel[panel["date"] > pd.Timestamp(VALIDATION_END)].copy()


def _split_training_rows(panel: pd.DataFrame, split_name: str) -> pd.DataFrame:
    if split_name == "validation":
        return panel[panel["date"] <= pd.Timestamp(WARMUP_TRAIN_END)].copy()
    return panel[panel["date"] <= pd.Timestamp(VALIDATION_END)].copy()


def _usable_feature_columns(train_frame: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    return [column for column in feature_cols if column in train_frame.columns and train_frame[column].notna().any()]


def _phase_persistence_predictions(panel: pd.DataFrame, horizon: int) -> ExpandingPredictionResult:
    target_col = f"phase_t_plus_{horizon}m_int"
    rows: list[dict[str, object]] = []
    for split_name in ("validation", "test"):
        split_frame = _expanding_prediction_rows(panel, split_name)
        for _, row in split_frame.iterrows():
            if pd.isna(row["current_phase_int"]) or pd.isna(row[target_col]):
                continue
            label = int(row[target_col])
            predicted = int(row["current_phase_int"])
            probabilities = np.zeros(len(PHASE_CLASSES))
            probabilities[predicted] = 1.0
            rows.append(
                {
                    "date": row["date"],
                    "split": split_name,
                    "label": label,
                    "predicted_label": predicted,
                    **{f"prob_{name}": float(probabilities[idx]) for idx, name in enumerate(PHASE_CLASSES)},
                }
            )
    prediction_frame = pd.DataFrame(rows)
    metrics = {}
    for split_name in ("validation", "test"):
        split_predictions = prediction_frame[prediction_frame["split"] == split_name]
        probs = split_predictions[[f"prob_{name}" for name in PHASE_CLASSES]].to_numpy()
        metrics[split_name] = _phase_metrics(
            split_predictions["label"].astype(int).tolist(),
            split_predictions["predicted_label"].astype(int).tolist(),
            probs,
        )
    return ExpandingPredictionResult(predictions=prediction_frame, metrics_by_split=metrics)


def _phase_markov_predictions(panel: pd.DataFrame, horizon: int) -> ExpandingPredictionResult:
    target_col = f"phase_t_plus_{horizon}m_int"
    rows: list[dict[str, object]] = []
    for split_name in ("validation", "test"):
        split_frame = _expanding_prediction_rows(panel, split_name)
        train_frame = _split_training_rows(panel, split_name)
        train_frame = train_frame[train_frame["current_phase_int"].notna() & train_frame[target_col].notna()].copy()
        if train_frame.empty:
            continue
        counts = np.ones((len(PHASE_CLASSES), len(PHASE_CLASSES)), dtype=float)
        for _, train_row in train_frame.iterrows():
            counts[int(train_row["current_phase_int"]), int(train_row[target_col])] += 1.0
        for _, row in split_frame.iterrows():
            if pd.isna(row["current_phase_int"]) or pd.isna(row[target_col]):
                continue
            current_state = int(row["current_phase_int"])
            probabilities = counts[current_state] / counts[current_state].sum()
            predicted = int(np.argmax(probabilities))
            rows.append(
                {
                    "date": row["date"],
                    "split": split_name,
                    "label": int(row[target_col]),
                    "predicted_label": predicted,
                    **{f"prob_{name}": float(probabilities[idx]) for idx, name in enumerate(PHASE_CLASSES)},
                }
            )
    prediction_frame = pd.DataFrame(rows)
    metrics = {}
    for split_name in ("validation", "test"):
        split_predictions = prediction_frame[prediction_frame["split"] == split_name]
        probs = split_predictions[[f"prob_{name}" for name in PHASE_CLASSES]].to_numpy()
        metrics[split_name] = _phase_metrics(
            split_predictions["label"].astype(int).tolist(),
            split_predictions["predicted_label"].astype(int).tolist(),
            probs,
        )
    return ExpandingPredictionResult(predictions=prediction_frame, metrics_by_split=metrics)


def _expanding_ml_predictions(panel: pd.DataFrame, feature_cols: list[str], target_col: str, model_name: str) -> ExpandingPredictionResult:
    rows: list[dict[str, object]] = []
    for split_name in ("validation", "test"):
        split_frame = _expanding_prediction_rows(panel, split_name)
        train_frame = _split_training_rows(panel, split_name)
        train_frame = train_frame[train_frame[target_col].notna()].copy()
        split_frame = split_frame[split_frame[target_col].notna()].copy()
        usable_features = _usable_feature_columns(train_frame, feature_cols)
        if train_frame.shape[0] < 60 or split_frame.empty or not usable_features:
            continue
        estimator = _make_multiclass_model(model_name)
        estimator.fit(train_frame[usable_features], train_frame[target_col].astype(int))
        probabilities_matrix = estimator.predict_proba(split_frame[usable_features])
        predicted_labels = probabilities_matrix.argmax(axis=1)
        for row_index, (_, row) in enumerate(split_frame.iterrows()):
            probabilities = probabilities_matrix[row_index]
            predicted = int(predicted_labels[row_index])
            rows.append(
                {
                    "date": row["date"],
                    "split": split_name,
                    "label": int(row[target_col]),
                    "predicted_label": predicted,
                    **{f"prob_{name}": float(probabilities[idx]) for idx, name in enumerate(PHASE_CLASSES)},
                }
            )
    prediction_frame = pd.DataFrame(rows)
    metrics = {}
    for split_name in ("validation", "test"):
        split_predictions = prediction_frame[prediction_frame["split"] == split_name]
        probs = split_predictions[[f"prob_{name}" for name in PHASE_CLASSES]].to_numpy()
        metrics[split_name] = _phase_metrics(
            split_predictions["label"].astype(int).tolist(),
            split_predictions["predicted_label"].astype(int).tolist(),
            probs,
        )
    return ExpandingPredictionResult(predictions=prediction_frame, metrics_by_split=metrics)


def _plot_confusion(y_true: list[int], y_pred: list[int], path, title: str) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(PHASE_CLASSES))))
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(range(len(PHASE_CLASSES)))
    ax.set_xticklabels(PHASE_CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(PHASE_CLASSES)))
    ax.set_yticklabels(PHASE_CLASSES)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def train_phase_forecasts() -> dict[str, object]:
    panel = _load_labeled_panel()
    feature_cols = _feature_columns(panel)
    phase_model_names = ["phase_persistence", "phase_markov_baseline", "phase_logreg", "phase_xgboost"]
    results: dict[str, dict[str, object]] = {name: {} for name in phase_model_names}
    prediction_rows: list[pd.DataFrame] = []
    validation_ranking_rows: list[dict[str, object]] = []

    for horizon in PHASE_HORIZONS:
        target_col = f"phase_t_plus_{horizon}m_int"
        horizon_panel = panel[panel["current_phase_int"].notna() & panel[target_col].notna()].copy()
        persistence_result = _phase_persistence_predictions(horizon_panel, horizon=horizon)
        markov_result = _phase_markov_predictions(horizon_panel, horizon=horizon)
        logreg_result = _expanding_ml_predictions(horizon_panel, feature_cols, target_col, "phase_logreg")
        xgb_result = _expanding_ml_predictions(horizon_panel, feature_cols, target_col, "phase_xgboost")
        horizon_results = {
            "phase_persistence": persistence_result,
            "phase_markov_baseline": markov_result,
            "phase_logreg": logreg_result,
            "phase_xgboost": xgb_result,
        }
        for model_name, result in horizon_results.items():
            results[model_name][f"horizon_{horizon}m"] = result.metrics_by_split
            frame = result.predictions.copy()
            frame["model"] = model_name
            frame["horizon_months"] = horizon
            prediction_rows.append(frame)
            validation_metrics = result.metrics_by_split["validation"]
            validation_ranking_rows.append(
                {
                    "model": model_name,
                    "horizon_months": horizon,
                    "macro_f1": float(validation_metrics["macro_f1"]),
                    "balanced_accuracy": float(validation_metrics["balanced_accuracy"]),
                }
            )

    ranking_frame = pd.DataFrame(validation_ranking_rows)
    selection = (
        ranking_frame.groupby("model", as_index=False)[["macro_f1", "balanced_accuracy"]]
        .mean()
        .sort_values(["macro_f1", "balanced_accuracy"], ascending=False)
        .reset_index(drop=True)
    )
    selected_model = str(selection.iloc[0]["model"])
    predictions = pd.concat(prediction_rows, ignore_index=True)
    predictions.to_csv(PHASE_FORECAST_PREDICTIONS_CSV, index=False)

    for horizon in PHASE_HORIZONS:
        test_predictions = predictions[
            (predictions["model"] == selected_model)
            & (predictions["horizon_months"] == horizon)
            & (predictions["split"] == "test")
        ].copy()
        if test_predictions.empty:
            continue
        _plot_confusion(
            test_predictions["label"].astype(int).tolist(),
            test_predictions["predicted_label"].astype(int).tolist(),
            PHASE_CONFUSION_DIR / f"{selected_model}_h{horizon}m_test_confusion.png",
            f"{selected_model} test confusion, horizon {horizon}m",
        )

    full_train = panel.copy()
    selected_bundle: dict[str, object] = {"model_name": selected_model, "feature_cols": feature_cols, "horizons": {}}
    for horizon in PHASE_HORIZONS:
        target_col = f"phase_t_plus_{horizon}m_int"
        train_frame = full_train[full_train[target_col].notna()].copy()
        usable_features = _usable_feature_columns(train_frame, feature_cols)
        if selected_model == "phase_logreg":
            estimator = _make_multiclass_model("phase_logreg")
            estimator.fit(train_frame[usable_features], train_frame[target_col].astype(int))
            selected_bundle["horizons"][horizon] = {"estimator": estimator, "feature_cols": usable_features}
        elif selected_model == "phase_xgboost":
            estimator = _make_multiclass_model("phase_xgboost")
            estimator.fit(train_frame[usable_features], train_frame[target_col].astype(int))
            selected_bundle["horizons"][horizon] = {"estimator": estimator, "feature_cols": usable_features}
        elif selected_model == "phase_markov_baseline":
            counts = np.ones((len(PHASE_CLASSES), len(PHASE_CLASSES)), dtype=float)
            for _, train_row in train_frame[train_frame["current_phase_int"].notna()].iterrows():
                counts[int(train_row["current_phase_int"]), int(train_row[target_col])] += 1.0
            selected_bundle["horizons"][horizon] = counts
        else:
            selected_bundle["horizons"][horizon] = {"baseline": "persistence"}
    joblib.dump(selected_bundle, PHASE_MODEL_PATH)

    summary = {
        "selected_model": selected_model,
        "feature_count": len(feature_cols),
        "validation_selection": selection.to_dict(orient="records"),
        "models": results,
    }
    save_json(PHASE_FORECAST_METRICS_JSON, summary)

    csv_rows = []
    for model_name, horizon_block in results.items():
        for horizon_name, split_block in horizon_block.items():
            for split_name, metrics in split_block.items():
                csv_rows.append({"model": model_name, "horizon": horizon_name, "split": split_name, **metrics})
    pd.DataFrame(csv_rows).to_csv(PHASE_FORECAST_METRICS_CSV, index=False)
    return summary


def _expanding_binary_predictions(panel: pd.DataFrame, feature_cols: list[str], target_col: str, model_name: str) -> ExpandingPredictionResult:
    rows: list[dict[str, object]] = []
    for split_name in ("validation", "test"):
        split_frame = _expanding_prediction_rows(panel, split_name)
        train_frame = _split_training_rows(panel, split_name)
        train_frame = train_frame[train_frame[target_col].notna()].copy()
        split_frame = split_frame[split_frame[target_col].notna()].copy()
        usable_features = _usable_feature_columns(train_frame, feature_cols)
        if train_frame.shape[0] < 60 or split_frame.empty or not usable_features:
            continue
        estimator = _make_binary_model(model_name)
        estimator.fit(train_frame[usable_features], train_frame[target_col].astype(int))
        probabilities = estimator.predict_proba(split_frame[usable_features])[:, 1]
        for row_index, (_, row) in enumerate(split_frame.iterrows()):
            probability = float(probabilities[row_index])
            rows.append(
                {
                    "date": row["date"],
                    "split": split_name,
                    "label": int(row[target_col]),
                    "probability": probability,
                }
            )
    prediction_frame = pd.DataFrame(rows)
    metrics = {}
    for split_name in ("validation", "test"):
        split_predictions = prediction_frame[prediction_frame["split"] == split_name]
        metrics[split_name] = _binary_metrics(
            split_predictions["label"].astype(int).tolist(),
            split_predictions["probability"].astype(float).tolist(),
        )
    return ExpandingPredictionResult(predictions=prediction_frame, metrics_by_split=metrics)


def _mean_lead_months(prediction_frame: pd.DataFrame, label_panel: pd.DataFrame, threshold: float = 0.5) -> float:
    alerts = prediction_frame[prediction_frame["probability"] >= threshold].copy()
    if alerts.empty:
        return float("nan")
    starts = label_panel[label_panel["recession_start"] == 1]["date"].sort_values().tolist()
    lead_values: list[int] = []
    for start_date in starts:
        prior_alerts = alerts[(alerts["date"] < start_date) & (alerts["date"] >= start_date - pd.DateOffset(months=6))]
        if prior_alerts.empty:
            continue
        first_alert = prior_alerts["date"].min()
        lead = (start_date.to_period("M") - first_alert.to_period("M")).n
        lead_values.append(int(lead))
    return float(np.mean(lead_values)) if lead_values else float("nan")


def _plot_binary_curves(predictions: pd.DataFrame) -> None:
    from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

    test_predictions = predictions[predictions["split"] == "test"].copy()
    if test_predictions.empty:
        return
    y_true = test_predictions["label"].astype(int).to_numpy()
    y_score = test_predictions["probability"].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(5, 4))
    PrecisionRecallDisplay.from_predictions(y_true, y_score, ax=ax)
    ax.set_title("Recession risk precision-recall, test")
    fig.tight_layout()
    fig.savefig(RECESSION_PR_CURVE_FIGURE, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_true, y_score, ax=ax)
    ax.set_title("Recession risk ROC, test")
    fig.tight_layout()
    fig.savefig(RECESSION_ROC_CURVE_FIGURE, dpi=180)
    plt.close(fig)


def train_recession_risk() -> dict[str, object]:
    panel = _load_labeled_panel()
    feature_cols = _feature_columns(panel)
    target_col = "recession_within_6m"
    model_specs = {
        "yield_curve_probit": ["term_spread"],
        "recession_logreg": feature_cols,
        "recession_xgboost": feature_cols,
    }
    results: dict[str, object] = {}
    prediction_rows: list[pd.DataFrame] = []
    selection_rows: list[dict[str, object]] = []

    for model_name, model_features in model_specs.items():
        result = _expanding_binary_predictions(panel, model_features, target_col, model_name)
        results[model_name] = result.metrics_by_split
        frame = result.predictions.copy()
        frame["model"] = model_name
        prediction_rows.append(frame)
        validation_metrics = result.metrics_by_split["validation"]
        selection_rows.append(
            {
                "model": model_name,
                "pr_auc": float(validation_metrics["pr_auc"]),
                "brier_score": float(validation_metrics["brier_score"]),
            }
        )

    selection_frame = pd.DataFrame(selection_rows).sort_values(["pr_auc", "brier_score"], ascending=[False, True]).reset_index(drop=True)
    selected_model = str(selection_frame.iloc[0]["model"])
    predictions = pd.concat(prediction_rows, ignore_index=True)
    selected_predictions = predictions[predictions["model"] == selected_model].copy()

    validation_selected = selected_predictions[selected_predictions["split"] == "validation"].copy()
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(validation_selected["probability"].astype(float), validation_selected["label"].astype(int))
    selected_predictions["calibrated_probability"] = calibrator.transform(selected_predictions["probability"].astype(float))

    calibrated_metrics = {}
    for split_name in ("validation", "test"):
        split_predictions = selected_predictions[selected_predictions["split"] == split_name].copy()
        metrics = _binary_metrics(
            split_predictions["label"].astype(int).tolist(),
            split_predictions["calibrated_probability"].astype(float).tolist(),
        )
        if split_name == "validation":
            split_panel = panel[
                (panel["date"] > pd.Timestamp(WARMUP_TRAIN_END))
                & (panel["date"] <= pd.Timestamp(VALIDATION_END))
            ].copy()
        else:
            split_panel = panel[panel["date"] > pd.Timestamp(VALIDATION_END)].copy()
        lead_frame = split_predictions[["date", "split", "label", "calibrated_probability"]].rename(
            columns={"calibrated_probability": "probability"}
        )
        metrics["mean_lead_months_at_0.50"] = _mean_lead_months(
            lead_frame,
            split_panel,
            threshold=0.50,
        )
        calibrated_metrics[split_name] = metrics

    predictions = predictions.merge(
        selected_predictions[["date", "split", "model", "calibrated_probability"]],
        on=["date", "split", "model"],
        how="left",
    )
    predictions.to_csv(RECESSION_RISK_PREDICTIONS_CSV, index=False)

    final_feature_cols = model_specs[selected_model]
    labeled_train = panel[panel[target_col].notna()].copy()
    usable_features = _usable_feature_columns(labeled_train, final_feature_cols)
    final_estimator = _make_binary_model(selected_model)
    final_estimator.fit(labeled_train[usable_features], labeled_train[target_col].astype(int))
    bundle = {
        "model_name": selected_model,
        "feature_cols": usable_features,
        "estimator": final_estimator,
        "calibrator": calibrator,
    }
    joblib.dump(bundle, RECESSION_MODEL_PATH)

    selected_test_predictions = selected_predictions[selected_predictions["split"] == "test"].copy()
    selected_test_predictions["probability"] = selected_test_predictions["calibrated_probability"]
    _plot_binary_curves(selected_test_predictions[["date", "split", "label", "probability"]])

    summary = {
        "selected_model": selected_model,
        "feature_count": len(usable_features),
        "validation_selection": selection_frame.to_dict(orient="records"),
        "models": results,
        "selected_calibrated": calibrated_metrics,
    }
    save_json(RECESSION_RISK_METRICS_JSON, summary)

    csv_rows = []
    for model_name, split_block in results.items():
        for split_name, metrics in split_block.items():
            csv_rows.append({"model": model_name, "split": split_name, **metrics})
    for split_name, metrics in calibrated_metrics.items():
        csv_rows.append({"model": f"{selected_model}_calibrated", "split": split_name, **metrics})
    pd.DataFrame(csv_rows).to_csv(RECESSION_RISK_METRICS_CSV, index=False)
    return summary
