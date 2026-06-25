"""Regression and physics-aware metrics for PASS surrogate evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .physics import evaluate_pass_batch
from .schema import PASS_OUTPUT_COLUMNS, PASS_POSITION_COLUMNS, PASS_POWER_COLUMNS, SystemConfig


@dataclass(frozen=True)
class RegressionReport:
    """Regression metric summary."""

    overall: dict[str, float]
    per_output: pd.DataFrame


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot < 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def regression_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    column_names: Sequence[str],
    tolerance: float = 0.10,
) -> RegressionReport:
    """Build common regression metrics for tabular outputs."""

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError("Regression targets and predictions must have the same shape.")
    if y_true.ndim != 2:
        raise ValueError("Regression inputs must be 2D matrices.")

    diff = y_pred - y_true
    abs_diff = np.abs(diff)
    sq_diff = diff ** 2

    rows = []
    for idx, name in enumerate(column_names):
        rows.append(
            {
                "output": name,
                "mae": float(abs_diff[:, idx].mean()),
                "rmse": float(np.sqrt(sq_diff[:, idx].mean())),
                "r2": float(_safe_r2(y_true[:, idx], y_pred[:, idx])),
                "max_abs_error": float(abs_diff[:, idx].max()),
            }
        )

    per_output = pd.DataFrame(rows)
    overall = {
        "mae": float(abs_diff.mean()),
        "rmse": float(np.sqrt(sq_diff.mean())),
        "r2": float(_safe_r2(y_true.reshape(-1), y_pred.reshape(-1))),
        "max_abs_error": float(abs_diff.max()),
        "tolerance": float(tolerance),
        "within_tolerance_rate": float(np.mean(abs_diff <= tolerance)),
    }
    if np.isclose(tolerance, 0.10):
        overall["within_0p10_rate"] = overall["within_tolerance_rate"]
    return RegressionReport(overall=overall, per_output=per_output)


@dataclass(frozen=True)
class PassSystemReport:
    """End-to-end physics-aware evaluation summary."""

    overall: dict[str, float]
    per_user: pd.DataFrame


def pass_system_report(
    inputs: np.ndarray,
    true_outputs: np.ndarray,
    pred_outputs: np.ndarray,
    config: SystemConfig,
) -> PassSystemReport:
    """Compare predicted PASS configurations against the ground-truth MATLAB labels."""

    true_eval = evaluate_pass_batch(inputs, true_outputs, config)
    pred_eval = evaluate_pass_batch(inputs, pred_outputs, config)

    rows = []
    for user_idx in range(config.num_users):
        rows.append(
            {
                "user": f"user{user_idx + 1}",
                "true_rate_mean": float(true_eval.rates[:, user_idx].mean()),
                "pred_rate_mean": float(pred_eval.rates[:, user_idx].mean()),
                "true_margin_mean": float(true_eval.qos_margin[:, user_idx].mean()),
                "pred_margin_mean": float(pred_eval.qos_margin[:, user_idx].mean()),
                "pred_rate_mae": float(np.abs(pred_eval.rates[:, user_idx] - true_eval.rates[:, user_idx]).mean()),
            }
        )

    per_user = pd.DataFrame(rows)
    overall = {
        "true_sum_rate_mean": float(true_eval.sum_rate.mean()),
        "pred_sum_rate_mean": float(pred_eval.sum_rate.mean()),
        "sum_rate_mae": float(np.abs(pred_eval.sum_rate - true_eval.sum_rate).mean()),
        "true_ee_mean": float(true_eval.energy_efficiency.mean()),
        "pred_ee_mean": float(pred_eval.energy_efficiency.mean()),
        "ee_mae": float(np.abs(pred_eval.energy_efficiency - true_eval.energy_efficiency).mean()),
        "qos_satisfaction_rate_pred": float(pred_eval.qos_satisfied.mean()),
        "qos_satisfaction_rate_true": float(true_eval.qos_satisfied.mean()),
        "mean_qos_margin_pred": float(pred_eval.qos_margin.mean()),
        "mean_qos_margin_true": float(true_eval.qos_margin.mean()),
    }
    return PassSystemReport(overall=overall, per_user=per_user)


def classification_like_report(
    reference_labels: np.ndarray,
    predicted_labels: np.ndarray,
) -> dict[str, float]:
    """Binary classification scores when you do have positive/negative labels."""

    reference_labels = np.asarray(reference_labels).astype(bool)
    predicted_labels = np.asarray(predicted_labels).astype(bool)
    if reference_labels.shape != predicted_labels.shape:
        raise ValueError("Classification label arrays must have the same shape.")

    tp = float(np.sum(reference_labels & predicted_labels))
    tn = float(np.sum(~reference_labels & ~predicted_labels))
    fp = float(np.sum(~reference_labels & predicted_labels))
    fn = float(np.sum(reference_labels & ~predicted_labels))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1.0)
    balanced_accuracy = 0.5 * (
        (tp / max(tp + fn, 1.0)) +
        (tn / max(tn + fp, 1.0))
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "balanced_accuracy": float(balanced_accuracy),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }
