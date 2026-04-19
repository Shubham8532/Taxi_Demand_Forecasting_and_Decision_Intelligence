"""Evaluation metrics used in training and model selection."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def smape(y_true, y_pred) -> float:
    """Symmetric MAPE."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.abs(y_true) + np.abs(y_pred)
    return float(np.mean(2.0 * np.abs(y_true - y_pred) / np.where(denom == 0, 1.0, denom)))


def safe_mape(y_true, y_pred, eps: float = 1.0) -> float:
    """MAPE with denominator clipping for zero-heavy demand targets."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred) / np.maximum(np.abs(y_true), eps)))


def evaluate_metrics(y_true, y_pred) -> dict[str, float]:
    """Return consistent regression metrics for all train/valid/test evaluations."""
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE": safe_mape(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
    }

