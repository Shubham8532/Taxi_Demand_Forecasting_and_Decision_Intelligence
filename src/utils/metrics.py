"""Evaluation metrics used in training and model selection.

Matches the final metric definitions from notebook 6 (Training_model),
including the ``safe_mape`` with floor=10 to stabilise MAPE on
zero-heavy 15-minute demand bins.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def smape(y_true, y_pred) -> float:
    """Symmetric Mean Absolute Percentage Error.

    Parameters
    ----------
    y_true : array-like
        Ground truth values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    float
        sMAPE in [0, 2] range.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = np.abs(y_true) + np.abs(y_pred)
    return float(
        np.mean(2.0 * np.abs(y_true - y_pred) / np.where(denom == 0, 1.0, denom))
    )


def safe_mape(y_true, y_pred, floor: float = 10.0) -> float:
    """MAPE with a denominator floor to avoid explosion on near-zero targets.

    This matches notebook 6's approach: ``y_true = np.maximum(y_true, 10)``
    before computing standard MAPE.

    Parameters
    ----------
    y_true : array-like
        Ground truth values.
    y_pred : array-like
        Predicted values.
    floor : float, default 10.0
        Minimum value applied to ``|y_true|`` in the denominator.

    Returns
    -------
    float
        Stabilised MAPE value.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(
        np.mean(np.abs(y_true - y_pred) / np.maximum(np.abs(y_true), floor))
    )


def evaluate_metrics(y_true, y_pred) -> dict[str, float]:
    """Return a consistent dictionary of regression metrics.

    Used across training validation, Optuna objective, and final
    test-set evaluation.

    Parameters
    ----------
    y_true : array-like
        Ground truth values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    dict[str, float]
        Keys: MAE, RMSE, MAPE, sMAPE, R2.
    """
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE": safe_mape(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
    }
