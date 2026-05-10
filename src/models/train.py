"""Training pipeline matching notebook 6's final approach.

Key differences from the previous version:
- Uses ``log1p(y)`` target transform (notebook 6's key trick)
- Uses notebook 6's exact XGB hyperparameters as defaults
- Generates ``region_inference_stats.pkl`` after training
- Feature engineering delegated to ``src.features.feature_engineering``
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.data.load_data import load_or_build_train_test, resolve_project_paths
from src.features.feature_engineering import prepare_train_test_features
from src.inference.realtime import build_region_stats
from src.utils.metrics import evaluate_metrics

logger = logging.getLogger(__name__)

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    XGBRegressor = None


@dataclass
class TrainingOutputs:
    """Container for training run outputs."""

    model: Any
    metrics_summary: pd.DataFrame
    model_path: Path
    stats_path: Path
    feature_cols: list[str]
    region_smooth_map: pd.Series


# Notebook 6's final XGB hyperparameters
DEFAULT_XGB_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.1026,
    "max_depth": 6,
    "subsample": 0.8295,
    "colsample_bytree": 0.9024,
    "reg_alpha": 1.2257,
    "reg_lambda": 0.8597,
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",
}


def train_model(
    project_root: Path | None = None,
    xgb_params: dict | None = None,
    force_rebuild_split: bool = True,
) -> TrainingOutputs:
    """Train XGBoost model with log1p target transform.

    Reproduces notebook 6's final training cell exactly:
    1. Load historical features → train/test split
    2. Engineer features (lags, calendar, extras)
    3. Train XGB on log1p(y_train)
    4. Evaluate with expm1(predictions)
    5. Save model + region inference stats

    Parameters
    ----------
    project_root : Path, optional
        Project root directory.
    xgb_params : dict, optional
        XGBoost hyperparameters. Uses notebook 6's tuned values if None.
    force_rebuild_split : bool
        Whether to rebuild train/test from historical data.

    Returns
    -------
    TrainingOutputs
        Trained model and metadata.
    """
    if not HAS_XGB:
        raise ImportError("xgboost is required for training")

    paths = resolve_project_paths(project_root)
    params = xgb_params or DEFAULT_XGB_PARAMS

    # 1. Load data
    train_df, test_df = load_or_build_train_test(
        paths, force_rebuild_split=force_rebuild_split
    )
    logger.info("Train: %s, Test: %s", train_df.shape, test_df.shape)

    # 2. Feature engineering
    features = prepare_train_test_features(train_df, test_df)
    X_train = features["X_train"]
    y_train = features["y_train"]
    X_test = features["X_test"]
    y_test = features["y_test"]

    logger.info("Features: %d, X_train: %s", len(features["feature_cols"]), X_train.shape)

    # 3. Train with log1p transform (notebook 6's approach)
    model = XGBRegressor(**params)
    y_train_log = np.log1p(y_train)
    model.fit(X_train, y_train_log)

    # 4. Evaluate
    y_pred_train = np.clip(np.expm1(model.predict(X_train)), 0, None)
    y_pred_test = np.clip(np.expm1(model.predict(X_test)), 0, None)

    train_metrics = evaluate_metrics(y_train, y_pred_train)
    test_metrics = evaluate_metrics(y_test, y_pred_test)
    metrics_summary = pd.DataFrame([
        {"split": "train", **train_metrics},
        {"split": "test", **test_metrics},
    ])

    logger.info("Train MAPE: %.4f, Test MAPE: %.4f",
                train_metrics["MAPE"], test_metrics["MAPE"])

    # 5. Save model
    model_path = paths.models_dir / "xgb_model.pkl"
    joblib.dump(model, model_path)
    metrics_summary.to_csv(paths.reports_dir / "metrics_summary.csv", index=False)

    # 6. Build inference stats
    data_path = next(
        (p for p in [
            paths.data_interim / "final_data.csv",
            paths.data_interim / "historical_features.csv",
        ] if p.exists()),
        None,
    )

    stats_path = paths.models_dir / "region_inference_stats.pkl"
    if data_path:
        build_region_stats(data_path=data_path, output_path=stats_path)
        logger.info("Inference stats saved to %s", stats_path)
    else:
        logger.warning("No data file found to build inference stats")

    return TrainingOutputs(
        model=model,
        metrics_summary=metrics_summary,
        model_path=model_path,
        stats_path=stats_path,
        feature_cols=features["feature_cols"],
        region_smooth_map=features["region_smooth_map"],
    )
