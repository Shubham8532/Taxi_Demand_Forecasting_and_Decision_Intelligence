"""High-level pipeline entry points for training and inference.

Usage:
    from src.pipeline import train_model
    outputs = train_model()

    # After training, the model + stats are saved.
    # Deploy app.py which uses RealtimePredictor automatically.
"""

from __future__ import annotations

from pathlib import Path

from src.models.train import TrainingOutputs, train_model as _train_model


def train_model(
    project_root: Path | None = None,
    xgb_params: dict | None = None,
    force_rebuild_split: bool = True,
) -> TrainingOutputs:
    """Train the demand forecasting model and generate deployment artifacts.

    This is the main entry point for the training pipeline.
    After completion, both ``xgb_model.pkl`` and
    ``region_inference_stats.pkl`` will be in ``models/``.

    Parameters
    ----------
    project_root : Path, optional
    xgb_params : dict, optional
        XGBoost hyperparameters override.
    force_rebuild_split : bool
        Rebuild train/test from historical data.

    Returns
    -------
    TrainingOutputs
    """
    return _train_model(
        project_root=project_root,
        xgb_params=xgb_params,
        force_rebuild_split=force_rebuild_split,
    )
