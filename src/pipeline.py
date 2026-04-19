"""High-level pipeline entry points for training and inference."""

from __future__ import annotations

from pathlib import Path

from src.models.predict import PredictionOutputs, predict_with_saved_model
from src.models.train import TrainingOutputs, train_model as _train_model


def train_model(
    project_root: Path | None = None,
    n_trials: int = 80,
    random_state: int = 42,
    use_mlflow: bool = False,
    force_rebuild_split: bool = True,
) -> TrainingOutputs:
    """Train model with Optuna selection and save model/report artifacts."""
    return _train_model(
        project_root=project_root,
        n_trials=n_trials,
        random_state=random_state,
        use_mlflow=use_mlflow,
        force_rebuild_split=force_rebuild_split,
    )


def predict(
    project_root: Path | None = None,
    model_path: Path | None = None,
    force_rebuild_split: bool = True,
) -> PredictionOutputs:
    """Run inference using saved model and save business-ready outputs."""
    return predict_with_saved_model(
        project_root=project_root,
        model_path=model_path,
        force_rebuild_split=force_rebuild_split,
    )

