"""Training and Optuna model selection pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.data.load_data import load_or_build_train_test, resolve_project_paths
from src.features.feature_engineering import prepare_model_features, time_based_validation_split
from src.utils.metrics import evaluate_metrics

logger = logging.getLogger(__name__)

try:
    import optuna
except Exception:  # pragma: no cover - optional dependency
    optuna = None

try:
    from xgboost import XGBRegressor

    HAS_XGB = True
except Exception:  # pragma: no cover - optional dependency
    HAS_XGB = False
    XGBRegressor = None

try:
    import mlflow

    HAS_MLFLOW = True
except Exception:  # pragma: no cover - optional dependency
    HAS_MLFLOW = False
    mlflow = None


@dataclass
class TrainingOutputs:
    """Container for train run outputs."""

    best_model_name: str
    best_params: dict[str, Any]
    best_value: float
    pipeline: Pipeline
    metrics_summary: pd.DataFrame
    leaderboard: pd.DataFrame
    predictions: pd.DataFrame
    model_path: Path
    leaderboard_path: Path
    metrics_path: Path
    predictions_path: Path


def build_preprocessor(cat_cols: list[str], num_cols: list[str]) -> ColumnTransformer:
    """Build model preprocessor from categorical/numeric columns."""
    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformers = []
    if cat_cols:
        transformers.append(("cat", categorical_pipeline, cat_cols))
    if num_cols:
        transformers.append(("num", numeric_pipeline, num_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def make_model_from_trial(trial, random_state: int = 42):
    """Model factory with same search space as notebook."""
    options = ["LR", "RIDGE", "RF", "GBR"]
    if HAS_XGB:
        options.append("XGBR")
    model_name = trial.suggest_categorical("model_name", options)

    if model_name == "LR":
        model = LinearRegression()
    elif model_name == "RIDGE":
        alpha = trial.suggest_float("ridge_alpha", 0.1, 200.0, log=True)
        model = Ridge(alpha=alpha, random_state=random_state)
    elif model_name == "RF":
        n_estimators = trial.suggest_int("rf_n_estimators", 100, 400, step=50)
        max_depth = trial.suggest_int("rf_max_depth", 5, 24)
        min_samples_leaf = trial.suggest_int("rf_min_samples_leaf", 1, 8)
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_name == "GBR":
        n_estimators = trial.suggest_int("gbr_n_estimators", 80, 400, step=40)
        learning_rate = trial.suggest_float("gbr_learning_rate", 0.01, 0.2, log=True)
        max_depth = trial.suggest_int("gbr_max_depth", 2, 8)
        subsample = trial.suggest_float("gbr_subsample", 0.6, 1.0)
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            random_state=random_state,
        )
    else:
        n_estimators = trial.suggest_int("xgb_n_estimators", 120, 500, step=40)
        learning_rate = trial.suggest_float("xgb_learning_rate", 0.01, 0.2, log=True)
        max_depth = trial.suggest_int("xgb_max_depth", 3, 10)
        subsample = trial.suggest_float("xgb_subsample", 0.6, 1.0)
        colsample_bytree = trial.suggest_float("xgb_colsample", 0.6, 1.0)
        model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        )
    return model_name, model


def _build_objective(
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    preprocessor: ColumnTransformer,
    random_state: int,
    use_mlflow: bool,
):
    def objective(trial):
        model_name, model = make_model_from_trial(trial, random_state=random_state)
        pipeline = Pipeline([("prep", preprocessor), ("model", model)])

        sample_size = min(50_000, len(X_fit))
        if sample_size < len(X_fit):
            rng = np.random.default_rng(random_state + trial.number)
            idx = rng.choice(len(X_fit), size=sample_size, replace=False)
            X_fit_sample = X_fit.iloc[idx]
            y_fit_sample = y_fit.iloc[idx]
        else:
            X_fit_sample = X_fit
            y_fit_sample = y_fit

        if use_mlflow and HAS_MLFLOW:
            mlflow.start_run(nested=True)
            mlflow.log_param("model_name", model_name)

        pipeline.fit(X_fit_sample, y_fit_sample)
        y_pred_valid = pipeline.predict(X_valid)
        metrics = evaluate_metrics(y_valid, y_pred_valid)

        if metrics["MAPE"] < 1e-3:
            raise ValueError(f"Leakage suspected: unrealistically low validation MAPE={metrics['MAPE']:.3e}")

        if use_mlflow and HAS_MLFLOW:
            for k, v in metrics.items():
                mlflow.log_metric(f"valid_{k}", v)
            mlflow.log_params(model.get_params())
            mlflow.end_run()

        return metrics["MAPE"]

    return objective


def run_optuna(
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    preprocessor: ColumnTransformer,
    n_trials: int = 80,
    random_state: int = 42,
    use_mlflow: bool = False,
) -> tuple[Any, pd.DataFrame]:
    """Run Optuna model search and return study + leaderboard."""
    if optuna is None:
        raise ImportError("optuna is not installed. Add it to your environment to run model search.")

    objective = _build_objective(
        X_fit=X_fit,
        y_fit=y_fit,
        X_valid=X_valid,
        y_valid=y_valid,
        preprocessor=preprocessor,
        random_state=random_state,
        use_mlflow=use_mlflow,
    )

    study = optuna.create_study(
        study_name="model_selection",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=random_state, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
    )

    if use_mlflow and HAS_MLFLOW:
        with mlflow.start_run(run_name="model_selection_optuna"):
            study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True, catch=(ValueError,))
            mlflow.log_params(study.best_params)
            mlflow.log_metric("best_valid_MAPE", study.best_value)
    else:
        study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True, catch=(ValueError,))

    trials_df = study.trials_dataframe()
    leaderboard_cols = ["number", "value", "params_model_name", "state"]
    leaderboard = trials_df[[c for c in leaderboard_cols if c in trials_df.columns]].copy()
    leaderboard = leaderboard.rename(columns={"value": "valid_MAPE"}).sort_values("valid_MAPE").reset_index(drop=True)
    return study, leaderboard


class _DummyTrial:
    """Shim to rebuild model from Optuna best params."""

    def __init__(self, params: dict[str, Any]):
        self.params = params

    def suggest_categorical(self, name, choices):
        return self.params[name]

    def suggest_int(self, name, low, high, step=1):
        return int(self.params[name])

    def suggest_float(self, name, low, high, log=False):
        return float(self.params[name])


def train_model(
    project_root: Path | None = None,
    n_trials: int = 80,
    random_state: int = 42,
    use_mlflow: bool = False,
    force_rebuild_split: bool = True,
) -> TrainingOutputs:
    """Main training pipeline: load data -> engineer features -> Optuna -> train -> save artifacts."""
    paths = resolve_project_paths(project_root)
    train_df, test_df = load_or_build_train_test(paths, force_rebuild_split=force_rebuild_split)

    features = prepare_model_features(train_df=train_df, test_df=test_df)
    X_fit, y_fit, X_valid, y_valid = time_based_validation_split(
        train_df=features.train_df,
        X_train=features.X_train,
        y_train=features.y_train,
        time_col=features.time_col,
        valid_ratio=0.2,
    )

    preprocessor = build_preprocessor(features.cat_cols, features.num_cols)
    study, leaderboard = run_optuna(
        X_fit=X_fit,
        y_fit=y_fit,
        X_valid=X_valid,
        y_valid=y_valid,
        preprocessor=preprocessor,
        n_trials=n_trials,
        random_state=random_state,
        use_mlflow=use_mlflow,
    )

    best_trial = _DummyTrial(study.best_params)
    best_model_name, best_model = make_model_from_trial(best_trial, random_state=random_state)
    best_pipeline = Pipeline([("prep", preprocessor), ("model", best_model)])
    best_pipeline.fit(features.X_train, features.y_train)

    y_pred_train = np.clip(best_pipeline.predict(features.X_train), a_min=0, a_max=None)
    y_pred_test = np.clip(best_pipeline.predict(features.X_test), a_min=0, a_max=None)

    train_metrics = evaluate_metrics(features.y_train, y_pred_train)
    test_metrics = evaluate_metrics(features.y_test, y_pred_test)
    metrics_summary = pd.DataFrame([{"split": "train", **train_metrics}, {"split": "test", **test_metrics}])

    predictions = pd.DataFrame(
        {"actual_demand": features.y_test.values, "predicted_demand": y_pred_test},
        index=features.y_test.index,
    ).reset_index(drop=True)

    model_path = paths.models_dir / "best_model_selection_pipeline.joblib"
    leaderboard_path = paths.reports_dir / "model_selection_leaderboard.csv"
    metrics_path = paths.reports_dir / "model_metrics_summary.csv"
    predictions_path = paths.data_interim / "model_test_predictions.csv"

    joblib.dump(best_pipeline, model_path)
    leaderboard.to_csv(leaderboard_path, index=False)
    metrics_summary.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)

    logger.info(
        "Training complete. model=%s best_valid_mape=%.6f test_mape=%.6f",
        best_model_name,
        float(study.best_value),
        float(test_metrics["MAPE"]),
    )

    return TrainingOutputs(
        best_model_name=best_model_name,
        best_params=study.best_params,
        best_value=float(study.best_value),
        pipeline=best_pipeline,
        metrics_summary=metrics_summary,
        leaderboard=leaderboard,
        predictions=predictions,
        model_path=model_path,
        leaderboard_path=leaderboard_path,
        metrics_path=metrics_path,
        predictions_path=predictions_path,
    )

