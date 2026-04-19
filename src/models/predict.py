"""Inference utilities for trained demand models."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.data.load_data import load_neighbors, load_or_build_train_test, resolve_project_paths
from src.features.feature_engineering import get_time_col, prepare_model_features

logger = logging.getLogger(__name__)


@dataclass
class PredictionOutputs:
    """Container for inference outputs."""

    predictions: pd.DataFrame
    top_moves: pd.DataFrame
    best_time_recommendations: pd.DataFrame
    predictions_path: Path
    business_output_path: Path
    relocation_path: Path
    best_time_path: Path


def _build_prediction_table(
    y_true: pd.Series,
    y_pred: np.ndarray,
    context_test: pd.DataFrame,
) -> pd.DataFrame:
    predictions = pd.DataFrame(
        {"actual_demand": y_true.values, "predicted_demand": y_pred},
        index=y_true.index,
    ).reset_index(drop=True)

    time_col = get_time_col(context_test)
    if time_col is not None and time_col in context_test.columns:
        predictions["pickup_slot"] = pd.to_datetime(context_test[time_col].values)
    else:
        predictions["pickup_slot"] = pd.RangeIndex(start=0, stop=len(predictions), step=1)

    if "region" in context_test.columns:
        predictions["region"] = pd.to_numeric(context_test["region"].values, errors="coerce").astype("Int64")
    else:
        predictions["region"] = pd.Series([pd.NA] * len(predictions), dtype="Int64")

    for col in ["rolling_mean", "rolling_std", "avg_fare_region_slot", "avg_pickups", "avg_pickups_ewm"]:
        if col in context_test.columns:
            predictions[col] = pd.to_numeric(context_test[col].values, errors="coerce")

    predictions = predictions.sort_values(["region", "pickup_slot"], na_position="last").reset_index(drop=True)
    return predictions


def _apply_business_logic(
    predictions: pd.DataFrame,
    train_df: pd.DataFrame | None = None,
    neighbors: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # 1) Surge detection
    if "rolling_mean" not in predictions.columns:
        predictions["rolling_mean"] = predictions.groupby("region")["actual_demand"].transform("mean")
    predictions["rolling_mean"] = predictions["rolling_mean"].fillna(predictions["rolling_mean"].median())

    predictions["surge_threshold"] = predictions["rolling_mean"] * 1.3
    predictions["surge_flag"] = predictions["predicted_demand"] > predictions["surge_threshold"]

    surge_ratio = predictions["predicted_demand"] / (predictions["surge_threshold"] + 1e-6)
    predictions["surge_level"] = "none"
    predictions.loc[predictions["surge_flag"] & (surge_ratio <= 1.10), "surge_level"] = "low"
    predictions.loc[
        predictions["surge_flag"] & (surge_ratio > 1.10) & (surge_ratio <= 1.25),
        "surge_level",
    ] = "medium"
    predictions.loc[predictions["surge_flag"] & (surge_ratio > 1.25), "surge_level"] = "high"

    # 2) Risk score
    if "rolling_std" not in predictions.columns:
        predictions["rolling_std"] = predictions.groupby("region")["actual_demand"].transform("std")
    predictions["rolling_std"] = predictions["rolling_std"].fillna(0)

    predictions["risk_score"] = predictions["rolling_std"] / (predictions["rolling_mean"] + 1e-6)
    predictions["risk_band"] = pd.cut(
        predictions["risk_score"],
        bins=[-np.inf, 0.35, 0.75, np.inf],
        labels=["Stable", "Moderate", "Volatile"],
    ).astype(str)

    # 3) Revenue and pressure
    if "avg_fare_region_slot" not in predictions.columns:
        fare_baseline = train_df["avg_fare_region_slot"].median() if train_df is not None and "avg_fare_region_slot" in train_df.columns else np.nan
        predictions["avg_fare_region_slot"] = fare_baseline

    predictions["avg_fare_region_slot"] = predictions["avg_fare_region_slot"].fillna(
        predictions["avg_fare_region_slot"].median()
    )
    predictions["expected_revenue"] = predictions["predicted_demand"] * predictions["avg_fare_region_slot"]

    if "avg_pickups" in predictions.columns:
        denom = predictions["avg_pickups"]
    elif "avg_pickups_ewm" in predictions.columns:
        denom = predictions["avg_pickups_ewm"]
    else:
        denom = predictions.groupby("region")["actual_demand"].transform("mean")
    predictions["demand_pressure"] = predictions["predicted_demand"] / (pd.to_numeric(denom, errors="coerce") + 1e-6)

    # 4) Driver relocation
    if neighbors is not None and not neighbors.empty and predictions["region"].notna().any():
        base = predictions[["pickup_slot", "region", "predicted_demand"]].copy()
        candidate_moves = base.merge(neighbors[["region", "target_region", "distance_km"]], on="region", how="left")

        target_demand = base.rename(
            columns={"region": "target_region", "predicted_demand": "target_predicted_demand"}
        )
        candidate_moves = candidate_moves.merge(target_demand, on=["pickup_slot", "target_region"], how="left")
        candidate_moves["target_predicted_demand"] = candidate_moves["target_predicted_demand"].fillna(0)

        candidate_moves["demand_gain"] = (
            candidate_moves["target_predicted_demand"] - candidate_moves["predicted_demand"]
        ).clip(lower=0)
        candidate_moves["relocation_score"] = candidate_moves["demand_gain"] / (candidate_moves["distance_km"] + 1e-3)

        top_moves = (
            candidate_moves.sort_values(["pickup_slot", "region", "relocation_score"], ascending=[True, True, False])
            .groupby(["pickup_slot", "region"], as_index=False)
            .head(3)
            .copy()
        )
        top_moves["rank"] = top_moves.groupby(["pickup_slot", "region"]).cumcount() + 1

        best_moves = top_moves[top_moves["rank"] == 1].copy()
        low_gain = best_moves["demand_gain"] < 1.0
        best_moves.loc[low_gain, ["target_region", "distance_km", "demand_gain", "relocation_score"]] = [
            np.nan,
            np.nan,
            0.0,
            0.0,
        ]
        best_moves = best_moves.rename(
            columns={
                "target_region": "recommended_next_zone",
                "distance_km": "recommended_distance_km",
                "demand_gain": "expected_demand_gain",
            }
        )

        predictions = predictions.merge(
            best_moves[
                [
                    "pickup_slot",
                    "region",
                    "recommended_next_zone",
                    "recommended_distance_km",
                    "expected_demand_gain",
                    "relocation_score",
                ]
            ],
            on=["pickup_slot", "region"],
            how="left",
        )
    else:
        top_moves = pd.DataFrame()
        predictions["recommended_next_zone"] = pd.Series([pd.NA] * len(predictions), dtype="Int64")
        predictions["recommended_distance_km"] = np.nan
        predictions["expected_demand_gain"] = 0.0
        predictions["relocation_score"] = 0.0

    # 5) Best-time recommendations
    if np.issubdtype(predictions["pickup_slot"].dtype, np.datetime64):
        predictions["pickup_day_of_week"] = predictions["pickup_slot"].dt.dayofweek
        predictions["pickup_hour"] = predictions["pickup_slot"].dt.hour

        slot_profile = (
            predictions.groupby(["region", "pickup_day_of_week", "pickup_hour"], dropna=False, as_index=False)[
                "predicted_demand"
            ]
            .mean()
            .rename(columns={"predicted_demand": "avg_predicted_demand"})
        )

        best_time_recommendations = (
            slot_profile.sort_values(["region", "avg_predicted_demand"], ascending=[True, False])
            .groupby("region", as_index=False)
            .head(3)
            .copy()
        )
        best_time_recommendations["rank"] = best_time_recommendations.groupby("region").cumcount() + 1

        day_names = {
            0: "Monday",
            1: "Tuesday",
            2: "Wednesday",
            3: "Thursday",
            4: "Friday",
            5: "Saturday",
            6: "Sunday",
        }
        best_time_recommendations["day_name"] = best_time_recommendations["pickup_day_of_week"].map(day_names)
        best_time_recommendations["best_time_window"] = best_time_recommendations["pickup_hour"].astype(int).map(
            lambda h: f"{h:02d}:00-{(h + 1) % 24:02d}:00"
        )

        top1_time = (
            best_time_recommendations[best_time_recommendations["rank"] == 1][["region", "best_time_window"]]
            .drop_duplicates(subset=["region"])
        )
        predictions = predictions.merge(top1_time, on="region", how="left")
    else:
        best_time_recommendations = pd.DataFrame()
        predictions["best_time_window"] = np.nan

    return predictions, top_moves, best_time_recommendations


def predict(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    context_test: pd.DataFrame,
    train_df: pd.DataFrame | None = None,
    neighbors: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run model inference and apply business post-processing."""
    y_pred = np.clip(pipeline.predict(X_test), a_min=0, a_max=None)
    predictions = _build_prediction_table(y_true=y_test, y_pred=y_pred, context_test=context_test)
    predictions, top_moves, best_time_recommendations = _apply_business_logic(
        predictions=predictions,
        train_df=train_df,
        neighbors=neighbors,
    )
    return predictions, top_moves, best_time_recommendations


def predict_with_saved_model(
    project_root: Path | None = None,
    model_path: Path | None = None,
    force_rebuild_split: bool = True,
) -> PredictionOutputs:
    """Main inference pipeline: load model + data, predict, and save outputs."""
    paths = resolve_project_paths(project_root)
    if model_path is None:
        model_path = paths.models_dir / "best_model_selection_pipeline.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    pipeline = joblib.load(model_path)
    train_df, test_df = load_or_build_train_test(paths, force_rebuild_split=force_rebuild_split)
    features = prepare_model_features(train_df=train_df, test_df=test_df)
    neighbors = load_neighbors(paths)

    predictions, top_moves, best_time_recommendations = predict(
        pipeline=pipeline,
        X_test=features.X_test,
        y_test=features.y_test,
        context_test=features.context_test,
        train_df=features.train_df,
        neighbors=neighbors,
    )

    predictions_path = paths.data_interim / "model_test_predictions.csv"
    business_output_path = paths.data_interim / "model_business_output.csv"
    relocation_path = paths.data_interim / "model_relocation_candidates.csv"
    best_time_path = paths.data_interim / "model_best_time_recommendations.csv"

    predictions.to_csv(predictions_path, index=False)

    ui_cols = [
        "pickup_slot",
        "region",
        "actual_demand",
        "predicted_demand",
        "surge_flag",
        "surge_level",
        "risk_score",
        "risk_band",
        "expected_revenue",
        "demand_pressure",
        "recommended_next_zone",
        "recommended_distance_km",
        "expected_demand_gain",
        "relocation_score",
        "best_time_window",
    ]
    predictions[[c for c in ui_cols if c in predictions.columns]].to_csv(business_output_path, index=False)
    (top_moves if not top_moves.empty else pd.DataFrame()).to_csv(relocation_path, index=False)
    (best_time_recommendations if not best_time_recommendations.empty else pd.DataFrame()).to_csv(
        best_time_path,
        index=False,
    )

    logger.info("Inference complete. Predictions saved to %s", predictions_path)
    return PredictionOutputs(
        predictions=predictions,
        top_moves=top_moves,
        best_time_recommendations=best_time_recommendations,
        predictions_path=predictions_path,
        business_output_path=business_output_path,
        relocation_path=relocation_path,
        best_time_path=best_time_path,
    )

