"""Feature engineering logic extracted from notebook workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error


def safe_fill_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Fill nulls for mixed-type feature frames."""
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            median_val = out[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            out[col] = out[col].fillna(median_val)
        else:
            mode_vals = out[col].mode(dropna=True)
            fill_val = mode_vals.iloc[0] if len(mode_vals) else "unknown"
            out[col] = out[col].fillna(fill_val)
    return out


def get_time_col(df: pd.DataFrame) -> str | None:
    """Return first supported datetime column name."""
    for col in ["pickup_slot", "tpep_pickup_datetime", "timestamp"]:
        if col in df.columns:
            return col
    return None


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred)
    return float(np.mean(2.0 * np.abs(y_true - y_pred) / np.where(denom == 0, 1.0, denom)))


@dataclass
class HistoricalFeatureOutputs:
    """Container for historical and business-ready feature tables."""

    historical: pd.DataFrame
    smoothing_metrics: pd.DataFrame
    top_moves: pd.DataFrame
    best_time_recommendations: pd.DataFrame
    slot_profile: pd.DataFrame


def build_historical_features(
    df: pd.DataFrame,
    neighbors: pd.DataFrame | None = None,
    slot_freq: str = "15min",
    epsilon_val: int = 10,
    ma_windows: Iterable[int] = (3, 4, 6, 8, 12, 16),
    ewma_alphas: Iterable[float] = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    min_val_points: int = 24,
    surge_k: float = 1.0,
) -> HistoricalFeatureOutputs:
    """Create Notebook-4 historical features and business outputs with identical formulas."""
    working = df.copy()
    working["tpep_pickup_datetime"] = pd.to_datetime(working["tpep_pickup_datetime"], errors="coerce")
    working = working.dropna(subset=["tpep_pickup_datetime", "region"]).copy()
    working["region"] = pd.to_numeric(working["region"], errors="coerce")
    working = working.dropna(subset=["region"]).copy()
    working["region"] = working["region"].astype(int)
    working = working.sort_values("tpep_pickup_datetime").reset_index(drop=True)

    if "fare_efficiency" not in working.columns and {"total_amount", "trip_distance"}.issubset(working.columns):
        working["fare_efficiency"] = working["total_amount"] / working["trip_distance"].clip(lower=1e-3)

    # 1) Base 15-minute aggregation
    working["pickup_slot"] = working["tpep_pickup_datetime"].dt.floor(slot_freq)
    grouped_counts = (
        working.groupby(["region", "pickup_slot"], as_index=False)
        .size()
        .rename(columns={"size": "total_pickups_raw"})
    )

    metric_agg: dict[str, str] = {}
    if "total_amount" in working.columns:
        metric_agg["total_amount"] = "sum"
    if "tip_amount" in working.columns:
        metric_agg["tip_amount"] = "sum"
    if "trip_distance" in working.columns:
        metric_agg["trip_distance"] = "sum"
    if "trip_duration_min" in working.columns:
        metric_agg["trip_duration_min"] = "mean"
    if "passenger_count" in working.columns:
        metric_agg["passenger_count"] = "mean"
    if "fare_efficiency" in working.columns:
        metric_agg["fare_efficiency"] = "mean"

    if metric_agg:
        grouped_metrics = working.groupby(["region", "pickup_slot"], as_index=False).agg(metric_agg)
        grouped_metrics = grouped_metrics.rename(
            columns={
                "total_amount": "total_revenue",
                "tip_amount": "total_tip",
                "trip_distance": "total_trip_distance",
                "trip_duration_min": "avg_trip_duration_min",
                "passenger_count": "avg_passenger_count",
                "fare_efficiency": "avg_fare_efficiency",
            }
        )
        historical = grouped_counts.merge(grouped_metrics, on=["region", "pickup_slot"], how="left")
    else:
        historical = grouped_counts.copy()

    all_regions = np.sort(historical["region"].unique())
    all_slots = pd.date_range(historical["pickup_slot"].min(), historical["pickup_slot"].max(), freq=slot_freq)
    full_index = pd.MultiIndex.from_product([all_regions, all_slots], names=["region", "pickup_slot"])
    historical = historical.set_index(["region", "pickup_slot"]).reindex(full_index).reset_index()

    zero_fill_cols = ["total_pickups_raw", "total_revenue", "total_tip", "total_trip_distance"]
    for col in zero_fill_cols:
        if col in historical.columns:
            historical[col] = historical[col].fillna(0)

    mean_like_cols = ["avg_trip_duration_min", "avg_passenger_count", "avg_fare_efficiency"]
    for col in mean_like_cols:
        if col in historical.columns:
            historical[col] = historical.groupby("region")[col].transform(lambda s: s.ffill().bfill())
            historical[col] = historical[col].fillna(historical[col].median())

    historical = historical.sort_values(["region", "pickup_slot"]).reset_index(drop=True)

    # 2) Smoothing tuning
    historical["total_pickups_model"] = historical["total_pickups_raw"].replace(0, epsilon_val)
    tuning_df = historical[["region", "pickup_slot", "total_pickups_model"]].copy()
    tuning_df = tuning_df.sort_values(["region", "pickup_slot"]).reset_index(drop=True)
    tuning_df["row_num"] = tuning_df.groupby("region").cumcount()
    tuning_df["region_size"] = tuning_df.groupby("region")["total_pickups_model"].transform("size")

    val_start_idx = np.maximum(
        (tuning_df["region_size"] * 0.8).astype(int),
        tuning_df["region_size"] - min_val_points,
    )
    tuning_df["is_validation"] = tuning_df["row_num"] >= val_start_idx

    ma_results: list[dict[str, float | int | str]] = []
    for window in ma_windows:
        pred = tuning_df.groupby("region")["total_pickups_model"].transform(
            lambda s: s.shift(1).rolling(window=window, min_periods=1).mean()
        )
        val_mask = tuning_df["is_validation"] & pred.notna()
        y_true = tuning_df.loc[val_mask, "total_pickups_model"].values
        y_pred = pred.loc[val_mask].values
        ma_results.append(
            {
                "method": "moving_average",
                "param": int(window),
                "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "smape": _smape(y_true, y_pred),
                "eval_points": int(val_mask.sum()),
            }
        )

    ewm_results: list[dict[str, float | int | str]] = []
    for alpha in ewma_alphas:
        pred = tuning_df.groupby("region")["total_pickups_model"].transform(
            lambda s: s.shift(1).ewm(alpha=alpha, adjust=False).mean()
        )
        val_mask = tuning_df["is_validation"] & pred.notna()
        y_true = tuning_df.loc[val_mask, "total_pickups_model"].values
        y_pred = pred.loc[val_mask].values
        ewm_results.append(
            {
                "method": "ewma",
                "param": float(alpha),
                "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "smape": _smape(y_true, y_pred),
                "eval_points": int(val_mask.sum()),
            }
        )

    ma_metrics = pd.DataFrame(ma_results).sort_values(["mape", "mae", "rmse"]).reset_index(drop=True)
    ewma_metrics = pd.DataFrame(ewm_results).sort_values(["mape", "mae", "rmse"]).reset_index(drop=True)
    best_ma_row = ma_metrics.iloc[0]
    best_ewma_row = ewma_metrics.iloc[0]

    best_ma_window = int(best_ma_row["param"])
    best_ewma_alpha = float(best_ewma_row["param"])
    selected_smoothing_method = "ewma" if best_ewma_row["mape"] <= best_ma_row["mape"] else "moving_average"

    historical["avg_pickups_ma_tuned"] = historical.groupby("region")["total_pickups_model"].transform(
        lambda s: s.rolling(window=best_ma_window, min_periods=1).mean()
    )
    historical["avg_pickups_ewm_tuned"] = historical.groupby("region")["total_pickups_model"].transform(
        lambda s: s.ewm(alpha=best_ewma_alpha, adjust=False).mean()
    )

    historical["predicted_demand_proxy_ma"] = historical.groupby("region")["avg_pickups_ma_tuned"].shift(1)
    historical["predicted_demand_proxy_ewm"] = historical.groupby("region")["avg_pickups_ewm_tuned"].shift(1)
    historical["predicted_demand_proxy_ma"] = historical["predicted_demand_proxy_ma"].fillna(
        historical["avg_pickups_ma_tuned"]
    )
    historical["predicted_demand_proxy_ewm"] = historical["predicted_demand_proxy_ewm"].fillna(
        historical["avg_pickups_ewm_tuned"]
    )

    historical["predicted_demand_proxy"] = (
        historical["predicted_demand_proxy_ewm"]
        if selected_smoothing_method == "ewma"
        else historical["predicted_demand_proxy_ma"]
    )
    historical["predicted_demand"] = historical["predicted_demand_proxy"].clip(lower=0)

    historical["avg_pickups_ewm"] = historical["avg_pickups_ewm_tuned"]
    historical["avg_pickups"] = historical["predicted_demand_proxy"]
    historical["smoothing_method"] = selected_smoothing_method
    historical["selected_ma_window"] = best_ma_window
    historical["selected_ewma_alpha"] = best_ewma_alpha

    # 3) Time flags
    historical["pickup_day_of_week"] = historical["pickup_slot"].dt.dayofweek
    historical["pickup_hour"] = historical["pickup_slot"].dt.hour
    historical["is_weekend"] = historical["pickup_day_of_week"].isin([5, 6]).astype(int)
    historical["rush_hour"] = historical["pickup_hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)
    historical["is_night"] = ((historical["pickup_hour"] >= 22) | (historical["pickup_hour"] <= 5)).astype(int)

    # 4) Surge and risk
    historical["rolling_mean"] = historical.groupby("region")["predicted_demand"].transform(
        lambda s: s.rolling(window=96, min_periods=1).mean()
    )
    historical["rolling_std"] = historical.groupby("region")["predicted_demand"].transform(
        lambda s: s.rolling(window=96, min_periods=1).std()
    )
    historical["rolling_mean"] = historical["rolling_mean"].fillna(historical["avg_pickups_ewm"])
    historical["rolling_std"] = historical["rolling_std"].fillna(0)

    historical["surge_threshold"] = historical["rolling_mean"] + surge_k * historical["rolling_std"]
    historical["surge_flag"] = historical["predicted_demand"] > historical["surge_threshold"]
    historical["surge_score"] = (
        (historical["predicted_demand"] - historical["rolling_mean"]) / (historical["rolling_std"] + 1e-6)
    )
    historical["surge_intensity"] = historical["predicted_demand"] / (historical["surge_threshold"] + 1e-6)

    historical["surge_level"] = "none"
    historical.loc[historical["surge_flag"] & (historical["surge_intensity"] <= 1.15), "surge_level"] = "low"
    historical.loc[
        historical["surge_flag"]
        & (historical["surge_intensity"] > 1.15)
        & (historical["surge_intensity"] <= 1.35),
        "surge_level",
    ] = "medium"
    historical.loc[historical["surge_flag"] & (historical["surge_intensity"] > 1.35), "surge_level"] = "high"

    historical["risk_score"] = historical["rolling_std"] / (historical["rolling_mean"] + 1e-6)
    historical["risk_band"] = pd.cut(
        historical["risk_score"],
        bins=[-np.inf, 0.35, 0.75, np.inf],
        labels=["Stable", "Moderate", "Volatile"],
    ).astype(str)

    # 5) Revenue and pressure
    if "total_revenue" in historical.columns:
        historical["avg_fare_region_slot"] = historical["total_revenue"] / historical["total_pickups_raw"].clip(lower=1)
        region_fare_baseline = historical.groupby("region")["avg_fare_region_slot"].transform("mean")
        historical["avg_fare_region_slot"] = historical["avg_fare_region_slot"].fillna(region_fare_baseline)
        historical["avg_fare_region_slot"] = historical["avg_fare_region_slot"].fillna(
            historical["avg_fare_region_slot"].median()
        )
    else:
        historical["avg_fare_region_slot"] = np.nan

    historical["expected_revenue"] = historical["predicted_demand"] * historical["avg_fare_region_slot"]

    if "total_trip_distance" in historical.columns and "total_revenue" in historical.columns:
        historical["fare_per_km"] = historical["total_revenue"] / historical["total_trip_distance"].clip(lower=1e-3)
    else:
        historical["fare_per_km"] = np.nan

    if "total_revenue" in historical.columns and "total_tip" in historical.columns:
        historical["tip_ratio"] = historical["total_tip"] / historical["total_revenue"].clip(lower=1e-3)
        historical["tip_per_pickup"] = historical["total_tip"] / historical["total_pickups_raw"].clip(lower=1)
    else:
        historical["tip_ratio"] = np.nan
        historical["tip_per_pickup"] = np.nan

    historical["revenue_density_15min"] = (
        historical["total_revenue"] if "total_revenue" in historical.columns else np.nan
    )

    region_avg_proxy = historical.groupby("region")["predicted_demand"].transform("mean")
    historical["demand_pressure"] = historical["predicted_demand"] / (region_avg_proxy + 1e-6)

    if {"total_trip_distance", "avg_trip_duration_min", "total_pickups_raw"}.issubset(historical.columns):
        historical["avg_trip_distance_per_ride"] = (
            historical["total_trip_distance"] / historical["total_pickups_raw"].clip(lower=1)
        )
        historical["avg_speed_kmh"] = historical["avg_trip_distance_per_ride"] / (
            historical["avg_trip_duration_min"].clip(lower=1e-3) / 60.0
        )
        historical["congestion_band"] = pd.cut(
            historical["avg_speed_kmh"],
            bins=[-np.inf, 12, 22, np.inf],
            labels=["High Congestion", "Moderate", "Low Congestion"],
        ).astype(str)
    else:
        historical["avg_speed_kmh"] = np.nan
        historical["congestion_band"] = "unknown"

    # 6) Relocation recommendations
    if neighbors is not None and not neighbors.empty:
        base = historical[["pickup_slot", "region", "predicted_demand"]].copy()
        candidate_moves = base.merge(neighbors[["region", "target_region", "distance_km"]], on="region", how="left")
        target_demand = base.rename(
            columns={"region": "target_region", "predicted_demand": "target_predicted_demand"}
        )
        candidate_moves = candidate_moves.merge(target_demand, on=["pickup_slot", "target_region"], how="left")
        candidate_moves["target_predicted_demand"] = candidate_moves["target_predicted_demand"].fillna(0)
        candidate_moves["expected_demand_gain"] = (
            candidate_moves["target_predicted_demand"] - candidate_moves["predicted_demand"]
        ).clip(lower=0)
        candidate_moves["relocation_score"] = (
            candidate_moves["expected_demand_gain"] / (candidate_moves["distance_km"] + 1e-3)
        )

        top_moves = (
            candidate_moves.sort_values(
                ["pickup_slot", "region", "relocation_score", "expected_demand_gain"],
                ascending=[True, True, False, False],
            )
            .groupby(["pickup_slot", "region"], as_index=False)
            .head(3)
            .copy()
        )
        top_moves["recommendation_rank"] = top_moves.groupby(["pickup_slot", "region"]).cumcount() + 1

        best_moves = top_moves[top_moves["recommendation_rank"] == 1].copy()
        low_gain_mask = best_moves["expected_demand_gain"] < 1.0
        best_moves.loc[low_gain_mask, "target_region"] = np.nan
        best_moves.loc[low_gain_mask, "distance_km"] = np.nan
        best_moves.loc[low_gain_mask, "expected_demand_gain"] = 0.0
        best_moves.loc[low_gain_mask, "relocation_score"] = 0.0
        best_moves = best_moves.rename(
            columns={"target_region": "recommended_next_zone", "distance_km": "recommended_distance_km"}
        )

        historical = historical.merge(
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
        historical["recommended_next_zone"] = historical["recommended_next_zone"].astype("Float64")
        historical["recommended_next_zone"] = historical["recommended_next_zone"].round().astype("Int64")
    else:
        top_moves = pd.DataFrame()
        historical["recommended_next_zone"] = pd.Series([pd.NA] * len(historical), dtype="Int64")
        historical["recommended_distance_km"] = np.nan
        historical["expected_demand_gain"] = 0.0
        historical["relocation_score"] = 0.0

    # 7) Best-time recommendations
    slot_profile = (
        historical.groupby(["region", "pickup_day_of_week", "pickup_hour"], as_index=False)["predicted_demand"]
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

    day_names = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"}
    best_time_recommendations["day_name"] = best_time_recommendations["pickup_day_of_week"].map(day_names)
    best_time_recommendations["best_time_window"] = best_time_recommendations["pickup_hour"].astype(int).map(
        lambda h: f"{h:02d}:00-{(h + 1) % 24:02d}:00"
    )

    region_best_time = (
        best_time_recommendations[best_time_recommendations["rank"] == 1][["region", "day_name", "best_time_window"]]
        .rename(columns={"day_name": "best_day_name"})
    )
    historical = historical.merge(region_best_time, on="region", how="left")

    smoothing_metrics = pd.concat([ma_metrics, ewma_metrics], ignore_index=True)
    return HistoricalFeatureOutputs(
        historical=historical,
        smoothing_metrics=smoothing_metrics,
        top_moves=top_moves,
        best_time_recommendations=best_time_recommendations,
        slot_profile=slot_profile,
    )


@dataclass
class ModelFeatureOutputs:
    """Model-ready features and metadata for training/inference."""

    train_df: pd.DataFrame
    test_df: pd.DataFrame
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    context_test: pd.DataFrame
    feature_cols: list[str]
    cat_cols: list[str]
    num_cols: list[str]
    target_col: str
    time_col: str


def add_lag_columns(
    df: pd.DataFrame,
    target_col: str,
    time_col: str,
    lag_steps: Iterable[int] = (1, 2, 3, 6, 12, 96),
    rolling_windows: Iterable[int] = (3, 6),
) -> tuple[pd.DataFrame, list[str]]:
    """Create lag and rolling lag features (shifted) region-wise when region exists."""
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    out = out.dropna(subset=[time_col]).copy()

    lag_steps = list(lag_steps)
    rolling_windows = list(rolling_windows)

    if "region" in out.columns:
        out["region"] = pd.to_numeric(out["region"], errors="coerce")
        out = out.dropna(subset=["region"]).copy()
        out["region"] = out["region"].astype(int)
        out = out.sort_values(["region", time_col]).reset_index(drop=True)

        for lag in lag_steps:
            out[f"lag_{lag}"] = out.groupby("region")[target_col].shift(lag)

        for window in rolling_windows:
            out[f"lag_roll_mean_{window}"] = out.groupby("region")[target_col].transform(
                lambda s: s.shift(1).rolling(window=window, min_periods=window).mean()
            )
            out[f"lag_roll_std_{window}"] = out.groupby("region")[target_col].transform(
                lambda s: s.shift(1).rolling(window=window, min_periods=window).std()
            )
    else:
        out = out.sort_values([time_col]).reset_index(drop=True)
        for lag in lag_steps:
            out[f"lag_{lag}"] = out[target_col].shift(lag)
        for window in rolling_windows:
            out[f"lag_roll_mean_{window}"] = out[target_col].shift(1).rolling(window=window, min_periods=window).mean()
            out[f"lag_roll_std_{window}"] = out[target_col].shift(1).rolling(window=window, min_periods=window).std()

    lag_cols = [col for col in out.columns if col.startswith("lag_")]
    return out, lag_cols


def prepare_model_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_candidates: Iterable[str] = ("total_pickups_raw", "total_pickups"),
    lag_steps: Iterable[int] = (1, 2, 3, 6, 12, 96),
    rolling_windows: Iterable[int] = (3, 6),
) -> ModelFeatureOutputs:
    """Build leakage-safe train/test modeling frames from historical features."""
    working_train = train_df.copy()
    working_test = test_df.copy()

    target_col = next((col for col in target_candidates if col in working_train.columns), None)
    if target_col is None:
        raise ValueError(f"Target column missing. Expected one of: {list(target_candidates)}")
    if target_col == "total_pickups_model":
        raise ValueError("`total_pickups_model` is smoothed and not allowed as model-selection target.")

    time_col = get_time_col(working_train)
    if time_col is None:
        raise ValueError("Time column is required.")

    context_keep_cols = [
        col
        for col in [time_col, "region", "rolling_mean", "rolling_std", "avg_fare_region_slot", "avg_pickups", "avg_pickups_ewm"]
        if col in working_test.columns
    ]

    # Train lag features
    train_with_lags, lag_cols_train = add_lag_columns(
        working_train, target_col=target_col, time_col=time_col, lag_steps=lag_steps, rolling_windows=rolling_windows
    )
    train_with_lags = train_with_lags.dropna(subset=lag_cols_train).reset_index(drop=True)

    # Test lag features built from trailing train history to preserve temporal continuity.
    lag_steps = list(lag_steps)
    rolling_windows = list(rolling_windows)
    max_hist = max(lag_steps + rolling_windows) + 2

    if "region" in working_train.columns:
        train_hist_base = working_train.copy()
        train_hist_base[time_col] = pd.to_datetime(train_hist_base[time_col], errors="coerce")
        train_hist_base["region"] = pd.to_numeric(train_hist_base["region"], errors="coerce")
        train_hist_base = train_hist_base.dropna(subset=[time_col, "region"]).copy()
        train_hist_base["region"] = train_hist_base["region"].astype(int)
        train_hist_base = train_hist_base.sort_values(["region", time_col]).reset_index(drop=True)
        history = train_hist_base.groupby("region", group_keys=False).tail(max_hist)
    else:
        train_hist_base = working_train.copy()
        train_hist_base[time_col] = pd.to_datetime(train_hist_base[time_col], errors="coerce")
        train_hist_base = train_hist_base.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
        history = train_hist_base.tail(max_hist)

    history = history.copy()
    history["_is_test"] = 0
    test_tagged = working_test.copy()
    test_tagged["_is_test"] = 1
    test_temp = pd.concat([history, test_tagged], axis=0, ignore_index=True)

    test_with_lags, lag_cols_test = add_lag_columns(
        test_temp, target_col=target_col, time_col=time_col, lag_steps=lag_steps, rolling_windows=rolling_windows
    )
    test_with_lags = test_with_lags[test_with_lags["_is_test"] == 1].copy()
    test_with_lags = test_with_lags.drop(columns=["_is_test"], errors="ignore")

    lag_feature_cols = sorted(set(lag_cols_train).intersection(set(lag_cols_test)))
    if not lag_feature_cols:
        raise ValueError("No common lag features between train and test.")
    test_with_lags = test_with_lags.dropna(subset=lag_feature_cols).reset_index(drop=True)

    allowed_time_cols = [
        "region",
        "pickup_day_of_week",
        "pickup_hour",
        "day_of_week",
        "month",
        "is_weekend",
        "rush_hour",
        "is_night",
    ]
    feature_cols = [col for col in allowed_time_cols if col in train_with_lags.columns and col in test_with_lags.columns]
    feature_cols += [col for col in lag_feature_cols if col in train_with_lags.columns and col in test_with_lags.columns]

    forbidden_prefixes = ("predicted_", "rolling_", "surge_", "risk_", "expected_")
    forbidden_exact = {
        "total_revenue",
        "total_tip",
        "total_trip_distance",
        "avg_pickups",
        "avg_pickups_ewm",
        "avg_pickups_ewm_tuned",
        "avg_pickups_ma_tuned",
        "fare_per_km",
        "tip_ratio",
        "tip_per_pickup",
        "revenue_density_15min",
        "avg_fare_region_slot",
        "avg_speed_kmh",
        "demand_pressure",
        "total_pickups_model",
        target_col,
    }

    bad_features = [
        col
        for col in feature_cols
        if col in forbidden_exact or any(col.lower().startswith(prefix) for prefix in forbidden_prefixes)
    ]
    if bad_features:
        raise ValueError(f"Leakage detected in feature list: {bad_features}")

    X_train = safe_fill_frame(train_with_lags[feature_cols].copy())
    X_test = safe_fill_frame(test_with_lags[feature_cols].copy())
    y_train = train_with_lags[target_col].copy()
    y_test = test_with_lags[target_col].copy()
    context_test = test_with_lags[context_keep_cols].copy() if context_keep_cols else pd.DataFrame(index=test_with_lags.index)

    exact_match_cols = []
    for col in X_train.columns:
        if pd.api.types.is_numeric_dtype(X_train[col]):
            match_ratio = np.mean(np.isclose(X_train[col].to_numpy(), y_train.to_numpy(), rtol=0, atol=1e-12))
            if match_ratio > 0.999:
                exact_match_cols.append((col, float(match_ratio)))
    if exact_match_cols:
        raise ValueError(f"Leakage detected: feature(s) nearly identical to target -> {exact_match_cols}")

    cat_cols = [
        col
        for col in X_train.columns
        if X_train[col].dtype == "object"
        or str(X_train[col].dtype).startswith("category")
        or str(X_train[col].dtype) == "bool"
    ]
    num_cols = [col for col in X_train.columns if col not in cat_cols]

    return ModelFeatureOutputs(
        train_df=train_with_lags,
        test_df=test_with_lags,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        context_test=context_test,
        feature_cols=feature_cols,
        cat_cols=cat_cols,
        num_cols=num_cols,
        target_col=target_col,
        time_col=time_col,
    )


def time_based_validation_split(
    train_df: pd.DataFrame,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    time_col: str,
    valid_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Strict time-aware split with no timestamp overlap."""
    if time_col in train_df.columns:
        time_series = pd.to_datetime(train_df[time_col], errors="coerce")
        unique_times = np.sort(time_series.dropna().unique())

        if len(unique_times) < 2:
            raise ValueError("Not enough unique timestamps for validation split.")

        valid_time_count = max(1, int(len(unique_times) * valid_ratio))
        valid_start_time = unique_times[-valid_time_count]

        fit_mask = time_series < valid_start_time
        valid_mask = time_series >= valid_start_time
        X_fit = X_train.loc[fit_mask].copy()
        y_fit = y_train.loc[fit_mask].copy()
        X_valid = X_train.loc[valid_mask].copy()
        y_valid = y_train.loc[valid_mask].copy()
    else:
        split_idx = int(len(X_train) * (1 - valid_ratio))
        X_fit = X_train.iloc[:split_idx].copy()
        y_fit = y_train.iloc[:split_idx].copy()
        X_valid = X_train.iloc[split_idx:].copy()
        y_valid = y_train.iloc[split_idx:].copy()

    return X_fit, y_fit, X_valid, y_valid

