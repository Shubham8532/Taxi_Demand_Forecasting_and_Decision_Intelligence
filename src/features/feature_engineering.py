"""Feature engineering matching notebook 6's final model pipeline.

This module provides:
- ``add_lag_features``: lag + rolling lag creation (per-region)
- ``add_extra_features``: trend_strength, region_hour, region_mean (Bayesian)
- ``add_time_features``: calendar features from timestamp
- ``prepare_train_test_features``: full pipeline from raw historical → X/y
- ``FEATURE_COLS``: the exact ordered feature list the XGB model expects
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────
# The EXACT feature list from notebook 6's final trained model.
# Order matters — must match what the .pkl model was fitted on.
# ──────────────────────────────────────────────────────────────
FEATURE_COLS = [
    # lag features
    "lag_1", "lag_2", "lag_3", "lag_6", "lag_12", "lag_24",
    "lag_roll_mean_3", "lag_roll_std_3", "lag_roll_mean_6", "lag_roll_std_6",
    # base features
    "region", "pickup_hour", "pickup_day_of_week",
    # calendar features (added in notebook 6)
    "week_of_year", "day_of_month", "is_month_start", "is_month_end",
    # engineered features
    "trend_strength", "region_hour", "region_mean",
]

LAG_STEPS = [1, 2, 3, 6, 12, 24]
ROLLING_WINDOWS = [3, 6]

TARGET_CANDIDATES = ["total_pickups_model", "total_pickups_raw", "total_pickups"]


def get_time_col(df: pd.DataFrame) -> str | None:
    """Return first recognised datetime column name, or None."""
    for c in ["pickup_slot", "tpep_pickup_datetime", "timestamp"]:
        if c in df.columns:
            return c
    return None


def safe_fill_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN values: median for numerics, mode for categoricals.

    Parameters
    ----------
    df : pd.DataFrame
        Input frame (not modified in place).

    Returns
    -------
    pd.DataFrame
        Copy with NaNs filled.
    """
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            med = out[col].median()
            out[col] = out[col].fillna(med if not pd.isna(med) else 0.0)
        else:
            modes = out[col].mode(dropna=True)
            out[col] = out[col].fillna(modes.iloc[0] if len(modes) else "unknown")
    return out


# ──────────────────────────────────────────────────────────────
# Lag features (notebook 6 final cell)
# ──────────────────────────────────────────────────────────────
def add_lag_features(
    df: pd.DataFrame,
    target_col: str,
    lag_steps: list[int] | None = None,
    rolling_windows: list[int] | None = None,
) -> pd.DataFrame:
    """Add per-region lag and rolling-lag features.

    Reproduces notebook 6's ``add_lag()`` function exactly.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``region`` and ``target_col`` columns,
        pre-sorted by [region, time].
    target_col : str
        Column to compute lags from.
    lag_steps : list[int], optional
        Lag offsets. Defaults to ``[1, 2, 3, 6, 12, 24]``.
    rolling_windows : list[int], optional
        Rolling window sizes. Defaults to ``[3, 6]``.

    Returns
    -------
    pd.DataFrame
        Input frame with new lag columns appended.
    """
    if lag_steps is None:
        lag_steps = LAG_STEPS
    if rolling_windows is None:
        rolling_windows = ROLLING_WINDOWS

    out = df.copy()

    for lag in lag_steps:
        out[f"lag_{lag}"] = out.groupby("region")[target_col].shift(lag)

    for w in rolling_windows:
        out[f"lag_roll_mean_{w}"] = out.groupby("region")[target_col].transform(
            lambda s: s.shift(1).rolling(w).mean()
        )
        out[f"lag_roll_std_{w}"] = out.groupby("region")[target_col].transform(
            lambda s: s.shift(1).rolling(w).std()
        )

    return out


# ──────────────────────────────────────────────────────────────
# Time / calendar features
# ──────────────────────────────────────────────────────────────
def add_time_features(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """Add calendar columns that notebook 6 uses.

    Adds: ``week_of_year``, ``day_of_month``, ``is_month_start``,
    ``is_month_end``, ``pickup_hour``, ``pickup_day_of_week``.

    Parameters
    ----------
    df : pd.DataFrame
    time_col : str
        Name of the datetime column.

    Returns
    -------
    pd.DataFrame
    """
    out = df.copy()
    dt = pd.to_datetime(out[time_col], errors="coerce")

    out["pickup_hour"] = dt.dt.hour
    out["pickup_day_of_week"] = dt.dt.dayofweek
    out["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    out["day_of_month"] = dt.dt.day
    out["is_month_start"] = dt.dt.is_month_start.astype(int)
    out["is_month_end"] = dt.dt.is_month_end.astype(int)

    return out


# ──────────────────────────────────────────────────────────────
# Extra engineered features (notebook 6 final cells)
# ──────────────────────────────────────────────────────────────
def add_extra_features(
    df: pd.DataFrame,
    target_col: str,
    region_smooth_map: pd.Series | None = None,
    smooth: int = 20,
) -> tuple[pd.DataFrame, pd.Series]:
    """Add trend_strength, region_hour, and Bayesian-smoothed region_mean.

    Parameters
    ----------
    df : pd.DataFrame
        Must already have lag columns and ``pickup_hour``.
    target_col : str
        Target column for computing region_mean stats.
    region_smooth_map : pd.Series, optional
        Pre-computed smoothed region means (from training set).
        If None, computed from ``df`` itself.
    smooth : int, default 20
        Smoothing prior count for Bayesian region mean.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        (DataFrame with new columns, region_smooth_map for reuse on test)
    """
    out = df.copy()

    # trend_strength: how far current demand is from recent average
    out["trend_strength"] = out["lag_1"] - out["lag_roll_mean_3"]

    # region × hour interaction
    out["region_hour"] = out["region"] * out["pickup_hour"]

    # Bayesian-smoothed region mean (avoids overfitting on low-count regions)
    if region_smooth_map is None:
        global_mean = out[target_col].mean()
        region_stats = out.groupby("region")[target_col].agg(["mean", "count"])
        region_smooth_map = (
            (region_stats["mean"] * region_stats["count"] + global_mean * smooth)
            / (region_stats["count"] + smooth)
        )

    out["region_mean"] = out["region"].map(region_smooth_map)

    return out, region_smooth_map


# ──────────────────────────────────────────────────────────────
# Full train/test feature preparation pipeline
# ──────────────────────────────────────────────────────────────
def prepare_train_test_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str | None = None,
    time_col: str | None = None,
) -> dict:
    """Full feature engineering pipeline matching notebook 6.

    Steps:
    1. Resolve target and time columns
    2. Sort by [region, time]
    3. Add time/calendar features
    4. Add lag features (train), then test using train tail as history
    5. Add extra features (trend, region_hour, region_mean)
    6. Build X_train, y_train, X_test, y_test

    Parameters
    ----------
    train_df, test_df : pd.DataFrame
        Raw historical feature tables.
    target_col : str, optional
        Override target column name.
    time_col : str, optional
        Override time column name.

    Returns
    -------
    dict
        Keys: X_train, y_train, X_test, y_test, target_col, time_col,
        feature_cols, region_smooth_map, train_df, test_df.
    """
    train = train_df.copy()
    test = test_df.copy()

    # --- resolve target ---
    if target_col is None:
        target_col = next(
            (c for c in TARGET_CANDIDATES if c in train.columns), None
        )
    if target_col is None:
        raise ValueError(f"Target column missing. Expected one of: {TARGET_CANDIDATES}")

    # --- resolve time column ---
    if time_col is None:
        time_col = get_time_col(train)
    if time_col is None:
        raise ValueError("Time column required (pickup_slot / tpep_pickup_datetime)")

    # --- parse & sort ---
    for df in [train, test]:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    sort_cols = ["region", time_col] if "region" in train.columns else [time_col]
    train = train.sort_values(sort_cols).reset_index(drop=True)
    test = test.sort_values(sort_cols).reset_index(drop=True)

    # --- time features ---
    train = add_time_features(train, time_col)
    test = add_time_features(test, time_col)

    # --- lag features (train) ---
    train = add_lag_features(train, target_col)
    lag_cols = [c for c in train.columns if c.startswith("lag_")]
    min_required = [c for c in lag_cols if "lag_1" in c or "lag_2" in c]
    train = train.dropna(subset=min_required).reset_index(drop=True)

    # --- lag features (test) — use train tail as history ---
    history = train.tail(200)
    test_temp = pd.concat([history, test], axis=0).reset_index(drop=True)
    test_temp = add_lag_features(test_temp, target_col)
    test = test_temp.iloc[len(history):].copy()
    test = test.dropna(subset=min_required)
    if len(test) == 0:
        test = test.ffill().bfill()
    test = test.reset_index(drop=True)

    # --- extra features ---
    train, region_smooth_map = add_extra_features(train, target_col)
    test, _ = add_extra_features(test, target_col, region_smooth_map=region_smooth_map)

    # --- build X / y ---
    feature_cols = [c for c in FEATURE_COLS if c in train.columns]
    X_train = safe_fill_frame(train[feature_cols])
    y_train = train[target_col]
    X_test = safe_fill_frame(test[feature_cols])
    y_test = test[target_col]

    # fix bool columns
    for col in X_train.select_dtypes(include=["bool"]).columns:
        X_train[col] = X_train[col].astype(int)
        X_test[col] = X_test[col].astype(int)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "target_col": target_col,
        "time_col": time_col,
        "feature_cols": feature_cols,
        "region_smooth_map": region_smooth_map,
        "train_df": train,
        "test_df": test,
    }
