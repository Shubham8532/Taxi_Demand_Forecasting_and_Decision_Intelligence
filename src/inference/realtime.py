"""Lightweight inference engine for Azure deployment.

Solves the core deployment problem: instead of loading the full CSV
on every request, this module uses a tiny pre-computed stats file
(``region_inference_stats.pkl``) containing per-region lag history
and statistics.

Usage at deployment time:
    1. Run ``build_region_stats()`` once (locally or in CI) with the
       full ``final_data.csv`` to produce ``region_inference_stats.pkl``
    2. Deploy ``xgb_model.pkl`` + ``region_inference_stats.pkl`` (~6 KB)
    3. At request time, call ``predict_demand()`` — no CSV needed

Classes
-------
RealtimePredictor
    Stateful predictor that caches model + stats on first load.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.features.feature_engineering import FEATURE_COLS, LAG_STEPS, ROLLING_WINDOWS

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# 1. BUILD STATS (run once, offline)
# ──────────────────────────────────────────────────────────────
def build_region_stats(
    data_path: str | Path,
    output_path: str | Path,
    target_col: str = "total_pickups_model",
    time_col: str | None = None,
    smooth: int = 20,
) -> dict:
    """Pre-compute per-region statistics from the full historical CSV.

    Creates a lightweight dictionary with everything needed for
    real-time inference without loading the CSV again.

    Parameters
    ----------
    data_path : str or Path
        Path to ``final_data.csv`` or ``historical_features.csv``.
    output_path : str or Path
        Where to save the ``.pkl`` stats file.
    target_col : str
        Target column to compute lags from.
    time_col : str, optional
        Datetime column name. Auto-detected if None.
    smooth : int
        Bayesian smoothing prior count for region_mean.

    Returns
    -------
    dict
        The stats dictionary that was saved.
    """
    df = pd.read_csv(data_path, low_memory=False)

    # resolve time column
    if time_col is None:
        for c in ["pickup_slot", "tpep_pickup_datetime", "timestamp"]:
            if c in df.columns:
                time_col = c
                break
    if time_col is None:
        raise ValueError("No time column found")

    # resolve target
    if target_col not in df.columns:
        for c in ["total_pickups_model", "total_pickups_raw", "total_pickups"]:
            if c in df.columns:
                target_col = c
                break

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])
    df["region"] = pd.to_numeric(df.get("region", df.get("region_id")), errors="coerce")
    df = df.dropna(subset=["region"])
    df["region"] = df["region"].astype(int)
    df = df.sort_values(["region", time_col]).reset_index(drop=True)

    # global stats for Bayesian smoothing
    global_mean = df[target_col].mean()
    region_agg = df.groupby("region")[target_col].agg(["mean", "count", "std"])

    # per-region stats
    max_lag = max(LAG_STEPS) + max(ROLLING_WINDOWS) + 2  # need enough history
    stats = {}

    for region_id, group in df.groupby("region"):
        tail = group.tail(max_lag)
        values = tail[target_col].tolist()

        # Bayesian-smoothed region mean
        rmean = region_agg.loc[region_id, "mean"]
        rcount = region_agg.loc[region_id, "count"]
        region_mean = (rmean * rcount + global_mean * smooth) / (rcount + smooth)

        stats[int(region_id)] = {
            "last_values": values,  # last N pickup counts
            "region_mean": float(region_mean),
            "region_std": float(region_agg.loc[region_id, "std"] or 0),
            "region_count": int(rcount),
        }

    output = {
        "stats": stats,
        "global_mean": float(global_mean),
        "target_col": target_col,
        "smooth": smooth,
        "n_regions": len(stats),
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(output, output_path)
    logger.info(
        "Saved region stats: %d regions, %s",
        len(stats), output_path,
    )
    return output


# ──────────────────────────────────────────────────────────────
# 2. BUILD FEATURES FROM STATS (at request time)
# ──────────────────────────────────────────────────────────────
def build_features_from_stats(
    region_id: int,
    timestamp: pd.Timestamp,
    region_stats: dict,
) -> pd.DataFrame:
    """Build a single-row feature DataFrame for one region + timestamp.

    Uses pre-computed lag history from ``region_stats`` instead of
    loading the full CSV. Produces exactly the columns in ``FEATURE_COLS``.

    Parameters
    ----------
    region_id : int
        Target region.
    timestamp : pd.Timestamp
        Prediction timestamp.
    region_stats : dict
        Per-region stats dict (from ``build_region_stats``).

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with columns matching ``FEATURE_COLS``.
    """
    info = region_stats.get(region_id)
    if info is None:
        # fallback: use first available region's structure with zeros
        info = {"last_values": [0] * 30, "region_mean": 0, "region_std": 0}

    values = info["last_values"]

    # Build lag features from the tail of historical values
    # values[-1] is the most recent, values[-2] is one step back, etc.
    def _safe_get(idx):
        """Get value at reverse index, 0 if out of range."""
        return values[-(idx)] if idx <= len(values) else 0

    row = {}

    # Lags: lag_k means "value k steps ago"
    for lag in LAG_STEPS:
        row[f"lag_{lag}"] = _safe_get(lag)

    # Rolling stats from the most recent values
    for w in ROLLING_WINDOWS:
        recent = values[-(w + 1):-1] if len(values) > w else values
        if len(recent) > 0:
            row[f"lag_roll_mean_{w}"] = float(np.mean(recent))
            row[f"lag_roll_std_{w}"] = float(np.std(recent, ddof=1)) if len(recent) > 1 else 0.0
        else:
            row[f"lag_roll_mean_{w}"] = 0.0
            row[f"lag_roll_std_{w}"] = 0.0

    # Base features
    row["region"] = region_id
    row["pickup_hour"] = timestamp.hour
    row["pickup_day_of_week"] = timestamp.dayofweek

    # Calendar features (notebook 6)
    row["week_of_year"] = timestamp.isocalendar()[1]
    row["day_of_month"] = timestamp.day
    row["is_month_start"] = int(timestamp.day == 1)
    row["is_month_end"] = int(
        timestamp.day == pd.Timestamp(
            year=timestamp.year, month=timestamp.month, day=1
        ).days_in_month
    )

    # Engineered features
    row["trend_strength"] = row["lag_1"] - row["lag_roll_mean_3"]
    row["region_hour"] = row["region"] * row["pickup_hour"]
    row["region_mean"] = info["region_mean"]

    # Build DataFrame in the exact column order the model expects
    feature_row = {col: row.get(col, 0) for col in FEATURE_COLS}
    return pd.DataFrame([feature_row], columns=FEATURE_COLS)


def build_features_all_regions(
    timestamp: pd.Timestamp,
    all_region_stats: dict,
) -> pd.DataFrame:
    """Build feature rows for ALL regions at once.

    Parameters
    ----------
    timestamp : pd.Timestamp
    all_region_stats : dict
        The ``stats`` sub-dict from ``build_region_stats`` output.

    Returns
    -------
    pd.DataFrame
        One row per region, columns matching ``FEATURE_COLS``.
    """
    rows = []
    for rid in sorted(all_region_stats.keys()):
        row_df = build_features_from_stats(rid, timestamp, all_region_stats)
        rows.append(row_df)
    if not rows:
        return pd.DataFrame(columns=FEATURE_COLS)
    return pd.concat(rows, ignore_index=True)


# ──────────────────────────────────────────────────────────────
# 3. REALTIME PREDICTOR (cached, stateful)
# ──────────────────────────────────────────────────────────────
class RealtimePredictor:
    """Lightweight predictor for Azure App Service.

    Loads the XGB model and region stats once, then serves
    predictions from memory with zero CSV I/O.

    Parameters
    ----------
    model_path : str or Path
        Path to ``xgb_model.pkl``.
    stats_path : str or Path
        Path to ``region_inference_stats.pkl``.
    region_mapping_path : str or Path
        Path to ``region_mapping.json``.
    """

    def __init__(
        self,
        model_path: str | Path,
        stats_path: str | Path,
        region_mapping_path: str | Path,
    ):
        self.model_path = Path(model_path)
        self.stats_path = Path(stats_path)
        self.region_mapping_path = Path(region_mapping_path)

        self._model = None
        self._stats = None
        self._region_mapping = None

    def _ensure_loaded(self) -> None:
        """Lazy-load model and stats on first call."""
        if self._model is None:
            self._model = joblib.load(self.model_path)
            logger.info("Model loaded from %s", self.model_path)

        if self._stats is None:
            raw = joblib.load(self.stats_path)
            self._stats = raw["stats"]
            self._global_mean = raw["global_mean"]
            logger.info(
                "Stats loaded: %d regions from %s",
                len(self._stats), self.stats_path,
            )

        if self._region_mapping is None:
            with open(self.region_mapping_path, "r") as f:
                self._region_mapping = json.load(f)
            self._region_mapping = {
                int(k): v for k, v in self._region_mapping.items()
            }
            logger.info("Region mapping loaded: %d regions", len(self._region_mapping))

    @property
    def region_mapping(self) -> dict:
        """Region mapping dict (lazy-loaded)."""
        self._ensure_loaded()
        return self._region_mapping

    def predict_single(
        self, region_id: int, timestamp: pd.Timestamp
    ) -> float:
        """Predict demand for one region at one timestamp.

        Parameters
        ----------
        region_id : int
        timestamp : pd.Timestamp

        Returns
        -------
        float
            Predicted pickup count (non-negative).
        """
        self._ensure_loaded()
        X = build_features_from_stats(region_id, timestamp, self._stats)
        y_log = self._model.predict(X)
        pred = float(np.expm1(y_log[0]))
        return max(pred, 0.0)

    def predict_all_regions(
        self, timestamp: pd.Timestamp
    ) -> list[dict[str, Any]]:
        """Predict demand for all regions at a timestamp.

        Parameters
        ----------
        timestamp : pd.Timestamp

        Returns
        -------
        list[dict]
            Each dict has region_id, predicted_demand, region info.
        """
        self._ensure_loaded()
        X_all = build_features_all_regions(timestamp, self._stats)

        y_log = self._model.predict(X_all)
        preds = np.clip(np.expm1(y_log), 0, None)

        results = []
        for i, rid in enumerate(sorted(self._stats.keys())):
            info = self._region_mapping.get(
                rid, {"name": f"Region {rid}", "lat": 0, "lon": 0}
            )
            results.append({
                "region_id": rid,
                "name": info.get("name", f"Region {rid}"),
                "lat": info.get("lat", 0),
                "lon": info.get("lon", 0),
                "predicted_demand": int(preds[i]),
            })

        return results
