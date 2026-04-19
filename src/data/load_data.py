"""Data loading utilities extracted from notebook workflow."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProjectPaths:
    """Canonical project paths used across data/features/models modules."""

    project_root: Path
    data_raw: Path
    data_interim: Path
    data_processed: Path
    models_dir: Path
    reports_dir: Path


def resolve_project_paths(project_root: Path | None = None) -> ProjectPaths:
    """Resolve cookiecutter-style project directories."""
    if project_root is None:
        cwd = Path.cwd().resolve()
        project_root = cwd.parent if cwd.name == "notebooks" else cwd

    paths = ProjectPaths(
        project_root=project_root,
        data_raw=project_root / "data" / "raw",
        data_interim=project_root / "data" / "interim",
        data_processed=project_root / "data" / "processed",
        models_dir=project_root / "models",
        reports_dir=project_root / "reports",
    )

    for folder in [
        paths.data_raw,
        paths.data_interim,
        paths.data_processed,
        paths.models_dir,
        paths.reports_dir,
    ]:
        folder.mkdir(parents=True, exist_ok=True)

    return paths


def get_time_col(df: pd.DataFrame) -> str | None:
    """Return first supported datetime column name."""
    for col in ["pickup_slot", "tpep_pickup_datetime", "timestamp"]:
        if col in df.columns:
            return col
    return None


def find_first_existing(candidates: Iterable[Path]) -> Path | None:
    """Return first existing path from candidate list."""
    for path in candidates:
        if path.exists():
            return path
    return None


def read_table(path: Path, **kwargs) -> pd.DataFrame:
    """Read CSV/Parquet table using path suffix."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, **kwargs)
    if suffix == ".parquet":
        return pd.read_parquet(path, **kwargs)
    raise ValueError(f"Unsupported file format: {path}")


def load_region_labeled_data(paths: ProjectPaths, input_path: Path | None = None) -> pd.DataFrame:
    """Load and normalize region-labeled trip data for historical feature creation."""
    if input_path is None:
        candidates = [
            paths.data_interim / "region_labeled_data.csv",
            paths.data_interim / "time_series.csv",
            paths.data_interim / "processing_data.csv",
            paths.data_interim / "region_labeled_data.parquet",
        ]
        input_path = find_first_existing(candidates)

    if input_path is None:
        raise FileNotFoundError("No region-labeled input found in data/interim.")

    df = read_table(input_path, low_memory=False) if input_path.suffix == ".csv" else read_table(input_path)

    if "region" in df.columns:
        df["region"] = pd.to_numeric(df["region"], errors="coerce")
    elif "region_id" in df.columns:
        df["region"] = pd.to_numeric(df["region_id"], errors="coerce")
    else:
        raise ValueError("Missing region column. Expected `region` or `region_id`.")

    if "tpep_pickup_datetime" not in df.columns:
        raise ValueError("Missing `tpep_pickup_datetime` column.")

    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"], errors="coerce")
    df = df.dropna(subset=["tpep_pickup_datetime", "region"]).copy()
    df["region"] = df["region"].astype(int)
    df = df.sort_values("tpep_pickup_datetime").reset_index(drop=True)

    logger.info("Loaded region-labeled data from %s (rows=%s)", input_path, f"{len(df):,}")
    return df


def load_neighbors(paths: ProjectPaths, neighbors_path: Path | None = None) -> pd.DataFrame | None:
    """Load optional region-neighbor table used by relocation scoring."""
    if neighbors_path is None:
        neighbors_path = paths.data_interim / "region_neighbors.csv"
    if not neighbors_path.exists():
        return None

    neighbors = pd.read_csv(neighbors_path)
    neighbors = neighbors.rename(columns={"region_id": "region", "neighbor_region_id": "target_region"})
    required = {"region", "target_region", "distance_km"}
    if not required.issubset(neighbors.columns):
        missing = sorted(required - set(neighbors.columns))
        raise ValueError(f"Neighbors file missing columns: {missing}")
    return neighbors


def _time_split(df: pd.DataFrame, time_col: str, split_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split by unique timestamps to avoid overlap leakage."""
    ordered = df.copy()
    ordered[time_col] = pd.to_datetime(ordered[time_col], errors="coerce")
    ordered = ordered.dropna(subset=[time_col]).copy()

    if "region" in ordered.columns:
        ordered["region"] = pd.to_numeric(ordered["region"], errors="coerce")
        ordered = ordered.dropna(subset=["region"]).copy()
        ordered["region"] = ordered["region"].astype(int)
        ordered = ordered.sort_values([time_col, "region"]).reset_index(drop=True)
    else:
        ordered = ordered.sort_values(time_col).reset_index(drop=True)

    unique_times = np.sort(ordered[time_col].unique())
    if len(unique_times) < 2:
        raise ValueError("Need at least 2 unique timestamps for split.")

    cut_idx = max(1, int(len(unique_times) * split_ratio))
    cut_idx = min(cut_idx, len(unique_times) - 1)
    cutoff_time = unique_times[cut_idx - 1]

    train_df = ordered[ordered[time_col] <= cutoff_time].copy()
    test_df = ordered[ordered[time_col] > cutoff_time].copy()
    return train_df, test_df


def load_or_build_train_test(
    paths: ProjectPaths,
    force_rebuild_split: bool = True,
    split_ratio: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train/test if available or build a fresh chronological split from historical features."""
    train_path = paths.data_processed / "train.csv"
    test_path = paths.data_processed / "test.csv"

    hist_candidates = [
        paths.data_interim / "historical_features.csv",
        paths.data_interim / "final_data.csv",
        Path("/kaggle/working/historical_features.csv"),
        Path("/kaggle/working/final_data.csv"),
        Path("/kaggle/input/notebooks/shubhamyadav74/notebookacca71d522/historical_features.csv"),
        Path("/kaggle/input/notebooks/shubhamyadav74/notebookacca71d522/final_data.csv"),
    ]
    hist_path = find_first_existing(hist_candidates)

    if force_rebuild_split and hist_path is not None:
        all_df = read_table(hist_path, low_memory=False) if hist_path.suffix == ".csv" else read_table(hist_path)
        time_col = get_time_col(all_df)
        if time_col is None:
            raise ValueError("Historical data missing time column.")

        train_df, test_df = _time_split(all_df, time_col=time_col, split_ratio=split_ratio)
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        logger.info(
            "Built fresh train/test split from %s -> train=%s test=%s",
            hist_path,
            train_df.shape,
            test_df.shape,
        )
        return train_df, test_df

    if train_path.exists() and test_path.exists():
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        logger.info("Loaded existing train/test split from data/processed")
        return train_df, test_df

    raise FileNotFoundError("No train/test split and no historical features found. Run historical pipeline first.")

