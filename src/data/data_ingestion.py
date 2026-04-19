"""Backward-compatible wrapper for old ingestion imports.

Deprecated:
Use `src.data.load_data` functions directly.
"""

from __future__ import annotations

from src.data.load_data import (
    find_first_existing,
    get_time_col,
    load_neighbors,
    load_or_build_train_test,
    load_region_labeled_data,
    read_table,
    resolve_project_paths,
)

__all__ = [
    "resolve_project_paths",
    "read_table",
    "find_first_existing",
    "load_region_labeled_data",
    "load_or_build_train_test",
    "load_neighbors",
    "get_time_col",
]

