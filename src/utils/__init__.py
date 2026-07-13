"""Shared utility functions: metrics, geo helpers."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

_module_path = Path(__file__).resolve().parents[1] / "utils.py"
_spec = spec_from_file_location("src.utils_impl", _module_path)
_utils_impl = module_from_spec(_spec)
sys.modules[_spec.name] = _utils_impl
_spec.loader.exec_module(_utils_impl)

calculate_distance = _utils_impl.calculate_distance
get_demand_color = _utils_impl.get_demand_color
validate_coordinates = _utils_impl.validate_coordinates

__all__ = ["calculate_distance", "get_demand_color", "validate_coordinates"]
