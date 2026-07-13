"""Feature engineering modules for demand forecasting."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

_module_path = Path(__file__).resolve().parents[1] / "features.py"
_spec = spec_from_file_location("src.features_impl", _module_path)
_feature_impl = module_from_spec(_spec)
sys.modules[_spec.name] = _feature_impl
_spec.loader.exec_module(_feature_impl)

_add_compatible_features = _feature_impl._add_compatible_features
_prepare_model_input = _feature_impl._prepare_model_input

__all__ = ["_add_compatible_features", "_prepare_model_input"]
