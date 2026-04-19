import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.pipeline import predict, train_model


def test_pipeline_exports():
    assert callable(train_model)
    assert callable(predict)
