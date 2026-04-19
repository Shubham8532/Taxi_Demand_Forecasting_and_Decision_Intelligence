# Production Checklist

## What Was Structured
- Notebook logic was split into reusable modules:
  - `src/data/load_data.py`
  - `src/features/feature_engineering.py`
  - `src/models/train.py`
  - `src/models/predict.py`
  - `src/utils/metrics.py`
  - `src/pipeline.py` (main `train_model()` and `predict()`)

## Notes
- Logic was kept aligned with your notebook formulas (lags, rolling stats, smoothing, surge/risk/revenue/relocation).
- No cell-order dependency remains in module code.
- Functions are reusable for Docker/FastAPI integration later.

## TODO Before Final Production Push
- Commit and push code to GitHub.
- Track datasets/models with DVC:
  - `dvc add data/interim/historical_features.csv`
  - `dvc add models/best_model_selection_pipeline.joblib`
  - `git add .`
  - `git commit -m "Refactor notebooks into production modules"`
  - `git push`
  - `dvc push`
- Add environment variables/config for MLflow/DagsHub if required.
- Add API layer (FastAPI) that calls `src.pipeline.predict(...)`.

