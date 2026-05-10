# Spatial-Temporal Taxi Demand Forecasting & Decision Intelligence System

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Built an end-to-end spatial-temporal machine learning system for taxi demand forecasting and operational decision intelligence using large-scale trip datasets.

The system combines time-series forecasting, spatial clustering, and feature engineering to predict region-wise taxi demand patterns and generate actionable insights such as surge detection, hotspot identification, and driver allocation recommendations.

## Features

- Spatial-temporal taxi demand forecasting
- Region-wise demand prediction using geo-spatial clustering
- Time-series feature engineering with lag and rolling-window features
- Peak-hour demand pattern analysis
- Demand hotspot and surge detection
- Decision intelligence layer for driver allocation recommendations
- Scalable preprocessing workflows for large-scale trip datasets

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- TensorFlow
- MLflow
- DVC
- FastAPI
- Azure

## Use Case

This project helps forecast taxi demand across different city regions using historical trip data, temporal trends, and spatial clustering techniques. The generated predictions and analytical insights can support intelligent fleet management, demand balancing, and operational optimization for ride-sharing platforms.

## Future Improvements

- Real-time streaming inference
- Distributed training pipelines
- Advanced geo-spatial modeling
- GPU-accelerated forecasting
- Dynamic pricing optimization

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         uber_demand_forecasting and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── uber_demand_forecasting   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes uber_demand_forecasting a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

