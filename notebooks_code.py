
# --- notebooks\1.EDA.ipynb ---
import numpy as np
import pandas as pd
import dask.dataframe as dd
import matplotlib.pyplot as plt
import seaborn as sns
# --------------------
# paths for the three dfs

df_jan_path = "/kaggle/input/datasets/elemento/nyc-yellow-taxi-trip-data/yellow_tripdata_2016-01.csv"
df_feb_path = "/kaggle/input/datasets/elemento/nyc-yellow-taxi-trip-data/yellow_tripdata_2016-02.csv"
df_mar_path = "/kaggle/input/datasets/elemento/nyc-yellow-taxi-trip-data/yellow_tripdata_2016-03.csv"

# load the dataframes

df_jan = dd.read_csv(df_jan_path, assume_missing=True)
df_feb = dd.read_csv(df_feb_path, assume_missing=True)
df_mar = dd.read_csv(df_mar_path, assume_missing=True)
# --------------------
df_jan
# --------------------
df_jan.visualize(tasks=True)
# --------------------
# Step 1 (memory-safe): read only needed columns, then concat

df_final = dd.concat([df_jan, df_feb, df_mar], axis=0)
# --------------------
# Step 2.1: parse datetime safely  --> string to datetime
df_final["tpep_pickup_datetime"] = dd.to_datetime(df_final["tpep_pickup_datetime"], errors="coerce")
df_final["tpep_dropoff_datetime"] = dd.to_datetime(df_final["tpep_dropoff_datetime"], errors="coerce")

# --------------------
# Step 2.2: minimal cleaning rules

df_final["trip_duration_min"] = (
    (df_final["tpep_dropoff_datetime"] - df_final["tpep_pickup_datetime"]).dt.total_seconds() / 60.0
)

clean_df = df_final[
    (df_final["tpep_pickup_datetime"].notnull()) &
    (df_final["tpep_dropoff_datetime"].notnull()) &
    (df_final["trip_duration_min"] > 0) &
    (df_final["trip_distance"] > 0) &
    (df_final["fare_amount"] > 0)
]

# --------------------
# Step 2.3: sanity check
clean_df[["tpep_pickup_datetime", "tpep_dropoff_datetime", "trip_duration_min", "trip_distance", "fare_amount"]].head()
# --------------------
# check for missing values in the data

clean_df.isna().sum().compute()
# --------------------
# datatypes

df_final.dtypes
# --------------------
# CHanging datatypes of few cols
float_cols = [
    "VendorID", "passenger_count", "trip_distance",
    "pickup_longitude", "pickup_latitude", "RatecodeID",
    "dropoff_longitude", "dropoff_latitude", "payment_type",
    "fare_amount", "extra", "mta_tax", "tip_amount",
    "tolls_amount", "improvement_surcharge", "total_amount",
    "trip_duration_min"
]

for c in float_cols:
    if c in df_final.columns:
        df_final[c] = df_final[c].astype("float32")

if "store_and_fwd_flag" in df_final.columns:
    df_final["store_and_fwd_flag"] = df_final["store_and_fwd_flag"].astype("category")

df_final.dtypes

# --------------------
df_final.describe().compute()
# --------------------
# countplot for passenger plot

passenger_count = df_final["passenger_count"].value_counts().compute()
passenger_count.sort_index().plot(kind="bar")
plt.show()
# --------------------
# boxplot for the trip distance

sns.boxplot(df_final.loc[:,"trip_distance"].compute())
# --------------------
# percentile values for trip distance

percentile_values = list(np.arange(0.1, 1.0, 0.1)) + list(np.arange(0.95, 1.01, 0.01))

for percentile in percentile_values:
    print(f"{int(percentile*100)}th percentile:",
          df_final['trip_distance'].quantile(q=percentile).compute())
# --------------------
# boxplot for the fare amount

sns.boxplot(df_final.loc[:,"fare_amount"].compute())
# --------------------
# percentile values for fare amount

percentile_values = list(np.arange(0.1, 1.0, 0.1)) + [0.95, 0.96, 0.97, 0.98, 0.99, 1.0]
percentile_values

for percentile in percentile_values:
    print(f"The fare amount value for {int(percentile * 100)}th percentile is {df_final['fare_amount'].quantile(q=percentile).compute()}")
# --------------------
sns.boxplot(df_final.loc[:,"trip_duration_min"].compute())
# --------------------
# percentile values for trip_duration min

percentile_values = list(np.arange(0.1, 1.0, 0.1)) + [0.95, 0.96, 0.97, 0.98, 0.99, 1.0]
percentile_values

for percentile in percentile_values:
    print(f"The fare amount value for {int(percentile * 100)}th percentile is {df_final['trip_duration_min'].quantile(q=percentile).compute()}")
# --------------------
location_subset = df_final[['pickup_latitude','pickup_longitude', 
                           'dropoff_latitude','dropoff_longitude']]

location_subset
# --------------------
# boxplots for location based columns

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=2, figsize=(12,5))
for i in range(4):
    if i <= 1:
        # plot the boxplot
        sns.boxplot(y=location_subset.iloc[:,i].compute(),ax=ax1[i])
    else:
        # plot the boxplot
        sns.boxplot(y=location_subset.iloc[:,i].compute(),ax=ax2[i - 2])
# --------------------
# Store and Fwd Flag

sns.countplot(df_final["store_and_fwd_flag"].compute())
# --------------------
# create new columns

df_final["pickup_months"] = df_final["tpep_pickup_datetime"].dt.month
df_final["pickup_day_of_week"] = df_final["tpep_pickup_datetime"].dt.dayofweek
df_final["pickup_hour"] = df_final["tpep_pickup_datetime"].dt.hour
# --------------------
# plot the number of pickups

pickups_every_1_days = (
    df_final[["tpep_pickup_datetime"]]
    .dropna(subset=["tpep_pickup_datetime"])
    .set_index("tpep_pickup_datetime")
    .resample("1D")
    .size()
    .compute()
)

sns.lineplot(x=pickups_every_1_days.index, y=pickups_every_1_days.values)
plt.xticks(rotation=45)
plt.title("Daily Pickups")
plt.show()
# --------------------
# pickups for each hour of the day

pickups_each_hour = (
    df_final
    .groupby(["pickup_hour", "pickup_day_of_week"])
    .size()
    .compute()
    .reset_index(name="Number of Pickups")
)

pickups_each_hour.head()
# --------------------
pickups_each_hour = pickups_each_hour.reset_index()

pickups_each_hour.rename(columns={"VendorID":"Number of Pickups"},inplace=True)

day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

replacement_mapper = {k:v for k,v in enumerate(day_names)}

replacement_mapper
# --------------------
pickups_each_hour.replace({"pickup_day_of_week":replacement_mapper},inplace=True)

# plot the lineplot

fig = plt.figure(figsize=(12,6))

sns.lineplot(pickups_each_hour, x="pickup_hour", y="Number of Pickups", 
             hue="pickup_day_of_week",hue_order=day_names)

plt.show()
# --------------------
# Use cleaned data for all downstream EDA
# ...existing code...
clean_df["pickup_months"] = clean_df["tpep_pickup_datetime"].dt.month
clean_df["pickup_day_of_week"] = clean_df["tpep_pickup_datetime"].dt.dayofweek
clean_df["pickup_hour"] = clean_df["tpep_pickup_datetime"].dt.hour
# ...existing code...
# --------------------
# ...existing code...

# Build zone_id safely
zone_df = clean_df[
    ["tpep_pickup_datetime", "pickup_latitude", "pickup_longitude",
     "trip_distance", "fare_amount", "trip_duration_min"]
].copy()

zone_df["lat_bin"] = zone_df["pickup_latitude"].round(2)
zone_df["lon_bin"] = zone_df["pickup_longitude"].round(2)
zone_df["zone_id"] = zone_df["lat_bin"].astype("string") + "_" + zone_df["lon_bin"].astype("string")

# Aggregate 1: means
zone_stats_pd = (
    zone_df.groupby("zone_id")
    .agg({
        "trip_distance": "mean",
        "fare_amount": "mean",
        "trip_duration_min": "mean"
    })
    .rename(columns={
        "trip_distance": "zone_avg_trip_distance",
        "fare_amount": "zone_avg_fare",
        "trip_duration_min": "zone_avg_duration_min"
    })
    .compute()
    .reset_index()
)

# Aggregate 2: counts
zone_counts_pd = (
    zone_df.groupby("zone_id")
    .size()
    .compute()
    .reset_index(name="zone_total_pickups")
)

# Merge in pandas (stable)
zone_features = zone_stats_pd.merge(zone_counts_pd, on="zone_id", how="left")

zone_features.sort_values("zone_total_pickups", ascending=False).head(20)
# ...existing code...
# --------------------
# each zone’s peak pickup hour

# What: find the busiest hour for each zone
# Why: captures zone-specific temporal behavior (business vs residential vs transit patterns)

zone_hour_pd = (
    zone_df.assign(pickup_hour=zone_df["tpep_pickup_datetime"].dt.hour)
    .groupby(["zone_id", "pickup_hour"])
    .size()
    .compute()
    .reset_index(name="pickup_count")
)

zone_peak_hour = (
    zone_hour_pd.sort_values(["zone_id", "pickup_count"], ascending=[True, False])
    .drop_duplicates("zone_id")
    .rename(columns={"pickup_hour": "zone_peak_hour"})
    [["zone_id", "zone_peak_hour"]]
)

zone_features = zone_features.merge(zone_peak_hour, on="zone_id", how="left")
zone_features.head()
# --------------------

--- notebooks\2.Removing_Outliers.ipynb ---
import numpy as np
import pandas as pd
import dask.dataframe as dd
import matplotlib.pyplot as plt
import seaborn as sns
# --------------------
# paths for the three dfs

df_jan_path = "/kaggle/input/datasets/elemento/nyc-yellow-taxi-trip-data/yellow_tripdata_2016-01.csv"
df_feb_path = "/kaggle/input/datasets/elemento/nyc-yellow-taxi-trip-data/yellow_tripdata_2016-02.csv"
df_mar_path = "/kaggle/input/datasets/elemento/nyc-yellow-taxi-trip-data/yellow_tripdata_2016-03.csv"

# load the dataframes

df_jan = dd.read_csv(df_jan_path, assume_missing=True)
df_feb = dd.read_csv(df_feb_path, assume_missing=True)
df_mar = dd.read_csv(df_mar_path, assume_missing=True)
# --------------------
# Step 1 (memory-safe): read only needed columns, then concat

df_final = dd.concat([df_jan, df_feb, df_mar], axis=0)
# --------------------
# Step 2.1: parse datetime safely  --> string to datetime
df_final["tpep_pickup_datetime"] = dd.to_datetime(df_final["tpep_pickup_datetime"], errors="coerce")
df_final["tpep_dropoff_datetime"] = dd.to_datetime(df_final["tpep_dropoff_datetime"], errors="coerce")

# --------------------
# Step 2.2: minimal cleaning rules

df_final["trip_duration_min"] = (
    (df_final["tpep_dropoff_datetime"] - df_final["tpep_pickup_datetime"]).dt.total_seconds() / 60.0
)

clean_df = df_final[
    (df_final["tpep_pickup_datetime"].notnull()) &
    (df_final["tpep_dropoff_datetime"].notnull()) &
    (df_final["trip_duration_min"] > 0) &
    (df_final["trip_distance"] > 0) &
    (df_final["fare_amount"] > 0)
]

# --------------------
# Step 2.3: sanity check
clean_df[["tpep_pickup_datetime", "tpep_dropoff_datetime", "trip_duration_min", "trip_distance", "fare_amount"]].head()
# --------------------
# datatypes

df_final.dtypes
# --------------------
# CHanging datatypes of few cols
float_cols = [
    "VendorID", "passenger_count", "trip_distance",
    "pickup_longitude", "pickup_latitude", "RatecodeID",
    "dropoff_longitude", "dropoff_latitude", "payment_type",
    "fare_amount", "extra", "mta_tax", "tip_amount",
    "tolls_amount", "improvement_surcharge", "total_amount",
    "trip_duration_min"
]

for c in float_cols:
    if c in df_final.columns:
        df_final[c] = df_final[c].astype("float32")

if "store_and_fwd_flag" in df_final.columns:
    df_final["store_and_fwd_flag"] = df_final["store_and_fwd_flag"].astype("category")

df_final.dtypes

# --------------------
sns.boxplot(df_final.loc[:,"trip_duration_min"].compute())
# --------------------
location_subset = df_final[['pickup_latitude','pickup_longitude', 
                           'dropoff_latitude','dropoff_longitude']]

location_subset
# --------------------
# boxplots for location based columns

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=2, figsize=(12,5))
for i in range(4):
    if i <= 1:
        # plot the boxplot
        sns.boxplot(y=location_subset.iloc[:,i].compute(),ax=ax1[i])
    else:
        # plot the boxplot
        sns.boxplot(y=location_subset.iloc[:,i].compute(),ax=ax2[i - 2])
# --------------------
# create new columns

df_final["pickup_months"] = df_final["tpep_pickup_datetime"].dt.month
df_final["pickup_day_of_week"] = df_final["tpep_pickup_datetime"].dt.dayofweek
df_final["pickup_hour"] = df_final["tpep_pickup_datetime"].dt.hour
# --------------------
# Use cleaned data for all downstream EDA
# ...existing code...
clean_df["pickup_months"] = clean_df["tpep_pickup_datetime"].dt.month
clean_df["pickup_day_of_week"] = clean_df["tpep_pickup_datetime"].dt.dayofweek
clean_df["pickup_hour"] = clean_df["tpep_pickup_datetime"].dt.hour
# ...existing code...
# --------------------
# ...existing code...

# Build zone_id safely
zone_df = clean_df[
    ["tpep_pickup_datetime", "pickup_latitude", "pickup_longitude",
     "trip_distance", "fare_amount", "trip_duration_min"]
].copy()

zone_df["lat_bin"] = zone_df["pickup_latitude"].round(2)
zone_df["lon_bin"] = zone_df["pickup_longitude"].round(2)
zone_df["zone_id"] = zone_df["lat_bin"].astype("string") + "_" + zone_df["lon_bin"].astype("string")

# Aggregate 1: means
zone_stats_pd = (
    zone_df.groupby("zone_id")
    .agg({
        "trip_distance": "mean",
        "fare_amount": "mean",
        "trip_duration_min": "mean"
    })
    .rename(columns={
        "trip_distance": "zone_avg_trip_distance",
        "fare_amount": "zone_avg_fare",
        "trip_duration_min": "zone_avg_duration_min"
    })
    .compute()
    .reset_index()
)

# Aggregate 2: counts
zone_counts_pd = (
    zone_df.groupby("zone_id")
    .size()
    .compute()
    .reset_index(name="zone_total_pickups")
)

# Merge in pandas (stable)
zone_features = zone_stats_pd.merge(zone_counts_pd, on="zone_id", how="left")

zone_features.sort_values("zone_total_pickups", ascending=False).head(20)
# ...existing code...
# --------------------
# each zone’s peak pickup hour

# What: find the busiest hour for each zone
# Why: captures zone-specific temporal behavior (business vs residential vs transit patterns)

zone_hour_pd = (
    zone_df.assign(pickup_hour=zone_df["tpep_pickup_datetime"].dt.hour)
    .groupby(["zone_id", "pickup_hour"])
    .size()
    .compute()
    .reset_index(name="pickup_count")
)

zone_peak_hour = (
    zone_hour_pd.sort_values(["zone_id", "pickup_count"], ascending=[True, False])
    .drop_duplicates("zone_id")
    .rename(columns={"pickup_hour": "zone_peak_hour"})
    [["zone_id", "zone_peak_hour"]]
)

zone_features = zone_features.merge(zone_peak_hour, on="zone_id", how="left")
zone_features.head()
# --------------------
# ...existing code...
# Step 1: Keep all useful columns for hidden-pattern analysis.
# (Do NOT drop trip_distance, fare_amount, dropoff coordinates yet.)

# old (disable/remove):
# df_final = df_final.drop(columns=['trip_distance', 'dropoff_longitude', 'dropoff_latitude', 'fare_amount'])

print("Current columns:", list(df_final.columns))
# --------------------
# set the values of coordinates

min_latitude = 40.60
max_latitude = 40.85

min_longitude = -74.05
max_longitude = -73.70
# --------------------
# select data points within the given ranges

df_final = df_final.loc[(df_final["pickup_latitude"].between(min_latitude, max_latitude, inclusive="both")) & 
(df_final["pickup_longitude"].between(min_longitude, max_longitude, inclusive="both")) & 
(df_final["dropoff_latitude"].between(min_latitude, max_latitude, inclusive="both")) & 
(df_final["dropoff_longitude"].between(min_longitude, max_longitude, inclusive="both")), :]
# --------------------
df_final
# --------------------
# make a subset of location based columns

location_subset = df_final[['pickup_latitude','pickup_longitude', 
                           'dropoff_latitude','dropoff_longitude']]

location_subset
# --------------------
# boxplots for location based columns

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=2, figsize=(12,5))
for i in range(4):
    if i <= 1:
        # plot the boxplot
        sns.boxplot(y=location_subset.iloc[:,i].compute(),ax=ax1[i],whis=3)
    else:
        # plot the boxplot
        sns.boxplot(y=location_subset.iloc[:,i].compute(),ax=ax2[i - 2],whis=3)
# --------------------
# boxplot for the trip distance

sns.boxplot(df_final.loc[:,"trip_distance"].compute(), whis=3)
# --------------------
# boxplot for the fare amount

sns.boxplot(df_final.loc[:,"fare_amount"].compute(), whis=3)
# --------------------
sns.boxplot(df_final.loc[:,"trip_duration_min"].compute(), whis=3)
# --------------------

--- notebooks\3.Breaking_NYC_to_Regions.ipynb ---
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# --------------------
# Project paths
project_root = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()

data_interim = project_root / "data" / "interim"
data_processed = project_root / "data" / "processed"
models_dir = project_root / "models"

for folder in [data_interim, data_processed, models_dir]:
    folder.mkdir(parents=True, exist_ok=True)

candidate_inputs = [
    data_interim / "location_data.csv",
    data_interim / "cleaned_data.csv",
    data_interim / "cleaned_data.parquet",
    data_processed / "cleaned_data.csv",
    data_processed / "cleaned_data.parquet",
]

input_path = next((path for path in candidate_inputs if path.exists()), None)

if input_path is None:
    raise FileNotFoundError(
        "No cleaned input file found. Expected one of: "
        + ", ".join(str(path) for path in candidate_inputs)
    )

print(f"Using input file: {input_path}")

# --------------------
NYC_BOUNDS = {
    "min_lat": 40.55,
    "max_lat": 40.95,
    "min_lon": -74.10,
    "max_lon": -73.65,
}


def coordinate_chunk_reader(path: Path, chunksize: int = 250_000):
    cols = ["pickup_latitude", "pickup_longitude"]

    if path.suffix.lower() == ".csv":
        for chunk in pd.read_csv(path, usecols=lambda c: c in cols, chunksize=chunksize):
            yield chunk
    else:
        parquet_df = pd.read_parquet(path, columns=cols)
        for start in range(0, len(parquet_df), chunksize):
            yield parquet_df.iloc[start:start + chunksize].copy()


def full_chunk_reader(path: Path, columns=None, chunksize: int = 200_000):
    if path.suffix.lower() == ".csv":
        kwargs = {"chunksize": chunksize}
        if columns is not None:
            kwargs["usecols"] = lambda c: c in columns
        for chunk in pd.read_csv(path, **kwargs):
            yield chunk
    else:
        if columns is not None:
            parquet_df = pd.read_parquet(path, columns=columns)
        else:
            parquet_df = pd.read_parquet(path)
        for start in range(0, len(parquet_df), chunksize):
            yield parquet_df.iloc[start:start + chunksize].copy()


def clean_coordinates(frame: pd.DataFrame):
    coords = frame[["pickup_latitude", "pickup_longitude"]].copy()
    coords["pickup_latitude"] = pd.to_numeric(coords["pickup_latitude"], errors="coerce")
    coords["pickup_longitude"] = pd.to_numeric(coords["pickup_longitude"], errors="coerce")

    valid_mask = (
        coords["pickup_latitude"].between(NYC_BOUNDS["min_lat"], NYC_BOUNDS["max_lat"])
        & coords["pickup_longitude"].between(NYC_BOUNDS["min_lon"], NYC_BOUNDS["max_lon"])
    )

    return coords.loc[valid_mask].reset_index(drop=True), valid_mask

# --------------------
# Build a representative sample without loading all rows at once
sample_parts = []
total_valid_points = 0

for chunk in coordinate_chunk_reader(input_path, chunksize=250_000):
    cleaned, _ = clean_coordinates(chunk)
    if cleaned.empty:
        continue

    total_valid_points += len(cleaned)
    take = min(4_000, len(cleaned))
    sample_parts.append(cleaned.sample(n=take, random_state=RANDOM_STATE))

if not sample_parts:
    raise ValueError("No valid pickup coordinate rows found after cleaning.")

sample_coords = pd.concat(sample_parts, ignore_index=True)

if len(sample_coords) > 200_000:
    sample_coords = sample_coords.sample(n=200_000, random_state=RANDOM_STATE).reset_index(drop=True)

print(f"Total valid coordinate rows found: {total_valid_points:,}")
print(f"Sample rows used for k-search and visual checks: {len(sample_coords):,}")
sample_coords.head()

# --------------------
fig, ax = plt.subplots(figsize=(10, 7))
ax.set_facecolor("#0f172a")
ax.scatter(
    sample_coords["pickup_longitude"],
    sample_coords["pickup_latitude"],
    s=1,
    alpha=0.35,
    color="#22d3ee",
)
ax.set_title("Pickup Distribution (Sample)", fontsize=13)
ax.set_xlabel("pickup_longitude")
ax.set_ylabel("pickup_latitude")
plt.show()

# --------------------
def haversine_distance(lat1, lon1, lat2, lon2):
    # Distance in kilometers between two geo points.
    radius_km = 6371.0088

    lat1, lon1, lat2, lon2 = map(
        np.radians,
        [lat1, lon1, lat2, lon2],
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return radius_km * c



def pairwise_haversine_matrix(centroids_lat_lon: np.ndarray) -> np.ndarray:
    lat = np.radians(centroids_lat_lon[:, 0])
    lon = np.radians(centroids_lat_lon[:, 1])

    lat1 = lat[:, None]
    lat2 = lat[None, :]
    lon1 = lon[:, None]
    lon2 = lon[None, :]

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

    radius_km = 6371.0088
    return radius_km * c



def select_optimum_k(
    scaled_features: np.ndarray,
    scaler: StandardScaler,
    k_values,
    neighbor_count: int = 8,
    movement_band_km=(1.0, 1.5),
    metric_sample_size: int = 6000,
    silhouette_sample_size: int = 3000,
):
    rows = []
    band_low, band_high = movement_band_km

    n_rows = len(scaled_features)
    eval_idx = None

    # Use a fixed subset for metric computation to keep runtime stable on Kaggle.
    if n_rows > metric_sample_size:
        rng = np.random.default_rng(RANDOM_STATE)
        eval_idx = rng.choice(n_rows, size=metric_sample_size, replace=False)

    for k in k_values:
        model = MiniBatchKMeans(
            n_clusters=k,
            n_init=10,
            random_state=RANDOM_STATE,
            batch_size=4096,
            reassignment_ratio=0.01,
        )
        labels = model.fit_predict(scaled_features)

        if eval_idx is None:
            X_eval = scaled_features
            y_eval = labels
        else:
            X_eval = scaled_features[eval_idx]
            y_eval = labels[eval_idx]

        # Clustering-quality metrics on sampled feature space.
        row = {
            "k": k,
            "inertia": model.inertia_,
            "silhouette": np.nan,
            "calinski_harabasz": np.nan,
            "davies_bouldin": np.nan,
            "metrics_eval_rows": int(len(X_eval)),
        }

        unique_clusters = np.unique(y_eval)
        if len(unique_clusters) > 1 and len(X_eval) > 2:
            safe_sil_size = min(silhouette_sample_size, len(X_eval))
            row["silhouette"] = silhouette_score(
                X_eval,
                y_eval,
                sample_size=safe_sil_size,
                random_state=RANDOM_STATE,
            )
            row["calinski_harabasz"] = calinski_harabasz_score(X_eval, y_eval)
            row["davies_bouldin"] = davies_bouldin_score(X_eval, y_eval)

        # Operational metric: driver movement distance in KM between regions.
        centroids_lat_lon = scaler.inverse_transform(model.cluster_centers_)
        distance_matrix = pairwise_haversine_matrix(centroids_lat_lon)
        sorted_distances = np.sort(distance_matrix, axis=1)

        selected_distances = sorted_distances[:, 1 : neighbor_count + 1]
        avg_neighbor_distance_km = selected_distances.mean(axis=1)
        movement_fit_mask = (
            (avg_neighbor_distance_km >= band_low) & (avg_neighbor_distance_km <= band_high)
        )

        row["neighbor_count"] = neighbor_count
        row["movement_band_low_km"] = band_low
        row["movement_band_high_km"] = band_high
        row["avg_neighbor_distance_km"] = float(avg_neighbor_distance_km.mean())
        row["movement_fit_count"] = int(movement_fit_mask.sum())
        row["movement_fit_pct"] = float(movement_fit_mask.mean())

        rows.append(row)

    return pd.DataFrame(rows)



def choose_best_k(
    metrics_df: pd.DataFrame,
    movement_fit_threshold: float = 0.60,
    return_details: bool = False,
):
    ranked = metrics_df.dropna(subset=["silhouette", "calinski_harabasz", "davies_bouldin"]).copy()

    if ranked.empty:
        fallback_row = metrics_df.sort_values("movement_fit_pct", ascending=False).iloc[0]
        best_k = int(fallback_row["k"])
        if return_details:
            return best_k, metrics_df.copy(), True
        return best_k

    feasible = ranked[ranked["movement_fit_pct"] >= movement_fit_threshold].copy()
    used_fallback = False

    if feasible.empty:
        # If strict movement threshold is not met, use top movement-fit candidates.
        feasible = ranked.sort_values(["movement_fit_pct", "silhouette"], ascending=[False, False]).head(3)
        used_fallback = True

    feasible = feasible.copy()
    feasible["rank_movement"] = feasible["movement_fit_pct"].rank(ascending=False, method="dense")
    feasible["rank_silhouette"] = feasible["silhouette"].rank(ascending=False, method="dense")
    feasible["rank_calinski"] = feasible["calinski_harabasz"].rank(ascending=False, method="dense")
    feasible["rank_davies"] = feasible["davies_bouldin"].rank(ascending=True, method="dense")

    feasible["rank_total"] = (
        feasible["rank_movement"]
        + feasible["rank_silhouette"]
        + feasible["rank_calinski"]
        + feasible["rank_davies"]
    )

    feasible_sorted = feasible.sort_values(
        ["rank_total", "movement_fit_pct", "silhouette"],
        ascending=[True, False, False],
    )
    best_k = int(feasible_sorted.iloc[0]["k"])

    if return_details:
        return best_k, feasible_sorted, used_fallback
    return best_k

# --------------------
scaler_for_search = StandardScaler()
scaled_sample = scaler_for_search.fit_transform(sample_coords)

k_values = list(range(12, 61, 6))
movement_band_km = (1.6, 2.4)
neighbor_count = 8

# Kaggle-safe metric settings (reduce silhouette compute cost)
metric_sample_size = 6000
silhouette_sample_size = 3000

k_metrics = select_optimum_k(
    scaled_features=scaled_sample,
    scaler=scaler_for_search,
    k_values=k_values,
    neighbor_count=neighbor_count,
    movement_band_km=movement_band_km,
    metric_sample_size=metric_sample_size,
    silhouette_sample_size=silhouette_sample_size,
)
k_metrics

# --------------------
fig, axes = plt.subplots(2, 3, figsize=(18, 9))

sns.lineplot(data=k_metrics, x="k", y="inertia", marker="o", ax=axes[0, 0])
axes[0, 0].set_title("Elbow View (Inertia)")

sns.lineplot(data=k_metrics, x="k", y="silhouette", marker="o", ax=axes[0, 1], color="#16a34a")
axes[0, 1].set_title("Silhouette Score")

sns.lineplot(data=k_metrics, x="k", y="calinski_harabasz", marker="o", ax=axes[0, 2], color="#f97316")
axes[0, 2].set_title("Calinski-Harabasz")

sns.lineplot(data=k_metrics, x="k", y="davies_bouldin", marker="o", ax=axes[1, 0], color="#ef4444")
axes[1, 0].set_title("Davies-Bouldin (Lower is Better)")

sns.lineplot(data=k_metrics, x="k", y="movement_fit_pct", marker="o", ax=axes[1, 1], color="#0ea5e9")
axes[1, 1].set_title(
    f"Movement Fit % (Avg {neighbor_count}-neighbor distance in {movement_band_km[0]}-{movement_band_km[1]} km)"
)
axes[1, 1].set_ylim(0, 1)

sns.lineplot(data=k_metrics, x="k", y="avg_neighbor_distance_km", marker="o", ax=axes[1, 2], color="#7c3aed")
axes[1, 2].axhline(movement_band_km[0], color="gray", linestyle="--", linewidth=1)
axes[1, 2].axhline(movement_band_km[1], color="gray", linestyle="--", linewidth=1)
axes[1, 2].set_title("Mean Neighbor Distance (km)")

plt.tight_layout()
plt.show()

# --------------------
movement_fit_threshold = 0.60
best_k_auto, feasible_ranked, used_fallback = choose_best_k(
    k_metrics,
    movement_fit_threshold=movement_fit_threshold,
    return_details=True,
)

# Final project decision: use K=30 as the operational sweet spot.
manual_k = 30
available_k = set(k_metrics["k"].tolist())

if manual_k in available_k:
    best_k = manual_k
    selection_mode = "manual_override"
else:
    best_k = best_k_auto
    selection_mode = "auto_fallback"

print(f"Movement band (km): {movement_band_km[0]} to {movement_band_km[1]}")
print(f"Movement fit threshold: {movement_fit_threshold:.0%}")
print(f"Auto-selected k: {best_k_auto}")
print(f"Final selected k: {best_k}")
print(f"Selection mode: {selection_mode}")
print(f"Fallback used by auto-selection: {used_fallback}")

k_metrics.sort_values("k")

# --------------------
# Pass 1: fit scaler on all valid coordinates
scaler = StandardScaler()
valid_rows_for_training = 0

for chunk in coordinate_chunk_reader(input_path, chunksize=250_000):
    cleaned, _ = clean_coordinates(chunk)
    if cleaned.empty:
        continue
    scaler.partial_fit(cleaned)
    valid_rows_for_training += len(cleaned)

print(f"Rows used to fit scaler: {valid_rows_for_training:,}")

# Pass 2: fit MiniBatchKMeans incrementally
cluster_model = MiniBatchKMeans(
    n_clusters=best_k,
    n_init=10,
    random_state=RANDOM_STATE,
    batch_size=8192,
    reassignment_ratio=0.01,
    max_no_improvement=20,
)

for chunk in coordinate_chunk_reader(input_path, chunksize=250_000):
    cleaned, _ = clean_coordinates(chunk)
    if cleaned.empty:
        continue

    scaled_chunk = scaler.transform(cleaned)
    cluster_model.partial_fit(scaled_chunk)

print("Final clustering model trained.")

# --------------------
# Cluster centers (inverse transformed to original lat/lon)
centroids_lat_lon = scaler.inverse_transform(cluster_model.cluster_centers_)

# Cluster sizes from full data
cluster_sizes = np.zeros(best_k, dtype=np.int64)

for chunk in coordinate_chunk_reader(input_path, chunksize=250_000):
    cleaned, _ = clean_coordinates(chunk)
    if cleaned.empty:
        continue

    labels = cluster_model.predict(scaler.transform(cleaned))
    cluster_sizes += np.bincount(labels, minlength=best_k)

cluster_summary = pd.DataFrame(
    {
        "region_id": np.arange(best_k, dtype=int),
        "center_latitude": centroids_lat_lon[:, 0],
        "center_longitude": centroids_lat_lon[:, 1],
        "cluster_size": cluster_sizes,
    }
)
cluster_summary["cluster_share"] = cluster_summary["cluster_size"] / cluster_summary["cluster_size"].sum()
cluster_summary = cluster_summary.sort_values("cluster_size", ascending=False).reset_index(drop=True)

cluster_summary.head(10)

# --------------------
# Nearby-region table (supports future relocation recommendations)
distance_matrix = pairwise_haversine_matrix(
    cluster_summary[["center_latitude", "center_longitude"]].to_numpy()
)

nearby_count = 5
neighbor_rows = []

for region_idx in range(distance_matrix.shape[0]):
    distance_row = distance_matrix[region_idx].copy()
    distance_row[region_idx] = np.inf
    nearest_region_ids = np.argsort(distance_row)[:nearby_count]

    for rank, neighbor_id in enumerate(nearest_region_ids, start=1):
        neighbor_rows.append(
            {
                "region_id": int(cluster_summary.iloc[region_idx]["region_id"]),
                "neighbor_region_id": int(cluster_summary.iloc[neighbor_id]["region_id"]),
                "distance_km": float(distance_row[neighbor_id]),
                "neighbor_rank": rank,
            }
        )

region_neighbors = pd.DataFrame(neighbor_rows)
region_neighbors.head(10)

# --------------------
# Label each trip row with region metadata and save as chunked CSV
output_labeled_path = data_interim / "region_labeled_data.csv"
if output_labeled_path.exists():
    output_labeled_path.unlink()

selected_cols = [
    "tpep_pickup_datetime",
    "pickup_latitude",
    "pickup_longitude",
    "trip_distance",
    "fare_amount",
    "tip_amount",
    "total_amount",
    "passenger_count",
]

header = True
total_labeled_rows = 0

for chunk in full_chunk_reader(input_path, columns=selected_cols, chunksize=200_000):
    if not {"pickup_latitude", "pickup_longitude"}.issubset(chunk.columns):
        continue

    cleaned_coords, valid_mask = clean_coordinates(chunk)
    if cleaned_coords.empty:
        continue

    labeled_chunk = chunk.loc[valid_mask].copy().reset_index(drop=True)
    region_ids = cluster_model.predict(scaler.transform(cleaned_coords))

    labeled_chunk["region_id"] = region_ids.astype("int16")
    labeled_chunk["region_label"] = "R" + labeled_chunk["region_id"].astype(str).str.zfill(2)

    if "tpep_pickup_datetime" in labeled_chunk.columns:
        pickup_dt = pd.to_datetime(labeled_chunk["tpep_pickup_datetime"], errors="coerce")
        labeled_chunk["pickup_hour"] = pickup_dt.dt.hour
        labeled_chunk["pickup_day_of_week"] = pickup_dt.dt.dayofweek
        labeled_chunk["is_weekend"] = pickup_dt.dt.dayofweek >= 5

    if {"total_amount", "trip_distance"}.issubset(labeled_chunk.columns):
        labeled_chunk["fare_efficiency"] = (
            labeled_chunk["total_amount"] / labeled_chunk["trip_distance"].clip(lower=1e-3)
        )

    labeled_chunk.to_csv(
        output_labeled_path,
        mode="w" if header else "a",
        header=header,
        index=False,
    )
    header = False
    total_labeled_rows += len(labeled_chunk)

print(f"Rows written with region labels: {total_labeled_rows:,}")
print(f"Saved: {output_labeled_path}")

# --------------------
# Build region-hour demand profile + risk baseline for downstream notebooks
pickup_counts_parts = []
fare_eff_parts = []

for chunk in pd.read_csv(
    output_labeled_path,
    usecols=lambda c: c in ["region_id", "pickup_hour", "fare_efficiency"],
    chunksize=250_000,
):
    if {"region_id", "pickup_hour"}.issubset(chunk.columns):
        hourly = (
            chunk.dropna(subset=["region_id", "pickup_hour"])
            .assign(
                region_id=lambda df_: df_["region_id"].astype(int),
                pickup_hour=lambda df_: df_["pickup_hour"].astype(int),
            )
            .groupby(["region_id", "pickup_hour"], as_index=False)
            .size()
            .rename(columns={"size": "pickup_count"})
        )
        pickup_counts_parts.append(hourly)

    if {"region_id", "fare_efficiency"}.issubset(chunk.columns):
        fare_summary = (
            chunk.dropna(subset=["region_id", "fare_efficiency"])
            .assign(region_id=lambda df_: df_["region_id"].astype(int))
            .groupby("region_id", as_index=False)["fare_efficiency"]
            .agg(["sum", "count"])
            .reset_index()
            .rename(columns={"sum": "fare_eff_sum", "count": "fare_eff_count"})
        )
        fare_eff_parts.append(fare_summary)

region_hour_profile = pd.concat(pickup_counts_parts, ignore_index=True)
region_hour_profile = (
    region_hour_profile.groupby(["region_id", "pickup_hour"], as_index=False)["pickup_count"].sum()
)

region_risk = (
    region_hour_profile.groupby("region_id", as_index=False)["pickup_count"]
    .agg(avg_hourly_pickups="mean", std_hourly_pickups="std")
    .fillna({"std_hourly_pickups": 0})
)
region_risk["risk_score"] = (
    region_risk["std_hourly_pickups"] / (region_risk["avg_hourly_pickups"] + 1e-6)
)
region_risk["risk_band"] = pd.cut(
    region_risk["risk_score"],
    bins=[-np.inf, 0.35, 0.75, np.inf],
    labels=["Stable", "Moderate", "Volatile"],
)

region_hour_profile = region_hour_profile.merge(
    region_risk[["region_id", "avg_hourly_pickups", "std_hourly_pickups"]],
    on="region_id",
    how="left",
)
region_hour_profile["surge_threshold"] = (
    region_hour_profile["avg_hourly_pickups"] + 1.5 * region_hour_profile["std_hourly_pickups"]
)
region_hour_profile["historical_surge_flag"] = (
    region_hour_profile["pickup_count"] > region_hour_profile["surge_threshold"]
)

if fare_eff_parts:
    fare_eff_df = pd.concat(fare_eff_parts, ignore_index=True)
    fare_eff_df = (
        fare_eff_df.groupby("region_id", as_index=False)[["fare_eff_sum", "fare_eff_count"]].sum()
    )
    fare_eff_df["avg_fare_efficiency"] = (
        fare_eff_df["fare_eff_sum"] / fare_eff_df["fare_eff_count"].clip(lower=1)
    )

    region_risk = region_risk.merge(
        fare_eff_df[["region_id", "avg_fare_efficiency"]],
        on="region_id",
        how="left",
    )

# --------------------
# Save all notebook outputs
output_cluster_summary = data_interim / "region_cluster_summary.csv"
output_centroids = data_interim / "region_centroids.csv"
output_neighbors = data_interim / "region_neighbors.csv"
output_hour_profile = data_interim / "region_hour_profile.csv"
output_business_baseline = data_interim / "region_business_baseline.csv"
output_model = models_dir / "region_cluster_pipeline.joblib"

cluster_summary.to_csv(output_cluster_summary, index=False)
cluster_summary[["region_id", "center_latitude", "center_longitude"]].to_csv(
    output_centroids,
    index=False,
)
region_neighbors.to_csv(output_neighbors, index=False)
region_hour_profile.to_csv(output_hour_profile, index=False)
region_risk.to_csv(output_business_baseline, index=False)

joblib.dump(
    {
        "scaler": scaler,
        "cluster_model": cluster_model,
        "k_metrics": k_metrics,
        "selected_k": best_k,
    },
    output_model,
)

print("Saved outputs:")
for path in [
    output_labeled_path,
    output_cluster_summary,
    output_centroids,
    output_neighbors,
    output_hour_profile,
    output_business_baseline,
    output_model,
]:
    print("-", path.relative_to(project_root))

# --------------------
# Visual validation of region assignments
sample_region_ids = cluster_model.predict(scaler.transform(sample_coords))
plot_df = sample_coords.copy()
plot_df["region_id"] = sample_region_ids

fig, ax = plt.subplots(figsize=(10, 7))
ax.set_facecolor("#0f172a")

ax.scatter(
    plot_df["pickup_longitude"],
    plot_df["pickup_latitude"],
    c=plot_df["region_id"],
    cmap="tab20",
    s=2,
    alpha=0.35,
)

ax.scatter(
    cluster_summary["center_longitude"],
    cluster_summary["center_latitude"],
    marker="x",
    c="white",
    s=120,
    linewidths=2,
    label="Centroids",
)

ax.set_title("NYC Regions from MiniBatchKMeans")
ax.set_xlabel("pickup_longitude")
ax.set_ylabel("pickup_latitude")
ax.legend(loc="upper right")
plt.show()

# --------------------

--- notebooks\4.Creating_Historical_Data.ipynb ---
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

sns.set_theme(style="whitegrid")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# --------------------
project_root = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()

data_interim = project_root / "data" / "interim"
data_processed = project_root / "data" / "processed"

for folder in [data_interim, data_processed]:
    folder.mkdir(parents=True, exist_ok=True)

candidate_inputs = [
    data_interim / "region_labeled_data.csv",
    data_interim / "time_series.csv",
    data_interim / "processing_data.csv",
]

input_path = next((p for p in candidate_inputs if p.exists()), None)
if input_path is None:
    raise FileNotFoundError(
        "Could not find historical input. Expected one of: " + ", ".join(str(p) for p in candidate_inputs)
    )

neighbors_path = data_interim / "region_neighbors.csv"

print(f"Using input data: {input_path}")
print(f"Using neighbors file: {neighbors_path} (exists={neighbors_path.exists()})")

# --------------------
# Load dataset
df = pd.read_csv(input_path, low_memory=False)

# Normalize region column name
if "region" in df.columns:
    df["region"] = pd.to_numeric(df["region"], errors="coerce")
elif "region_id" in df.columns:
    df["region"] = pd.to_numeric(df["region_id"], errors="coerce")
else:
    raise ValueError("Missing region column. Expected `region` or `region_id`.")

# Parse datetime
if "tpep_pickup_datetime" not in df.columns:
    raise ValueError("Missing `tpep_pickup_datetime` column in input data.")

df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"], errors="coerce")

df = df.dropna(subset=["tpep_pickup_datetime", "region"]).copy()
df["region"] = df["region"].astype(int)
df = df.sort_values("tpep_pickup_datetime").reset_index(drop=True)

# Revenue helper if not already present
if "fare_efficiency" not in df.columns and {"total_amount", "trip_distance"}.issubset(df.columns):
    df["fare_efficiency"] = df["total_amount"] / df["trip_distance"].clip(lower=1e-3)

df[["tpep_pickup_datetime", "region"]].head()

# --------------------
# Build 15-minute slots
df["pickup_slot"] = df["tpep_pickup_datetime"].dt.floor("15min")

# Base demand count per region/time-slot
grouped_counts = (
    df.groupby(["region", "pickup_slot"], as_index=False)
    .size()
    .rename(columns={"size": "total_pickups_raw"})
)

# Optional aggregates from available columns
metric_agg = {}
if "total_amount" in df.columns:
    metric_agg["total_amount"] = "sum"
if "tip_amount" in df.columns:
    metric_agg["tip_amount"] = "sum"
if "trip_distance" in df.columns:
    metric_agg["trip_distance"] = "sum"
if "trip_duration_min" in df.columns:
    metric_agg["trip_duration_min"] = "mean"
if "passenger_count" in df.columns:
    metric_agg["passenger_count"] = "mean"
if "fare_efficiency" in df.columns:
    metric_agg["fare_efficiency"] = "mean"

if metric_agg:
    grouped_metrics = df.groupby(["region", "pickup_slot"], as_index=False).agg(metric_agg)
    rename_map = {
        "total_amount": "total_revenue",
        "tip_amount": "total_tip",
        "trip_distance": "total_trip_distance",
        "trip_duration_min": "avg_trip_duration_min",
        "passenger_count": "avg_passenger_count",
        "fare_efficiency": "avg_fare_efficiency",
    }
    grouped_metrics = grouped_metrics.rename(columns=rename_map)
    historical = grouped_counts.merge(grouped_metrics, on=["region", "pickup_slot"], how="left")
else:
    historical = grouped_counts.copy()

# Build full region-time grid so missing intervals are explicit
all_regions = np.sort(historical["region"].unique())
all_slots = pd.date_range(historical["pickup_slot"].min(), historical["pickup_slot"].max(), freq="15min")

full_index = pd.MultiIndex.from_product([all_regions, all_slots], names=["region", "pickup_slot"])
historical = historical.set_index(["region", "pickup_slot"]).reindex(full_index).reset_index()

# Fill count/sum columns with 0
zero_fill_cols = [
    "total_pickups_raw",
    "total_revenue",
    "total_tip",
    "total_trip_distance",
]
for col in zero_fill_cols:
    if col in historical.columns:
        historical[col] = historical[col].fillna(0)

# Fill mean-style columns by region ffill/bfill fallback
mean_like_cols = [
    "avg_trip_duration_min",
    "avg_passenger_count",
    "avg_fare_efficiency",
]
for col in mean_like_cols:
    if col in historical.columns:
        historical[col] = historical.groupby("region")[col].transform(lambda s: s.ffill().bfill())
        historical[col] = historical[col].fillna(historical[col].median())

historical = historical.sort_values(["region", "pickup_slot"]).reset_index(drop=True)
historical.head()

# --------------------
# Keep raw pickups and create model-safe demand (for MAPE stability)
epsilon_val = 10
historical["total_pickups_model"] = historical["total_pickups_raw"].replace(0, epsilon_val)

# Candidate grids for smoothing tuning
ma_windows = [3, 4, 6, 8, 12, 16]
ewma_alphas = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Validation split per region: last 20% (with minimum tail)
tuning_df = historical[["region", "pickup_slot", "total_pickups_model"]].copy()
tuning_df = tuning_df.sort_values(["region", "pickup_slot"]).reset_index(drop=True)
tuning_df["row_num"] = tuning_df.groupby("region").cumcount()
tuning_df["region_size"] = tuning_df.groupby("region")["total_pickups_model"].transform("size")

min_val_points = 24  # 24 points = 6 hours at 15-minute granularity
val_start_idx = np.maximum(
    (tuning_df["region_size"] * 0.8).astype(int),
    tuning_df["region_size"] - min_val_points,
)
tuning_df["is_validation"] = tuning_df["row_num"] >= val_start_idx


def smape(y_true, y_pred):
    denom = np.abs(y_true) + np.abs(y_pred)
    return np.mean(2.0 * np.abs(y_true - y_pred) / np.where(denom == 0, 1.0, denom))


ma_results = []
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
            "param": window,
            "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "smape": float(smape(y_true, y_pred)),
            "eval_points": int(val_mask.sum()),
        }
    )


ewm_results = []
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
            "param": alpha,
            "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "smape": float(smape(y_true, y_pred)),
            "eval_points": int(val_mask.sum()),
        }
    )

ma_metrics = pd.DataFrame(ma_results).sort_values(["mape", "mae", "rmse"]).reset_index(drop=True)
ewma_metrics = pd.DataFrame(ewm_results).sort_values(["mape", "mae", "rmse"]).reset_index(drop=True)

best_ma_row = ma_metrics.iloc[0]
best_ewma_row = ewma_metrics.iloc[0]

best_ma_window = int(best_ma_row["param"])
best_ewma_alpha = float(best_ewma_row["param"])

if best_ewma_row["mape"] <= best_ma_row["mape"]:
    selected_smoothing_method = "ewma"
else:
    selected_smoothing_method = "moving_average"

# Build tuned smoothed series
historical["avg_pickups_ma_tuned"] = historical.groupby("region")["total_pickups_model"].transform(
    lambda s: s.rolling(window=best_ma_window, min_periods=1).mean()
)
historical["avg_pickups_ewm_tuned"] = historical.groupby("region")["total_pickups_model"].transform(
    lambda s: s.ewm(alpha=best_ewma_alpha, adjust=False).mean()
)

# One-step-ahead proxy (no leakage)
historical["predicted_demand_proxy_ma"] = historical.groupby("region")["avg_pickups_ma_tuned"].shift(1)
historical["predicted_demand_proxy_ewm"] = historical.groupby("region")["avg_pickups_ewm_tuned"].shift(1)

historical["predicted_demand_proxy_ma"] = historical["predicted_demand_proxy_ma"].fillna(historical["avg_pickups_ma_tuned"])
historical["predicted_demand_proxy_ewm"] = historical["predicted_demand_proxy_ewm"].fillna(historical["avg_pickups_ewm_tuned"])

if selected_smoothing_method == "ewma":
    historical["predicted_demand_proxy"] = historical["predicted_demand_proxy_ewm"]
else:
    historical["predicted_demand_proxy"] = historical["predicted_demand_proxy_ma"]

historical["predicted_demand"] = historical["predicted_demand_proxy"].clip(lower=0)

# Keep compatibility columns
historical["avg_pickups_ewm"] = historical["avg_pickups_ewm_tuned"]
historical["avg_pickups"] = historical["predicted_demand_proxy"]
historical["smoothing_method"] = selected_smoothing_method
historical["selected_ma_window"] = best_ma_window
historical["selected_ewma_alpha"] = best_ewma_alpha

# Time flags for downstream modeling and dashboard filters
historical["pickup_day_of_week"] = historical["pickup_slot"].dt.dayofweek
historical["pickup_hour"] = historical["pickup_slot"].dt.hour
historical["is_weekend"] = historical["pickup_day_of_week"] >= 5
historical["rush_hour"] = historical["pickup_hour"].isin([7, 8, 9, 17, 18, 19])
historical["is_night"] = (historical["pickup_hour"] >= 23) | (historical["pickup_hour"] <= 4)

smoothing_metrics = pd.concat([ma_metrics, ewma_metrics], ignore_index=True)

print("Best MA window:", best_ma_window, "| MAPE:", round(float(best_ma_row["mape"]), 4))
print("Best EWMA alpha:", best_ewma_alpha, "| MAPE:", round(float(best_ewma_row["mape"]), 4))
print("Selected smoothing method:", selected_smoothing_method)

# --------------------
# Surge detection + risk/stability score
rolling_window = 96  # 96 * 15min = 24 hours
surge_k = 1.5

historical["rolling_mean"] = (
    historical.groupby("region")["total_pickups_model"]
    .transform(lambda s: s.rolling(window=rolling_window, min_periods=16).mean())
)
historical["rolling_std"] = (
    historical.groupby("region")["total_pickups_model"]
    .transform(lambda s: s.rolling(window=rolling_window, min_periods=16).std())
)

historical["rolling_mean"] = historical["rolling_mean"].fillna(historical["avg_pickups_ewm"])
historical["rolling_std"] = historical["rolling_std"].fillna(0)

historical["surge_threshold"] = historical["rolling_mean"] + surge_k * historical["rolling_std"]
historical["surge_flag"] = historical["predicted_demand"] > historical["surge_threshold"]

historical["surge_score"] = (
    (historical["predicted_demand"] - historical["rolling_mean"])
    / (historical["rolling_std"] + 1e-6)
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

# --------------------
# Revenue, efficiency, congestion, and pressure features
if "total_revenue" in historical.columns:
    historical["avg_fare_region_slot"] = historical["total_revenue"] / historical["total_pickups_raw"].clip(lower=1)
    region_fare_baseline = historical.groupby("region")["avg_fare_region_slot"].transform("mean")
    historical["avg_fare_region_slot"] = historical["avg_fare_region_slot"].fillna(region_fare_baseline)
    historical["avg_fare_region_slot"] = historical["avg_fare_region_slot"].fillna(historical["avg_fare_region_slot"].median())
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

if "total_revenue" in historical.columns:
    historical["revenue_density_15min"] = historical["total_revenue"]
else:
    historical["revenue_density_15min"] = np.nan

region_avg_proxy = historical.groupby("region")["predicted_demand"].transform("mean")
historical["demand_pressure"] = historical["predicted_demand"] / (region_avg_proxy + 1e-6)

# Traffic / speed proxy from available aggregates
if {"total_trip_distance", "avg_trip_duration_min", "total_pickups_raw"}.issubset(historical.columns):
    historical["avg_trip_distance_per_ride"] = (
        historical["total_trip_distance"] / historical["total_pickups_raw"].clip(lower=1)
    )
    historical["avg_speed_kmh"] = (
        historical["avg_trip_distance_per_ride"]
        / (historical["avg_trip_duration_min"].clip(lower=1e-3) / 60.0)
    )
    historical["congestion_band"] = pd.cut(
        historical["avg_speed_kmh"],
        bins=[-np.inf, 12, 22, np.inf],
        labels=["High Congestion", "Moderate", "Low Congestion"],
    ).astype(str)
else:
    historical["avg_speed_kmh"] = np.nan
    historical["congestion_band"] = "unknown"

# --------------------
# Driver allocation optimization (nearby region demand gain / distance)
if neighbors_path.exists():
    neighbors = pd.read_csv(neighbors_path)

    neighbors = neighbors.rename(
        columns={
            "region_id": "region",
            "neighbor_region_id": "target_region",
        }
    )

    required_neighbor_cols = {"region", "target_region", "distance_km"}
    if not required_neighbor_cols.issubset(neighbors.columns):
        raise ValueError(f"Neighbor file missing columns: {required_neighbor_cols}")

    base = historical[["pickup_slot", "region", "predicted_demand"]].copy()

    candidate_moves = base.merge(neighbors[["region", "target_region", "distance_km"]], on="region", how="left")

    target_demand = base.rename(
        columns={
            "region": "target_region",
            "predicted_demand": "target_predicted_demand",
        }
    )

    candidate_moves = candidate_moves.merge(
        target_demand,
        on=["pickup_slot", "target_region"],
        how="left",
    )

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

    # If gain is too small, recommend staying in current region.
    min_gain_threshold = 1.0
    low_gain_mask = best_moves["expected_demand_gain"] < min_gain_threshold
    best_moves.loc[low_gain_mask, "target_region"] = np.nan
    best_moves.loc[low_gain_mask, "distance_km"] = np.nan
    best_moves.loc[low_gain_mask, "expected_demand_gain"] = 0.0
    best_moves.loc[low_gain_mask, "relocation_score"] = 0.0

    best_moves = best_moves.rename(
        columns={
            "target_region": "recommended_next_zone",
            "distance_km": "recommended_distance_km",
        }
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
    best_moves = pd.DataFrame()
    historical["recommended_next_zone"] = pd.Series([pd.NA] * len(historical), dtype="Int64")
    historical["recommended_distance_km"] = np.nan
    historical["expected_demand_gain"] = 0.0
    historical["relocation_score"] = 0.0

# --------------------
# Best-time recommendation by region (historical slot profile)
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

day_names = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday",
}

best_time_recommendations["day_name"] = best_time_recommendations["pickup_day_of_week"].map(day_names)
best_time_recommendations["best_time_window"] = (
    best_time_recommendations["pickup_hour"].astype(int).map(lambda h: f"{h:02d}:00-{(h + 1) % 24:02d}:00")
)

# Add top recommendation per region into main historical table
region_best_time = (
    best_time_recommendations[best_time_recommendations["rank"] == 1]
    [["region", "day_name", "best_time_window"]]
    .rename(columns={"day_name": "best_day_name"})
)

historical = historical.merge(region_best_time, on="region", how="left")

best_time_recommendations.head(10)

# --------------------
# Save outputs
historical_output = data_interim / "historical_features.csv"
final_data_output = data_interim / "final_data.csv"  # compatibility with older flow
relocation_output = data_interim / "driver_relocation_recommendations.csv"
best_time_output = data_interim / "best_time_recommendations.csv"
slot_profile_output = data_interim / "region_slot_profile.csv"
ui_output = data_interim / "ui_ready_timeslot_output.csv"
smoothing_metrics_output = data_interim / "smoothing_tuning_metrics.csv"

historical.to_csv(historical_output, index=False)

final_data = historical.rename(columns={"pickup_slot": "tpep_pickup_datetime"}).copy()
final_data.to_csv(final_data_output, index=False)

if not top_moves.empty:
    top_moves.to_csv(relocation_output, index=False)
else:
    pd.DataFrame(columns=[
        "pickup_slot",
        "region",
        "target_region",
        "distance_km",
        "target_predicted_demand",
        "expected_demand_gain",
        "relocation_score",
        "recommendation_rank",
    ]).to_csv(relocation_output, index=False)

best_time_recommendations.to_csv(best_time_output, index=False)
slot_profile.to_csv(slot_profile_output, index=False)
smoothing_metrics.to_csv(smoothing_metrics_output, index=False)

ui_cols = [
    "pickup_slot",
    "region",
    "predicted_demand",
    "surge_flag",
    "surge_level",
    "risk_score",
    "risk_band",
    "expected_revenue",
    "recommended_next_zone",
    "recommended_distance_km",
    "relocation_score",
    "demand_pressure",
    "best_day_name",
    "best_time_window",
    "smoothing_method",
    "selected_ma_window",
    "selected_ewma_alpha",
]
ui_ready = historical[[col for col in ui_cols if col in historical.columns]].copy()
ui_ready.to_csv(ui_output, index=False)

print("Saved files:")
for path in [
    historical_output,
    final_data_output,
    relocation_output,
    best_time_output,
    slot_profile_output,
    ui_output,
    smoothing_metrics_output,
]:
    print("-", path.relative_to(project_root))

# --------------------
# Quick validation checks
key_cols = [
    "pickup_slot",
    "region",
    "total_pickups_raw",
    "predicted_demand",
    "avg_pickups",
    "smoothing_method",
    "selected_ma_window",
    "selected_ewma_alpha",
    "surge_flag",
    "surge_level",
    "risk_score",
    "risk_band",
    "expected_revenue",
    "recommended_next_zone",
    "relocation_score",
    "demand_pressure",
    "best_time_window",
]

available_cols = [col for col in key_cols if col in historical.columns]
historical[available_cols].head(10)

# --------------------
# High-level stats
print("Rows:", f"{len(historical):,}")
print("Regions:", historical["region"].nunique())
print("Surge rows:", int(historical["surge_flag"].sum()))
print("Average risk score:", round(float(historical["risk_score"].mean()), 4))
print("Average expected revenue:", round(float(historical["expected_revenue"].fillna(0).mean()), 2))
print("Rows with relocation recommendation:", int(historical["recommended_next_zone"].notna().sum()))
print("Selected smoothing method:", selected_smoothing_method)
print("Best MA window:", best_ma_window, "| Best MA MAPE:", round(float(best_ma_row["mape"]), 4))
print("Best EWMA alpha:", best_ewma_alpha, "| Best EWMA MAPE:", round(float(best_ewma_row["mape"]), 4))

# --------------------

--- notebooks\5.Model_selection.ipynb ---
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
)

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    import mlflow
    HAS_MLFLOW = True
except Exception:
    HAS_MLFLOW = False

try:
    import dagshub
    HAS_DAGSHUB = True
except Exception:
    HAS_DAGSHUB = False

import optuna

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# --------------------
project_root = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()

data_interim = project_root / "data" / "interim"
data_processed = project_root / "data" / "processed"
models_dir = project_root / "models"
reports_dir = project_root / "reports"

for folder in [data_interim, data_processed, models_dir, reports_dir]:
    folder.mkdir(parents=True, exist_ok=True)

# Tracking flags (safe defaults for Kaggle/local)
USE_MLFLOW = False
USE_DAGSHUB = False

if USE_MLFLOW and HAS_MLFLOW:
    mlflow.set_experiment("Model Selection")
    if USE_DAGSHUB and HAS_DAGSHUB:
        # Uncomment and update if you want DagsHub tracking
        # dagshub.init(repo_owner="<owner>", repo_name="<repo>", mlflow=True)
        pass

# --------------------
def smape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.abs(y_true) + np.abs(y_pred)
    return np.mean(2.0 * np.abs(y_true - y_pred) / np.where(denom == 0, 1.0, denom))


def safe_mape(y_true, y_pred, eps: float = 1.0):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(np.abs(y_true - y_pred) / np.maximum(np.abs(y_true), eps))


def evaluate_metrics(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE": float(safe_mape(y_true, y_pred)),
        "sMAPE": float(smape(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def safe_fill_frame(df: pd.DataFrame):
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


def get_time_col(df: pd.DataFrame):
    for c in ["pickup_slot", "tpep_pickup_datetime", "timestamp"]:
        if c in df.columns:
            return c
    return None

# --------------------
# Load data for model selection (prefer fresh historical split to avoid stale leakage)
train_path = data_processed / "train.csv"
test_path = data_processed / "test.csv"

local_hist_candidates = [
    data_interim / "historical_features.csv",
    data_interim / "final_data.csv",
]
kaggle_hist_candidates = [
    Path("/kaggle/working/historical_features.csv"),
    Path("/kaggle/working/final_data.csv"),
    Path("/kaggle/input/notebooks/shubhamyadav74/notebookacca71d522/historical_features.csv"),
    Path("/kaggle/input/notebooks/shubhamyadav74/notebookacca71d522/final_data.csv"),
]
hist_candidates = local_hist_candidates + kaggle_hist_candidates

FORCE_REBUILD_SPLIT = True

hist_path = next((p for p in hist_candidates if p.exists()), None)

if FORCE_REBUILD_SPLIT and hist_path is not None:
    all_df = pd.read_csv(hist_path)
    time_col = get_time_col(all_df)
    if time_col is None:
        raise ValueError("Historical data missing time column (`pickup_slot` or `tpep_pickup_datetime`).")

    all_df[time_col] = pd.to_datetime(all_df[time_col], errors="coerce")
    all_df = all_df.dropna(subset=[time_col]).copy()

    if "region" in all_df.columns:
        all_df["region"] = pd.to_numeric(all_df["region"], errors="coerce")
        all_df = all_df.dropna(subset=["region"]).copy()
        all_df["region"] = all_df["region"].astype(int)
        all_df = all_df.sort_values([time_col, "region"]).reset_index(drop=True)
    else:
        all_df = all_df.sort_values(time_col).reset_index(drop=True)

    unique_times = np.sort(all_df[time_col].unique())
    if len(unique_times) < 2:
        raise ValueError("Not enough unique timestamps to create a chronological train/test split.")

    cut_idx = max(1, int(len(unique_times) * 0.8))
    cut_idx = min(cut_idx, len(unique_times) - 1)
    cutoff_time = unique_times[cut_idx - 1]

    train_df = all_df[all_df[time_col] <= cutoff_time].copy()
    test_df = all_df[all_df[time_col] > cutoff_time].copy()

    source_mode = f"historical_time_split ({Path(hist_path).name})"

    # Save split for reproducibility in current run environment.
    try:
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
    except Exception:
        pass

elif train_path.exists() and test_path.exists():
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    source_mode = "processed_train_test"
else:
    raise FileNotFoundError(
        "No historical features found and no train/test split available. Run Notebook 4 first."
    )

print("Source mode:", source_mode)
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

_load_time_col = get_time_col(train_df)
if _load_time_col is not None and _load_time_col in test_df.columns:
    print("Train range:", pd.to_datetime(train_df[_load_time_col]).min(), "->", pd.to_datetime(train_df[_load_time_col]).max())
    print("Test range :", pd.to_datetime(test_df[_load_time_col]).min(), "->", pd.to_datetime(test_df[_load_time_col]).max())


# --------------------
# Target and feature preparation + strict no-leak lag engineering
# IMPORTANT: model-selection must use raw demand target, not smoothed/model target.
LEAKAGE_GUARD_VERSION = "v2_2026-04-14"
print("Leakage guard:", LEAKAGE_GUARD_VERSION)

target_candidates = ["total_pickups_raw", "total_pickups"]
target_col = next((c for c in target_candidates if c in train_df.columns), None)

if target_col is None:
    if "total_pickups_model" in train_df.columns:
        raise ValueError(
            "Leakage-safe target not found. `total_pickups_model` is smoothed and should not be used for model selection. "
            "Use Notebook 4 output that includes `total_pickups_raw` (or `total_pickups`)."
        )
    raise ValueError(f"Target column missing. Expected one of: {target_candidates}")

time_col = get_time_col(train_df)
if time_col is None:
    raise ValueError("Time column is required for lag feature creation (`pickup_slot` or `tpep_pickup_datetime`).")

# Keep these only for post-prediction business outputs (not model fitting)
context_keep_cols = [
    c
    for c in [
        time_col,
        "region",
        "rolling_mean",
        "rolling_std",
        "avg_fare_region_slot",
        "avg_pickups",
        "avg_pickups_ewm",
    ]
    if c in test_df.columns
]

# 15-min lag setup
lag_steps = [1, 2, 3, 6, 12, 96]
rolling_windows = [3, 6]


def add_lag_columns(df: pd.DataFrame, target: str, t_col: str):
    out = df.copy()
    out[t_col] = pd.to_datetime(out[t_col], errors="coerce")
    out = out.dropna(subset=[t_col]).copy()

    if "region" in out.columns:
        out["region"] = pd.to_numeric(out["region"], errors="coerce")
        out = out.dropna(subset=["region"]).copy()
        out["region"] = out["region"].astype(int)
        out = out.sort_values(["region", t_col]).reset_index(drop=True)

        for lag in lag_steps:
            out[f"lag_{lag}"] = out.groupby("region")[target].shift(lag)

        for w in rolling_windows:
            out[f"lag_roll_mean_{w}"] = out.groupby("region")[target].transform(
                lambda s: s.shift(1).rolling(window=w, min_periods=w).mean()
            )
            out[f"lag_roll_std_{w}"] = out.groupby("region")[target].transform(
                lambda s: s.shift(1).rolling(window=w, min_periods=w).std()
            )
    else:
        out = out.sort_values([t_col]).reset_index(drop=True)

        for lag in lag_steps:
            out[f"lag_{lag}"] = out[target].shift(lag)

        for w in rolling_windows:
            out[f"lag_roll_mean_{w}"] = out[target].shift(1).rolling(window=w, min_periods=w).mean()
            out[f"lag_roll_std_{w}"] = out[target].shift(1).rolling(window=w, min_periods=w).std()

    lag_cols = [c for c in out.columns if c.startswith("lag_")]
    return out, lag_cols


# Keep raw copies for leakage-safe test lag creation.
train_source = train_df.copy()
test_source = test_df.copy()

# Train lag features
train_with_lags, lag_cols_train = add_lag_columns(train_source, target_col, time_col)
before_train = len(train_with_lags)
train_df = train_with_lags.dropna(subset=lag_cols_train).reset_index(drop=True)
dropped_train = before_train - len(train_df)

# Test lag features using trailing TRAIN history (region-wise)
max_hist = max(lag_steps + rolling_windows) + 2
if "region" in train_source.columns:
    train_hist_base = train_source.copy()
    train_hist_base[time_col] = pd.to_datetime(train_hist_base[time_col], errors="coerce")
    train_hist_base["region"] = pd.to_numeric(train_hist_base["region"], errors="coerce")
    train_hist_base = train_hist_base.dropna(subset=[time_col, "region"]).copy()
    train_hist_base["region"] = train_hist_base["region"].astype(int)
    train_hist_base = train_hist_base.sort_values(["region", time_col]).reset_index(drop=True)
    history = train_hist_base.groupby("region", group_keys=False).tail(max_hist)
else:
    train_hist_base = train_source.copy()
    train_hist_base[time_col] = pd.to_datetime(train_hist_base[time_col], errors="coerce")
    train_hist_base = train_hist_base.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    history = train_hist_base.tail(max_hist)

history = history.copy()
history["_is_test"] = 0

test_tagged = test_source.copy()
test_tagged["_is_test"] = 1

test_temp = pd.concat([history, test_tagged], axis=0, ignore_index=True)
test_temp_with_lags, lag_cols_test = add_lag_columns(test_temp, target_col, time_col)

test_df = test_temp_with_lags[test_temp_with_lags["_is_test"] == 1].copy()
test_df = test_df.drop(columns=["_is_test"], errors="ignore")

lag_feature_cols = sorted(list(set(lag_cols_train).intersection(set(lag_cols_test))))
if not lag_feature_cols:
    raise ValueError("No common lag features found between train and test after lag processing.")

before_test = len(test_df)
test_df = test_df.dropna(subset=lag_feature_cols).reset_index(drop=True)
dropped_test = before_test - len(test_df)

# Strict whitelist only
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

feature_cols = [c for c in allowed_time_cols if c in train_df.columns and c in test_df.columns]
feature_cols += [c for c in lag_feature_cols if c in train_df.columns and c in test_df.columns]

if not feature_cols:
    raise ValueError("No features available after strict allowlist filtering.")

# Hard safety: ensure forbidden columns never enter model
forbidden_prefixes = ("predicted_", "rolling_", "surge_", "risk_", "expected_")
forbidden_exact = {
    "total_revenue", "total_tip", "total_trip_distance", "avg_pickups", "avg_pickups_ewm",
    "avg_pickups_ewm_tuned", "avg_pickups_ma_tuned", "fare_per_km", "tip_ratio", "tip_per_pickup",
    "revenue_density_15min", "avg_fare_region_slot", "avg_speed_kmh", "demand_pressure",
    "total_pickups_model", target_col,
}

bad_features = [
    c for c in feature_cols
    if c in forbidden_exact or any(c.lower().startswith(p) for p in forbidden_prefixes)
]
if bad_features:
    raise ValueError(f"Leakage detected in feature list: {bad_features}")

X_train = train_df[feature_cols].copy()
y_train = train_df[target_col].copy()

X_test = test_df[feature_cols].copy()
y_test = test_df[target_col].copy()

context_test = test_df[context_keep_cols].copy() if context_keep_cols else pd.DataFrame(index=test_df.index)

X_train = safe_fill_frame(X_train)
X_test = safe_fill_frame(X_test)

# Additional hard leakage check: no feature should be identical to target.
exact_match_cols = []
for c in X_train.columns:
    if pd.api.types.is_numeric_dtype(X_train[c]):
        match_ratio = np.mean(np.isclose(X_train[c].to_numpy(), y_train.to_numpy(), rtol=0, atol=1e-12))
        if match_ratio > 0.999:
            exact_match_cols.append((c, float(match_ratio)))

if exact_match_cols:
    raise ValueError(f"Leakage detected: feature(s) nearly identical to target -> {exact_match_cols}")

# Dtypes
cat_cols = [
    c for c in X_train.columns
    if X_train[c].dtype == "object"
    or str(X_train[c].dtype).startswith("category")
    or str(X_train[c].dtype) == "bool"
]
num_cols = [c for c in X_train.columns if c not in cat_cols]

print("Target:", target_col)
print("Rows dropped (train/test) due to lag NaNs:", dropped_train, dropped_test)
print("Train/Test shapes after lag prep:", train_df.shape, test_df.shape)
if time_col in train_df.columns and time_col in test_df.columns:
    print("Train max time:", train_df[time_col].max(), "| Test min time:", test_df[time_col].min())
print("Features:", len(feature_cols), "| Numeric:", len(num_cols), "| Categorical:", len(cat_cols))
print("Feature list:", feature_cols)

X_train.head(3)



# --------------------
# Strict time-aware validation split (no timestamp overlap between fit/valid)
if time_col is not None and time_col in train_df.columns:
    time_series = pd.to_datetime(train_df[time_col], errors="coerce")
    unique_times = np.sort(time_series.dropna().unique())

    if len(unique_times) < 2:
        raise ValueError("Not enough unique timestamps to create validation split.")

    valid_time_count = max(1, int(len(unique_times) * 0.2))
    valid_start_time = unique_times[-valid_time_count]

    fit_mask = time_series < valid_start_time
    valid_mask = time_series >= valid_start_time

    X_fit = X_train.loc[fit_mask].copy()
    y_fit = y_train.loc[fit_mask].copy()

    X_valid = X_train.loc[valid_mask].copy()
    y_valid = y_train.loc[valid_mask].copy()

    print("Validation starts at:", pd.to_datetime(valid_start_time))
    print("Fit max time:", time_series[fit_mask].max(), "| Valid min time:", time_series[valid_mask].min())
else:
    split_idx = int(len(X_train) * 0.8)
    X_fit = X_train.iloc[:split_idx].copy()
    y_fit = y_train.iloc[:split_idx].copy()

    X_valid = X_train.iloc[split_idx:].copy()
    y_valid = y_train.iloc[split_idx:].copy()

print("Fit split:", X_fit.shape, "| Validation split:", X_valid.shape)


# --------------------
# NaN-safe preprocessing pipelines
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

transformers = []
if cat_cols:
    transformers.append(("cat", categorical_pipeline, cat_cols))
if num_cols:
    transformers.append(("num", numeric_pipeline, num_cols))

preprocessor = ColumnTransformer(
    transformers=transformers,
    remainder="drop",
)


def make_model_from_trial(trial):
    options = ["LR", "RIDGE", "RF", "GBR"]
    if HAS_XGB:
        options.append("XGBR")

    model_name = trial.suggest_categorical("model_name", options)

    if model_name == "LR":
        model = LinearRegression()

    elif model_name == "RIDGE":
        alpha = trial.suggest_float("ridge_alpha", 0.1, 200.0, log=True)
        model = Ridge(alpha=alpha, random_state=RANDOM_STATE)

    elif model_name == "RF":
        n_estimators = trial.suggest_int("rf_n_estimators", 100, 400, step=50)
        max_depth = trial.suggest_int("rf_max_depth", 5, 24)
        min_samples_leaf = trial.suggest_int("rf_min_samples_leaf", 1, 8)
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    elif model_name == "GBR":
        n_estimators = trial.suggest_int("gbr_n_estimators", 80, 400, step=40)
        learning_rate = trial.suggest_float("gbr_learning_rate", 0.01, 0.2, log=True)
        max_depth = trial.suggest_int("gbr_max_depth", 2, 8)
        subsample = trial.suggest_float("gbr_subsample", 0.6, 1.0)
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            random_state=RANDOM_STATE,
        )

    else:  # XGBR
        n_estimators = trial.suggest_int("xgb_n_estimators", 120, 500, step=40)
        learning_rate = trial.suggest_float("xgb_learning_rate", 0.01, 0.2, log=True)
        max_depth = trial.suggest_int("xgb_max_depth", 3, 10)
        subsample = trial.suggest_float("xgb_subsample", 0.6, 1.0)
        colsample_bytree = trial.suggest_float("xgb_colsample", 0.6, 1.0)
        model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    return model_name, model


def objective(trial):
    model_name, model = make_model_from_trial(trial)

    pipeline = Pipeline([
        ("prep", preprocessor),
        ("model", model),
    ])

    # Optional subsampling for faster Optuna iterations on large data.
    sample_size = min(50_000, len(X_fit))
    if sample_size < len(X_fit):
        rng = np.random.default_rng(RANDOM_STATE + trial.number)
        idx = rng.choice(len(X_fit), size=sample_size, replace=False)
        X_fit_sample = X_fit.iloc[idx]
        y_fit_sample = y_fit.iloc[idx]
    else:
        X_fit_sample = X_fit
        y_fit_sample = y_fit

    if USE_MLFLOW and HAS_MLFLOW:
        mlflow.start_run(nested=True)
        mlflow.log_param("model_name", model_name)

    pipeline.fit(X_fit_sample, y_fit_sample)
    y_pred_valid = pipeline.predict(X_valid)

    metrics = evaluate_metrics(y_valid, y_pred_valid)

    # Leakage-sanity guard on metric scale.
    if metrics["MAPE"] < 1e-3:
        raise ValueError(f"Leakage suspected: unrealistically low validation MAPE={metrics['MAPE']:.3e}")

    if USE_MLFLOW and HAS_MLFLOW:
        for k, v in metrics.items():
            mlflow.log_metric(f"valid_{k}", v)
        mlflow.log_params(model.get_params())
        mlflow.end_run()

    return metrics["MAPE"]



# --------------------
# Run Optuna model selection
n_trials = 80

study = optuna.create_study(
    study_name="model_selection",
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE, multivariate=True),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
)

if USE_MLFLOW and HAS_MLFLOW:
    with mlflow.start_run(run_name="model_selection_optuna"):
        study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True, catch=(ValueError,))
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_valid_MAPE", study.best_value)
else:
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True, catch=(ValueError,))

print("Best validation MAPE:", round(study.best_value, 6))
print("Best params:")
study.best_params


# --------------------
# Trials leaderboard
trials_df = study.trials_dataframe()
leaderboard_cols = [
    "number",
    "value",
    "params_model_name",
    "state",
]

leaderboard = trials_df[[c for c in leaderboard_cols if c in trials_df.columns]].copy()
leaderboard = leaderboard.rename(columns={"value": "valid_MAPE"}).sort_values("valid_MAPE").reset_index(drop=True)
leaderboard.head(10)

# --------------------
# Train final model on full train set and evaluate on test set
class DummyTrial:
    def __init__(self, params):
        self.params = params

    def suggest_categorical(self, name, choices):
        return self.params[name]

    def suggest_int(self, name, low, high, step=1):
        return int(self.params[name])

    def suggest_float(self, name, low, high, log=False):
        return float(self.params[name])


best_trial = DummyTrial(study.best_params)
best_model_name, best_model = make_model_from_trial(best_trial)

best_pipeline = Pipeline([
    ("prep", preprocessor),
    ("model", best_model),
])

best_pipeline.fit(X_train, y_train)

y_pred_train = np.clip(best_pipeline.predict(X_train), a_min=0, a_max=None)
y_pred_test = np.clip(best_pipeline.predict(X_test), a_min=0, a_max=None)

train_metrics = evaluate_metrics(y_train, y_pred_train)
test_metrics = evaluate_metrics(y_test, y_pred_test)

metrics_summary = pd.DataFrame([
    {"split": "train", **train_metrics},
    {"split": "test", **test_metrics},
])

print("Final model:", best_model_name)
metrics_summary

# --------------------
# Build prediction output table
predictions = pd.DataFrame({
    "actual_demand": y_test.values,
    "predicted_demand": y_pred_test,
}, index=y_test.index).reset_index(drop=True)

if time_col is not None and time_col in context_test.columns:
    predictions["pickup_slot"] = pd.to_datetime(context_test[time_col].values)
else:
    predictions["pickup_slot"] = pd.RangeIndex(start=0, stop=len(predictions), step=1)

if "region" in context_test.columns:
    predictions["region"] = pd.to_numeric(context_test["region"].values, errors="coerce").astype("Int64")
else:
    predictions["region"] = pd.Series([pd.NA] * len(predictions), dtype="Int64")

# Bring context features for business logic
for col in ["rolling_mean", "rolling_std", "avg_fare_region_slot", "avg_pickups", "avg_pickups_ewm"]:
    if col in context_test.columns:
        predictions[col] = pd.to_numeric(context_test[col].values, errors="coerce")

predictions = predictions.sort_values(["region", "pickup_slot"], na_position="last").reset_index(drop=True)
predictions.head()

# --------------------
# Business intelligence layer from model predictions

# 1) Surge detection
if "rolling_mean" not in predictions.columns:
    predictions["rolling_mean"] = predictions.groupby("region")["actual_demand"].transform("mean")
predictions["rolling_mean"] = predictions["rolling_mean"].fillna(predictions["rolling_mean"].median())

# Notebook-5 requested rule: threshold = rolling_mean * 1.3
predictions["surge_threshold"] = predictions["rolling_mean"] * 1.3
predictions["surge_flag"] = predictions["predicted_demand"] > predictions["surge_threshold"]

surge_ratio = predictions["predicted_demand"] / (predictions["surge_threshold"] + 1e-6)
predictions["surge_level"] = "none"
predictions.loc[predictions["surge_flag"] & (surge_ratio <= 1.10), "surge_level"] = "low"
predictions.loc[predictions["surge_flag"] & (surge_ratio > 1.10) & (surge_ratio <= 1.25), "surge_level"] = "medium"
predictions.loc[predictions["surge_flag"] & (surge_ratio > 1.25), "surge_level"] = "high"

# 2) Risk / stability score
if "rolling_std" not in predictions.columns:
    predictions["rolling_std"] = predictions.groupby("region")["actual_demand"].transform("std")
predictions["rolling_std"] = predictions["rolling_std"].fillna(0)

predictions["risk_score"] = predictions["rolling_std"] / (predictions["rolling_mean"] + 1e-6)
predictions["risk_band"] = pd.cut(
    predictions["risk_score"],
    bins=[-np.inf, 0.35, 0.75, np.inf],
    labels=["Stable", "Moderate", "Volatile"],
).astype(str)

# 3) Revenue estimation
if "avg_fare_region_slot" not in predictions.columns:
    fare_baseline = train_df["avg_fare_region_slot"].median() if "avg_fare_region_slot" in train_df.columns else np.nan
    predictions["avg_fare_region_slot"] = fare_baseline

predictions["avg_fare_region_slot"] = predictions["avg_fare_region_slot"].fillna(predictions["avg_fare_region_slot"].median())
predictions["expected_revenue"] = predictions["predicted_demand"] * predictions["avg_fare_region_slot"]

# 4) Demand pressure
if "avg_pickups" in predictions.columns:
    denom = predictions["avg_pickups"]
elif "avg_pickups_ewm" in predictions.columns:
    denom = predictions["avg_pickups_ewm"]
else:
    denom = predictions.groupby("region")["actual_demand"].transform("mean")

predictions["demand_pressure"] = predictions["predicted_demand"] / (pd.to_numeric(denom, errors="coerce") + 1e-6)

# --------------------
# Driver relocation recommendation using neighbor regions
neighbors_path = data_interim / "region_neighbors.csv"

if neighbors_path.exists() and predictions["region"].notna().any():
    neighbors = pd.read_csv(neighbors_path)
    neighbors = neighbors.rename(columns={"region_id": "region", "neighbor_region_id": "target_region"})

    base = predictions[["pickup_slot", "region", "predicted_demand"]].copy()
    candidate_moves = base.merge(neighbors[["region", "target_region", "distance_km"]], on="region", how="left")

    target_demand = base.rename(
        columns={
            "region": "target_region",
            "predicted_demand": "target_predicted_demand",
        }
    )
    candidate_moves = candidate_moves.merge(target_demand, on=["pickup_slot", "target_region"], how="left")

    candidate_moves["target_predicted_demand"] = candidate_moves["target_predicted_demand"].fillna(0)
    candidate_moves["demand_gain"] = (
        candidate_moves["target_predicted_demand"] - candidate_moves["predicted_demand"]
    ).clip(lower=0)
    candidate_moves["relocation_score"] = candidate_moves["demand_gain"] / (candidate_moves["distance_km"] + 1e-3)

    top_moves = (
        candidate_moves.sort_values(["pickup_slot", "region", "relocation_score"], ascending=[True, True, False])
        .groupby(["pickup_slot", "region"], as_index=False)
        .head(3)
        .copy()
    )
    top_moves["rank"] = top_moves.groupby(["pickup_slot", "region"]).cumcount() + 1

    best_moves = top_moves[top_moves["rank"] == 1].copy()
    min_gain = 1.0
    low_gain = best_moves["demand_gain"] < min_gain
    best_moves.loc[low_gain, ["target_region", "distance_km", "demand_gain", "relocation_score"]] = [np.nan, np.nan, 0.0, 0.0]

    best_moves = best_moves.rename(
        columns={
            "target_region": "recommended_next_zone",
            "distance_km": "recommended_distance_km",
            "demand_gain": "expected_demand_gain",
        }
    )

    predictions = predictions.merge(
        best_moves[["pickup_slot", "region", "recommended_next_zone", "recommended_distance_km", "expected_demand_gain", "relocation_score"]],
        on=["pickup_slot", "region"],
        how="left",
    )
else:
    top_moves = pd.DataFrame()
    predictions["recommended_next_zone"] = pd.Series([pd.NA] * len(predictions), dtype="Int64")
    predictions["recommended_distance_km"] = np.nan
    predictions["expected_demand_gain"] = 0.0
    predictions["relocation_score"] = 0.0

# --------------------
# Best-time recommendation by region from model predictions
if np.issubdtype(predictions["pickup_slot"].dtype, np.datetime64):
    predictions["pickup_day_of_week"] = predictions["pickup_slot"].dt.dayofweek
    predictions["pickup_hour"] = predictions["pickup_slot"].dt.hour

    slot_profile = (
        predictions.groupby(["region", "pickup_day_of_week", "pickup_hour"], dropna=False, as_index=False)["predicted_demand"]
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

    day_names = {
        0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
        4: "Friday", 5: "Saturday", 6: "Sunday",
    }
    best_time_recommendations["day_name"] = best_time_recommendations["pickup_day_of_week"].map(day_names)
    best_time_recommendations["best_time_window"] = best_time_recommendations["pickup_hour"].astype(int).map(
        lambda h: f"{h:02d}:00-{(h + 1) % 24:02d}:00"
    )

    top1_time = (
        best_time_recommendations[best_time_recommendations["rank"] == 1][["region", "best_time_window"]]
        .drop_duplicates(subset=["region"])
    )
    predictions = predictions.merge(top1_time, on="region", how="left")
else:
    slot_profile = pd.DataFrame()
    best_time_recommendations = pd.DataFrame()
    predictions["best_time_window"] = np.nan

# --------------------
# Save artifacts and outputs
model_path = models_dir / "best_model_selection_pipeline.joblib"
leaderboard_path = reports_dir / "model_selection_leaderboard.csv"
metrics_path = reports_dir / "model_metrics_summary.csv"

pred_path = data_interim / "model_test_predictions.csv"
business_output_path = data_interim / "model_business_output.csv"
relocation_path = data_interim / "model_relocation_candidates.csv"
best_time_path = data_interim / "model_best_time_recommendations.csv"

joblib.dump(best_pipeline, model_path)

leaderboard.to_csv(leaderboard_path, index=False)
metrics_summary.to_csv(metrics_path, index=False)

predictions.to_csv(pred_path, index=False)

ui_cols = [
    "pickup_slot",
    "region",
    "actual_demand",
    "predicted_demand",
    "surge_flag",
    "surge_level",
    "risk_score",
    "risk_band",
    "expected_revenue",
    "demand_pressure",
    "recommended_next_zone",
    "recommended_distance_km",
    "expected_demand_gain",
    "relocation_score",
    "best_time_window",
]
predictions[[c for c in ui_cols if c in predictions.columns]].to_csv(business_output_path, index=False)

if not top_moves.empty:
    top_moves.to_csv(relocation_path, index=False)
else:
    pd.DataFrame().to_csv(relocation_path, index=False)

if not best_time_recommendations.empty:
    best_time_recommendations.to_csv(best_time_path, index=False)
else:
    pd.DataFrame().to_csv(best_time_path, index=False)

print("Saved artifacts:")
for p in [
    model_path,
    leaderboard_path,
    metrics_path,
    pred_path,
    business_output_path,
    relocation_path,
    best_time_path,
]:
    print("-", p.relative_to(project_root))

# --------------------
# Quick checks
print("Best model:", best_model_name)
display(metrics_summary)
display(predictions.head(10))

# --------------------
# Simple plot: actual vs predicted demand (test)
plot_df = predictions.copy()
if np.issubdtype(plot_df["pickup_slot"].dtype, np.datetime64):
    plot_df = plot_df.sort_values("pickup_slot")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(plot_df["actual_demand"].values[:1000], label="Actual", alpha=0.8)
ax.plot(plot_df["predicted_demand"].values[:1000], label="Predicted", alpha=0.8)
ax.set_title("Actual vs Predicted Demand (first 1000 test rows)")
ax.set_xlabel("Row")
ax.set_ylabel("Demand")
ax.legend()
plt.tight_layout()
plt.show()

# --------------------

--- notebooks\6. Training_model.ipynb ---
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# --------------------
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
)

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

!pip install xgboost
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

!pip install mlflow
try:
    import mlflow
    HAS_MLFLOW = True
except Exception:
    HAS_MLFLOW = False

!pip install dagshub
try:
    import dagshub
    HAS_DAGSHUB = True
except Exception:
    HAS_DAGSHUB = False

import optuna

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# --------------------
# import dagshub
# import mlflow

# dagshub.init(
#     repo_owner="Shubham39275",
#     repo_name="Taxi_Updated",
#     mlflow=True
# )

# mlflow.set_experiment("Model Selection")
# --------------------
from pathlib import Path
import pandas as pd

# helper
def get_time_col(df):
    for c in ["pickup_slot", "tpep_pickup_datetime", "timestamp"]:
        if c in df.columns:
            return c
    return None

# Kaggle-aware paths
KAGGLE_INPUT = Path("/kaggle/input/notebooks/shubhamyadav74/notebookacca71d522")
data_processed = Path("/kaggle/working")
data_interim = Path("/kaggle/working")

train_path = data_processed / "train.csv"
test_path = data_processed / "test.csv"

time_col = None

if train_path.exists() and test_path.exists():
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    source_mode = "processed_train_test"

    time_col = get_time_col(train_df) or get_time_col(test_df)
    if time_col:
        train_df[time_col] = pd.to_datetime(train_df[time_col], errors="coerce")
        test_df[time_col] = pd.to_datetime(test_df[time_col], errors="coerce")

else:
    hist_candidates = [
        data_interim / "historical_features.csv",
        data_interim / "final_data.csv",
        KAGGLE_INPUT / "historical_features.csv",
        KAGGLE_INPUT / "final_data.csv",
    ]

    hist_path = next((p for p in hist_candidates if p.exists()), None)

    if hist_path is None:
        raise FileNotFoundError(
            "No train/test or historical features found. Check Kaggle input path."
        )

    print("Using file:", hist_path)
    all_df = pd.read_csv(hist_path)

    time_col = get_time_col(all_df)
    if time_col is None:
        raise ValueError(
            "Historical data missing time column (`pickup_slot` or `tpep_pickup_datetime`)."
        )

    all_df[time_col] = pd.to_datetime(all_df[time_col], errors="coerce")
    all_df = all_df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)

    split_idx = int(len(all_df) * 0.8)
    train_df = all_df.iloc[:split_idx].copy()
    test_df = all_df.iloc[split_idx:].copy()
    source_mode = "historical_split"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

print("Source mode:", source_mode)
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

if time_col and time_col in train_df.columns and time_col in test_df.columns:
    print("Train time range:", train_df[time_col].min(), "→", train_df[time_col].max())
    print("Test time range:", test_df[time_col].min(), "→", test_df[time_col].max())
else:
    print("No time column found in train/test for range debug.")

# --------------------
# def smape(y_true, y_pred):
#     y_true = np.asarray(y_true)
#     y_pred = np.asarray(y_pred)
#     denom = np.abs(y_true) + np.abs(y_pred)
#     return np.mean(2.0 * np.abs(y_true - y_pred) / np.where(denom == 0, 1.0, denom))


# def evaluate_metrics(y_true, y_pred):
#     return {
#         "MAE": float(mean_absolute_error(y_true, y_pred)),
#         "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
#         "MAPE": float(mean_absolute_percentage_error(y_true, y_pred)),
#         "sMAPE": float(smape(y_true, y_pred)),
#         "R2": float(r2_score(y_true, y_pred)),
#     }


# def safe_fill_frame(df: pd.DataFrame):
#     out = df.copy()
#     for col in out.columns:
#         if pd.api.types.is_numeric_dtype(out[col]):
#             median_val = out[col].median()
#             if pd.isna(median_val):
#                 median_val = 0.0
#             out[col] = out[col].fillna(median_val)
#         else:
#             mode_vals = out[col].mode(dropna=True)
#             fill_val = mode_vals.iloc[0] if len(mode_vals) else "unknown"
#             out[col] = out[col].fillna(fill_val)
#     return out


# def get_time_col(df: pd.DataFrame):
#     for c in ["pickup_slot", "tpep_pickup_datetime", "timestamp"]:
#         if c in df.columns:
#             return c
#     return None

# --------------------
# def safe_mape(y_true, y_pred):
#     y_true = np.maximum(y_true, 1)   # 🔥 avoid explosion when near 0
#     return np.mean(np.abs((y_true - y_pred) / y_true))


def safe_mape(y_true, y_pred):
    y_true = np.maximum(y_true, 10)   # 🔥 key trick
    return mean_absolute_percentage_error(y_true, y_pred)

def smape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.abs(y_true) + np.abs(y_pred)
    return np.mean(2.0 * np.abs(y_true - y_pred) / np.where(denom == 0, 1.0, denom))

def evaluate_metrics(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE": float(safe_mape(y_true, y_pred)),   # 🔥 use safe version
        "sMAPE": float(smape(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }
def safe_fill_frame(df: pd.DataFrame):
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


def get_time_col(df: pd.DataFrame):
    for c in ["pickup_slot", "tpep_pickup_datetime", "timestamp"]:
        if c in df.columns:
            return c
    return None

# --------------------
# Load training/test data
train_path = data_processed / "train.csv"
test_path = data_processed / "test.csv"

if train_path.exists() and test_path.exists():
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    source_mode = "processed_train_test"
else:
    hist_candidates = [
        data_interim / "historical_features.csv",
        data_interim / "final_data.csv",
    ]
    hist_path = next((p for p in hist_candidates if p.exists()), None)
    if hist_path is None:
        raise FileNotFoundError(
            "No train/test or historical features found. Run Notebook 4 first."
        )

    all_df = pd.read_csv(hist_path)
    time_col = get_time_col(all_df)
    if time_col is None:
        raise ValueError("Historical data missing time column (`pickup_slot` or `tpep_pickup_datetime`).")

    all_df[time_col] = pd.to_datetime(all_df[time_col], errors="coerce")
    all_df = all_df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)

    split_idx = int(len(all_df) * 0.8)
    train_df = all_df.iloc[:split_idx].copy()
    test_df = all_df.iloc[split_idx:].copy()
    source_mode = "historical_split"

print("Source mode:", source_mode)
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

# --------------------
# # Target and feature preparation + lag feature engineering (FINAL FIX)

# target_candidates = ["total_pickups_model", "total_pickups_raw", "total_pickups"]
# target_col = next((c for c in target_candidates if c in train_df.columns), None)

# if target_col is None:
#     raise ValueError(f"Target column missing. Expected one of: {target_candidates}")

# time_col = get_time_col(train_df)
# if time_col is None:
#     raise ValueError("Time column required")

# # Ensure datetime
# train_df[time_col] = pd.to_datetime(train_df[time_col], errors="coerce")
# test_df[time_col] = pd.to_datetime(test_df[time_col], errors="coerce")

# # Sort
# sort_cols = ["region", time_col] if "region" in train_df.columns else [time_col]
# train_df = train_df.sort_values(sort_cols).reset_index(drop=True)
# test_df = test_df.sort_values(sort_cols).reset_index(drop=True)

# # ---------------- LAG FEATURES ----------------
# lag_steps = [1, 2, 3, 4, 6, 12, 24, 96]
# rolling_windows = [3, 6, 12]

# import numpy as np

# if "pickup_hour" in train_df.columns:
#     for df in [train_df, test_df]:
#         df["hour_sin"] = np.sin(2 * np.pi * df["pickup_hour"] / 24)
#         df["hour_cos"] = np.cos(2 * np.pi * df["pickup_hour"] / 24)

# def add_lag(df):
#     if "region" in df.columns:
#         for lag in lag_steps:
#             df[f"lag_{lag}"] = df.groupby("region")[target_col].shift(lag)

#         for w in rolling_windows:
#             df[f"lag_roll_mean_{w}"] = df.groupby("region")[target_col].transform(
#                 lambda s: s.shift(1).rolling(w).mean()
#             )
#             df[f"lag_roll_std_{w}"] = df.groupby("region")[target_col].transform(
#                 lambda s: s.shift(1).rolling(w).std()
#             )
#     else:
#         for lag in lag_steps:
#             df[f"lag_{lag}"] = df[target_col].shift(lag)

#         for w in rolling_windows:
#             df[f"lag_roll_mean_{w}"] = df[target_col].shift(1).rolling(w).mean()

#     return df

# # Train lag
# train_df = add_lag(train_df)
# lag_cols = [c for c in train_df.columns if c.startswith("lag_")]
# train_df = train_df.dropna(subset=lag_cols).reset_index(drop=True)

# # Test lag (only past)
# history = train_df.tail(150)
# test_temp = pd.concat([history, test_df], axis=0).reset_index(drop=True)
# test_temp = add_lag(test_temp)

# test_df = test_temp.iloc[len(history):].copy()
# test_df = test_df.dropna(subset=lag_cols).reset_index(drop=True)

# print("Train shape:", train_df.shape)
# print("Test shape:", test_df.shape)

# # ---------------- STRICT FEATURE SELECTION ----------------

# # ONLY allow SAFE columns
# safe_base_features = [
#     "region",
#     "pickup_hour",
#     "pickup_day_of_week",
#     "is_weekend",
#     "rush_hour",
#     "is_night",
#     "hour_sin",
#     "hour_cos",
# ]

# safe_base_features = [c for c in safe_base_features if c in train_df.columns]

# lag_features = [c for c in train_df.columns if c.startswith("lag_")]

# # FINAL FEATURES = ONLY THESE
# feature_cols = safe_base_features + lag_features

# if not feature_cols:
#     raise ValueError("No valid features")

# # ---------------- SPLIT ----------------

# X_train = train_df[feature_cols].copy()
# y_train = train_df[target_col].copy()

# X_test = test_df[feature_cols].copy()
# y_test = test_df[target_col].copy()

# X_train = safe_fill_frame(X_train)
# X_test = safe_fill_frame(X_test)

# # Convert boolean columns to int (fix sklearn error)
# for col in X_train.columns:
#     if X_train[col].dtype == "bool":
#         X_train[col] = X_train[col].astype(int)
#         X_test[col] = X_test[col].astype(int)

# print("Final Features:", feature_cols)
# print("Feature count:", len(feature_cols))
# --------------------
# #######
# # Target and feature preparation + lag feature engineering (IMPROVED)

# target_candidates = ["total_pickups_model", "total_pickups_raw", "total_pickups"]
# target_col = next((c for c in target_candidates if c in train_df.columns), None)

# if target_col is None:
#     raise ValueError(f"Target column missing. Expected one of: {target_candidates}")

# time_col = get_time_col(train_df)
# if time_col is None:
#     raise ValueError("Time column required")

# # Ensure datetime
# train_df[time_col] = pd.to_datetime(train_df[time_col], errors="coerce")
# test_df[time_col] = pd.to_datetime(test_df[time_col], errors="coerce")

# # Sort
# sort_cols = ["region", time_col] if "region" in train_df.columns else [time_col]
# train_df = train_df.sort_values(sort_cols).reset_index(drop=True)
# test_df = test_df.sort_values(sort_cols).reset_index(drop=True)

# # ---------------- LAG FEATURES ----------------
# lag_steps = [1, 2, 3, 4, 6, 12, 24, 96]
# rolling_windows = [3, 6, 12]

# import numpy as np

# # Cyclic time features
# if "pickup_hour" in train_df.columns:
#     for df in [train_df, test_df]:
#         df["hour_sin"] = np.sin(2 * np.pi * df["pickup_hour"] / 24)
#         df["hour_cos"] = np.cos(2 * np.pi * df["pickup_hour"] / 24)

# def add_lag(df):
#     if "region" in df.columns:
#         for lag in lag_steps:
#             df[f"lag_{lag}"] = df.groupby("region")[target_col].shift(lag)

#         for w in rolling_windows:
#             df[f"lag_roll_mean_{w}"] = df.groupby("region")[target_col].transform(
#                 lambda s: s.shift(1).rolling(w).mean()
#             )
#             df[f"lag_roll_std_{w}"] = df.groupby("region")[target_col].transform(
#                 lambda s: s.shift(1).rolling(w).std()
#             )
#     else:
#         for lag in lag_steps:
#             df[f"lag_{lag}"] = df[target_col].shift(lag)

#         for w in rolling_windows:
#             df[f"lag_roll_mean_{w}"] = df[target_col].shift(1).rolling(w).mean()

#     return df

# # Train lag
# train_df = add_lag(train_df)
# lag_cols = [c for c in train_df.columns if c.startswith("lag_")]
# train_df = train_df.dropna(subset=lag_cols).reset_index(drop=True)

# # Test lag (only past)
# history = train_df.tail(150)
# test_temp = pd.concat([history, test_df], axis=0).reset_index(drop=True)
# test_temp = add_lag(test_temp)

# test_df = test_temp.iloc[len(history):].copy()
# test_df = test_df.dropna(subset=lag_cols).reset_index(drop=True)

# # ---------------- NEW FEATURES (IMPORTANT) ----------------

# # Trend features
# train_df["lag_diff_1"] = train_df["lag_1"] - train_df["lag_2"]
# test_df["lag_diff_1"] = test_df["lag_1"] - test_df["lag_2"]

# train_df["lag_diff_24"] = train_df["lag_1"] - train_df["lag_24"]
# test_df["lag_diff_24"] = test_df["lag_1"] - test_df["lag_24"]

# train_df["trend_strength"] = train_df["lag_1"] - train_df["lag_roll_mean_3"]
# test_df["trend_strength"] = test_df["lag_1"] - test_df["lag_roll_mean_3"]

# # Peak hour
# peak_hours = [8, 9, 18, 19]
# train_df["is_peak"] = train_df["pickup_hour"].isin(peak_hours).astype(int)
# test_df["is_peak"] = test_df["pickup_hour"].isin(peak_hours).astype(int)

# # Region interaction
# train_df["region_hour"] = train_df["region"] * train_df["pickup_hour"]
# test_df["region_hour"] = test_df["region"] * test_df["pickup_hour"]

# print("Train shape:", train_df.shape)
# print("Test shape:", test_df.shape)

# # ---------------- FEATURE SELECTION ----------------

# safe_base_features = [
#     "region",
#     "pickup_hour",
#     "pickup_day_of_week",
#     "is_weekend",
#     "rush_hour",
#     "is_night",
#     "hour_sin",
#     "hour_cos",
# ]

# safe_base_features = [c for c in safe_base_features if c in train_df.columns]

# lag_features = [c for c in train_df.columns if c.startswith("lag_")]

# # feature_cols = safe_base_features + lag_features + [
# #     "lag_diff_1",
# #     "lag_diff_24",
# #     "trend_strength",
# #     "is_peak",
# #     "region_hour"
# # ]
# feature_cols = list(set(
#     safe_base_features + lag_features + [
#         "trend_strength",
#         "is_peak",
#         "region_hour"
#     ]
# ))

# X_train = train_df[feature_cols].copy()
# y_train = train_df[target_col].copy()

# X_test = test_df[feature_cols].copy()
# y_test = test_df[target_col].copy()

# X_train = safe_fill_frame(X_train)
# X_test = safe_fill_frame(X_test)

# # Fix bool issue
# bool_cols = X_train.select_dtypes(include=["bool"]).columns

# for col in bool_cols:
#     X_train[col] = X_train[col].astype(int)
#     X_test[col] = X_test[col].astype(int)

# print("Final Features:", feature_cols)
# print("Feature count:", len(feature_cols))
# --------------------
# # ================= FINAL FEATURE ENGINEERING =================

# target_candidates = ["total_pickups_model", "total_pickups_raw", "total_pickups"]
# target_col = next((c for c in target_candidates if c in train_df.columns), None)

# if target_col is None:
#     raise ValueError(f"Target column missing. Expected one of: {target_candidates}")

# time_col = get_time_col(train_df)
# if time_col is None:
#     raise ValueError("Time column required")

# # Ensure datetime
# train_df[time_col] = pd.to_datetime(train_df[time_col], errors="coerce")
# test_df[time_col] = pd.to_datetime(test_df[time_col], errors="coerce")

# # Sort
# sort_cols = ["region", time_col] if "region" in train_df.columns else [time_col]
# train_df = train_df.sort_values(sort_cols).reset_index(drop=True)
# test_df = test_df.sort_values(sort_cols).reset_index(drop=True)

# # ---------------- LAG FEATURES ----------------
# lag_steps = [1, 2, 3, 6, 12, 24]   # 🔥 reduced noise
# rolling_windows = [3, 6]

# import numpy as np

# # Cyclic features
# if "pickup_hour" in train_df.columns:
#     for df in [train_df, test_df]:
#         df["hour_sin"] = np.sin(2 * np.pi * df["pickup_hour"] / 24)
#         df["hour_cos"] = np.cos(2 * np.pi * df["pickup_hour"] / 24)

# def add_lag(df):
#     if "region" in df.columns:
#         for lag in lag_steps:
#             df[f"lag_{lag}"] = df.groupby("region")[target_col].shift(lag)

#         for w in rolling_windows:
#             df[f"lag_roll_mean_{w}"] = df.groupby("region")[target_col].transform(
#                 lambda s: s.shift(1).rolling(w).mean()
#             )
#             df[f"lag_roll_std_{w}"] = df.groupby("region")[target_col].transform(
#                 lambda s: s.shift(1).rolling(w).std()
#             )
#     else:
#         for lag in lag_steps:
#             df[f"lag_{lag}"] = df[target_col].shift(lag)

#     return df

# # Apply lag
# train_df = add_lag(train_df)
# lag_cols = [c for c in train_df.columns if c.startswith("lag_")]
# train_df = train_df.dropna(subset=lag_cols).reset_index(drop=True)

# # Test lag using history
# history = train_df.tail(150)
# test_temp = pd.concat([history, test_df], axis=0).reset_index(drop=True)
# test_temp = add_lag(test_temp)

# test_df = test_temp.iloc[len(history):].copy()
# test_df = test_df.dropna(subset=lag_cols).reset_index(drop=True)

# # ---------------- NEW POWER FEATURES ----------------

# # Trend
# train_df["trend_strength"] = train_df["lag_1"] - train_df["lag_roll_mean_3"]
# test_df["trend_strength"] = test_df["lag_1"] - test_df["lag_roll_mean_3"]

# # Peak
# peak_hours = [8, 9, 18, 19]
# train_df["is_peak"] = train_df["pickup_hour"].isin(peak_hours).astype(int)
# test_df["is_peak"] = test_df["pickup_hour"].isin(peak_hours).astype(int)

# # Region interaction
# train_df["region_hour"] = train_df["region"] * train_df["pickup_hour"]
# test_df["region_hour"] = test_df["region"] * test_df["pickup_hour"]

# # 🔥 MOST IMPORTANT (STAT FEATURES)
# # train_df["region_mean"] = train_df.groupby("region")[target_col].transform("mean")
# # test_df["region_mean"] = test_df.groupby("region")[target_col].transform("mean")

# # train_df["hour_mean"] = train_df.groupby("pickup_hour")[target_col].transform("mean")
# # test_df["hour_mean"] = test_df.groupby("pickup_hour")[target_col].transform("mean")

# # train_df["dow_mean"] = train_df.groupby("pickup_day_of_week")[target_col].transform("mean")
# # test_df["dow_mean"] = test_df.groupby("pickup_day_of_week")[target_col].transform("mean")

# # 🔥 FIXED (NO LEAKAGE)

# train_df["region_mean"] = train_df.groupby("region")[target_col].transform(
#     lambda s: s.shift(1).expanding().mean()
# )
# test_df["region_mean"] = test_df.groupby("region")[target_col].transform("mean")

# train_df["hour_mean"] = train_df.groupby("pickup_hour")[target_col].transform(
#     lambda s: s.shift(1).expanding().mean()
# )
# test_df["hour_mean"] = test_df.groupby("pickup_hour")[target_col].transform("mean")

# train_df["dow_mean"] = train_df.groupby("pickup_day_of_week")[target_col].transform(
#     lambda s: s.shift(1).expanding().mean()
# )
# test_df["dow_mean"] = test_df.groupby("pickup_day_of_week")[target_col].transform("mean")

# print("Train shape:", train_df.shape)
# print("Test shape:", test_df.shape)

# # ---------------- FEATURE SELECTION ----------------

# safe_base_features = [
#     "region",
#     "pickup_hour",
#     "pickup_day_of_week",
#     "is_weekend",
#     "rush_hour",
#     "is_night",
#     "hour_sin",
#     "hour_cos",
# ]

# safe_base_features = [c for c in safe_base_features if c in train_df.columns]

# lag_features = [c for c in train_df.columns if c.startswith("lag_")]

# extra_features = [
#     "trend_strength",
#     "is_peak",
#     "region_hour",
#     "region_mean",
#     "hour_mean",
#     "dow_mean",
# ]

# feature_cols = safe_base_features + lag_features + extra_features

# # remove duplicates safely
# feature_cols = list(dict.fromkeys(feature_cols))

# X_train = train_df[feature_cols].copy()
# y_train = train_df[target_col].copy()

# X_test = test_df[feature_cols].copy()
# y_test = test_df[target_col].copy()

# X_train = safe_fill_frame(X_train)
# X_test = safe_fill_frame(X_test)

# # Fix bool columns
# bool_cols = X_train.select_dtypes(include=["bool"]).columns
# for col in bool_cols:
#     X_train[col] = X_train[col].astype(int)
#     X_test[col] = X_test[col].astype(int)

# print("Final Features:", feature_cols)
# print("Feature count:", len(feature_cols))
# --------------------
# ================= FINAL FEATURE ENGINEERING =================

target_candidates = ["total_pickups_model", "total_pickups_raw", "total_pickups"]
target_col = next((c for c in target_candidates if c in train_df.columns), None)

if target_col is None:
    raise ValueError(f"Target column missing. Expected one of: {target_candidates}")

time_col = get_time_col(train_df)
if time_col is None:
    raise ValueError("Time column required")

# Ensure datetime
train_df[time_col] = pd.to_datetime(train_df[time_col], errors="coerce")
test_df[time_col] = pd.to_datetime(test_df[time_col], errors="coerce")

# Sort
sort_cols = ["region", time_col] if "region" in train_df.columns else [time_col]
train_df = train_df.sort_values(sort_cols).reset_index(drop=True)
test_df = test_df.sort_values(sort_cols).reset_index(drop=True)

# ================= TIME FEATURES =================
for df in [train_df, test_df]:
    df["week_of_year"] = df[time_col].dt.isocalendar().week.astype(int)
    df["day_of_month"] = df[time_col].dt.day
    df["is_month_start"] = df[time_col].dt.is_month_start.astype(int)
    df["is_month_end"] = df[time_col].dt.is_month_end.astype(int)

# ================= LAG FEATURES =================
lag_steps = [1, 2, 3, 6, 12, 24]
rolling_windows = [3, 6]

# Cyclic features
if "pickup_hour" in train_df.columns:
    for df in [train_df, test_df]:
        df["hour_sin"] = np.sin(2 * np.pi * df["pickup_hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["pickup_hour"] / 24)


def add_lag(df):
    if "region" in df.columns:
        for lag in lag_steps:
            df[f"lag_{lag}"] = df.groupby("region")[target_col].shift(lag)

        for w in rolling_windows:
            df[f"lag_roll_mean_{w}"] = df.groupby("region")[target_col].transform(
                lambda s: s.shift(1).rolling(w).mean()
            )
            df[f"lag_roll_std_{w}"] = df.groupby("region")[target_col].transform(
                lambda s: s.shift(1).rolling(w).std()
            )
    else:
        for lag in lag_steps:
            df[f"lag_{lag}"] = df[target_col].shift(lag)

    return df


# train_df = add_lag(train_df)
# lag_cols = [c for c in train_df.columns if c.startswith("lag_")]
# train_df = train_df.dropna(subset=lag_cols).reset_index(drop=True)

train_df = add_lag(train_df)

lag_cols = [c for c in train_df.columns if c.startswith("lag_")]

# 🔥 SAFE DROP (CRITICAL FIX)
min_required_lags = [c for c in lag_cols if "lag_1" in c or "lag_2" in c]

train_df = train_df.dropna(subset=min_required_lags)

# fallback if still empty
if len(train_df) == 0:
    print("⚠️ train empty after drop → using forward fill fallback")
    train_df = add_lag(train_df)
    train_df = train_df.fillna(method="ffill").fillna(method="bfill")

train_df = train_df.reset_index(drop=True)

# test lag
history = train_df.tail(150)
test_temp = pd.concat([history, test_df], axis=0).reset_index(drop=True)
test_temp = add_lag(test_temp)

# test_df = test_temp.iloc[len(history):].copy()
# test_df = test_df.dropna(subset=lag_cols).reset_index(drop=True)

test_df = test_temp.iloc[len(history):].copy()

# 🔥 SAFE DROP
test_df = test_df.dropna(subset=min_required_lags)

if len(test_df) == 0:
    print("⚠️ test empty → fallback fill")
    test_df = test_df.fillna(method="ffill").fillna(method="bfill")

test_df = test_df.reset_index(drop=True)

# ================= EXTRA FEATURES =================

train_df["trend_strength"] = train_df["lag_1"] - train_df["lag_roll_mean_3"]
test_df["trend_strength"] = test_df["lag_1"] - test_df["lag_roll_mean_3"]

peak_hours = [8, 9, 18, 19]
train_df["is_peak"] = train_df["pickup_hour"].isin(peak_hours).astype(int)
test_df["is_peak"] = test_df["pickup_hour"].isin(peak_hours).astype(int)

train_df["region_hour"] = train_df["region"] * train_df["pickup_hour"]
test_df["region_hour"] = test_df["region"] * test_df["pickup_hour"]

# ================= REGION SMOOTHING (🔥 BEST FIX) =================

global_mean = train_df[target_col].mean()

region_stats = train_df.groupby("region")[target_col].agg(["mean", "count"])

smooth = 20

region_smooth_map = (
    (region_stats["mean"] * region_stats["count"] + global_mean * smooth)
    / (region_stats["count"] + smooth)
)

train_df["region_mean"] = train_df["region"].map(region_smooth_map)
test_df["region_mean"] = test_df["region"].map(region_smooth_map)

# other stats (safe version)
train_df["hour_mean"] = train_df.groupby("pickup_hour")[target_col].transform(
    lambda s: s.shift(1).expanding().mean()
)
test_df["hour_mean"] = test_df.groupby("pickup_hour")[target_col].transform("mean")

train_df["dow_mean"] = train_df.groupby("pickup_day_of_week")[target_col].transform(
    lambda s: s.shift(1).expanding().mean()
)
test_df["dow_mean"] = test_df.groupby("pickup_day_of_week")[target_col].transform("mean")

# ================= FEATURES =================

safe_base_features = [
    "region",
    "pickup_hour",
    "pickup_day_of_week",
    "is_weekend",
    "rush_hour",
    "is_night",
    "hour_sin",
    "hour_cos",
    "week_of_year",
    "day_of_month",
    "is_month_start",
    "is_month_end",
]

safe_base_features = [c for c in safe_base_features if c in train_df.columns]

lag_features = [c for c in train_df.columns if c.startswith("lag_")]

extra_features = [
    "trend_strength",
    "is_peak",
    "region_hour",
    "region_mean",
    "hour_mean",
    "dow_mean",
]

feature_cols = list(dict.fromkeys(safe_base_features + lag_features + extra_features))

X_train = safe_fill_frame(train_df[feature_cols])
y_train = train_df[target_col]

X_test = safe_fill_frame(test_df[feature_cols])
y_test = test_df[target_col]

# bool fix
for col in X_train.select_dtypes(include=["bool"]).columns:
    X_train[col] = X_train[col].astype(int)
    X_test[col] = X_test[col].astype(int)

print("Final Features:", feature_cols)
print("Feature count:", len(feature_cols))
# --------------------
# Time-aware validation split from training set
if time_col is not None and time_col in train_df.columns:
    train_sorted_idx = train_df.sort_values(time_col).index
    X_train_sorted = X_train.loc[train_sorted_idx]
    y_train_sorted = y_train.loc[train_sorted_idx]
else:
    X_train_sorted = X_train.copy()
    y_train_sorted = y_train.copy()

split_idx = int(len(X_train_sorted) * 0.8)
X_fit = X_train_sorted.iloc[:split_idx].copy()
y_fit = y_train_sorted.iloc[:split_idx].copy()

X_valid = X_train_sorted.iloc[split_idx:].copy()
y_valid = y_train_sorted.iloc[split_idx:].copy()

print("Fit split:", X_fit.shape, "| Validation split:", X_valid.shape)

# --------------------
# Detect categorical and numerical columns (SAFE)

cat_cols = X_train.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

num_cols = X_train.select_dtypes(
    exclude=["object", "category", "bool"]
).columns.tolist()

print("Categorical:", cat_cols)
print("Numerical:", num_cols)
# --------------------
# # NaN-safe preprocessing pipelines
# numeric_pipeline = Pipeline([
#     ("imputer", SimpleImputer(strategy="median")),
# ])

# categorical_pipeline = Pipeline([
#     ("imputer", SimpleImputer(strategy="most_frequent")),
#     ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
# ])

# transformers = []
# if cat_cols:
#     transformers.append(("cat", categorical_pipeline, cat_cols))
# if num_cols:
#     transformers.append(("num", numeric_pipeline, num_cols))

# preprocessor = ColumnTransformer(
#     transformers=transformers,
#     remainder="drop",
# )


# def make_model_from_trial(trial):
#     options = ["LR", "RIDGE", "RF", "GBR"]
#     if HAS_XGB:
#         options.append("XGBR")

#     model_name = trial.suggest_categorical("model_name", options)

#     if model_name == "LR":
#         model = LinearRegression()

#     elif model_name == "RIDGE":
#         alpha = trial.suggest_float("ridge_alpha", 0.1, 200.0, log=True)
#         model = Ridge(alpha=alpha, random_state=RANDOM_STATE)

#     elif model_name == "RF":
#         n_estimators = trial.suggest_int("rf_n_estimators", 100, 400, step=50)
#         max_depth = trial.suggest_int("rf_max_depth", 5, 24)
#         min_samples_leaf = trial.suggest_int("rf_min_samples_leaf", 1, 8)
#         model = RandomForestRegressor(
#             n_estimators=n_estimators,
#             max_depth=max_depth,
#             min_samples_leaf=min_samples_leaf,
#             random_state=RANDOM_STATE,
#             n_jobs=-1,
#         )

#     elif model_name == "GBR":
#         n_estimators = trial.suggest_int("gbr_n_estimators", 80, 400, step=40)
#         learning_rate = trial.suggest_float("gbr_learning_rate", 0.01, 0.2, log=True)
#         max_depth = trial.suggest_int("gbr_max_depth", 2, 8)
#         subsample = trial.suggest_float("gbr_subsample", 0.6, 1.0)
#         model = GradientBoostingRegressor(
#             n_estimators=n_estimators,
#             learning_rate=learning_rate,
#             max_depth=max_depth,
#             subsample=subsample,
#             random_state=RANDOM_STATE,
#         )

#     else:  # XGBR
#         n_estimators = trial.suggest_int("xgb_n_estimators", 200, 800, step=100)
#         learning_rate = trial.suggest_float("xgb_learning_rate", 0.03, 0.2, log=True)
#         max_depth = trial.suggest_int("xgb_max_depth", 3, 8)
#         subsample = trial.suggest_float("xgb_subsample", 0.7, 1.0)
#         colsample_bytree = trial.suggest_float("xgb_colsample", 0.7, 1.0)
        
#         reg_alpha = trial.suggest_float("xgb_alpha", 0.0, 5.0)
#         reg_lambda = trial.suggest_float("xgb_lambda", 0.5, 5.0)
        
#         model = XGBRegressor(
#             n_estimators=n_estimators,
#             learning_rate=learning_rate,
#             max_depth=max_depth,
#             subsample=subsample,
#             colsample_bytree=colsample_bytree,
#             reg_alpha=reg_alpha,
#             reg_lambda=reg_lambda,
#             objective="reg:squarederror",
#             random_state=RANDOM_STATE,
#             n_jobs=-1,
#         )
#     return model_name, model


# def objective(trial):
#     model_name, model = make_model_from_trial(trial)

#     pipeline = Pipeline([
#         ("prep", preprocessor),
#         ("model", model),
#     ])

#     if HAS_MLFLOW:
#         mlflow.start_run(nested=True)
#         mlflow.log_param("model_name", model_name)

#     pipeline.fit(X_fit, y_fit)
#     y_pred_valid = pipeline.predict(X_valid)

#     metrics = evaluate_metrics(y_valid, y_pred_valid)

#     if HAS_MLFLOW:
#         for k, v in metrics.items():
#             mlflow.log_metric(f"valid_{k}", v)
#         mlflow.log_params(model.get_params())
#         mlflow.end_run()

#     return metrics["MAPE"]

# --------------------
# #####
# # NaN-safe preprocessing pipelines
# numeric_pipeline = Pipeline([
#     ("imputer", SimpleImputer(strategy="median")),
# ])

# categorical_pipeline = Pipeline([
#     ("imputer", SimpleImputer(strategy="most_frequent")),
#     ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
# ])

# transformers = []
# if cat_cols:
#     transformers.append(("cat", categorical_pipeline, cat_cols))
# if num_cols:
#     transformers.append(("num", numeric_pipeline, num_cols))

# preprocessor = ColumnTransformer(
#     transformers=transformers,
#     remainder="drop",
# )

# def make_model_from_trial(trial):
#     model_name = trial.suggest_categorical(
#         "model_name",
#         ["LR", "RIDGE", "RF", "GBR", "XGBR"]
#     )

#     if model_name == "LR":
#         model = LinearRegression()

#     elif model_name == "RIDGE":
#         alpha = trial.suggest_float("ridge_alpha", 0.1, 100.0, log=True)
#         model = Ridge(alpha=alpha, random_state=RANDOM_STATE)

#     elif model_name == "RF":
#         model = RandomForestRegressor(
#             n_estimators=trial.suggest_int("rf_n_estimators", 50, 200, step=50),
#             max_depth=trial.suggest_int("rf_max_depth", 5, 15),
#             min_samples_leaf=trial.suggest_int("rf_min_samples_leaf", 2, 10),
#             max_features=trial.suggest_float("rf_max_features", 0.5, 1.0),
#             random_state=RANDOM_STATE,
#             n_jobs=-1,
#         )

#     elif model_name == "GBR":
#         model = GradientBoostingRegressor(
#             n_estimators=trial.suggest_int("gbr_n_estimators", 80, 200, step=40),
#             learning_rate=trial.suggest_float("gbr_learning_rate", 0.01, 0.1, log=True),
#             max_depth=trial.suggest_int("gbr_max_depth", 2, 6),
#             subsample=trial.suggest_float("gbr_subsample", 0.7, 1.0),
#             random_state=RANDOM_STATE,
#         )

#     else:  # XGBR
#         model = XGBRegressor(
#             n_estimators=trial.suggest_int("xgb_n_estimators", 200, 600, step=100),
#             learning_rate=trial.suggest_float("xgb_learning_rate", 0.03, 0.15, log=True),
#             max_depth=trial.suggest_int("xgb_max_depth", 3, 8),
#             subsample=trial.suggest_float("xgb_subsample", 0.7, 1.0),
#             colsample_bytree=trial.suggest_float("xgb_colsample", 0.7, 1.0),
#             reg_alpha=trial.suggest_float("xgb_alpha", 0.0, 3.0),
#             reg_lambda=trial.suggest_float("xgb_lambda", 0.5, 3.0),
#             tree_method="hist",
#             objective="reg:squarederror",
#             random_state=RANDOM_STATE,
#             n_jobs=-1,
#         )

#     return model_name, model


# def objective(trial):
#     model_name, model = make_model_from_trial(trial)

#     pipeline = Pipeline([
#         ("prep", preprocessor),
#         ("model", model),
#     ])

#     # 🔥 Speed optimization (IMPORTANT)
#     X_fit_sample = X_fit.sample(n=50000, random_state=42)
#     y_fit_sample = y_fit.loc[X_fit_sample.index]

#     if HAS_MLFLOW:
#         mlflow.start_run(nested=True)
#         mlflow.log_param("model_name", model_name)

#     pipeline.fit(X_fit_sample, y_fit_sample)
#     y_pred_valid = pipeline.predict(X_valid)

#     metrics = evaluate_metrics(y_valid, y_pred_valid)

#     if HAS_MLFLOW:
#         for k, v in metrics.items():
#             mlflow.log_metric(f"valid_{k}", v)
#         mlflow.log_params(model.get_params())
#         mlflow.end_run()

#     return metrics["MAPE"]
# --------------------
# ================= OPTUNA PIPELINE (FIXED) =================

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

transformers = []
if cat_cols:
    transformers.append(("cat", categorical_pipeline, cat_cols))
if num_cols:
    transformers.append(("num", numeric_pipeline, num_cols))

preprocessor = ColumnTransformer(
    transformers=transformers,
    remainder="drop",
)

def make_model_from_trial(trial):
    # 🔥 biased toward strong models
    model_name = trial.suggest_categorical(
        "model_name",
        ["XGBR", "XGBR", "RF", "RF", "GBR"]
    )

    if model_name == "RF":
        model = RandomForestRegressor(
            n_estimators=trial.suggest_int("rf_n_estimators", 100, 200, step=50),
            max_depth=trial.suggest_int("rf_max_depth", 8, 18),
            min_samples_leaf=trial.suggest_int("rf_min_samples_leaf", 2, 8),
            max_features=trial.suggest_float("rf_max_features", 0.5, 0.9),
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    elif model_name == "GBR":
        model = GradientBoostingRegressor(
            n_estimators=trial.suggest_int("gbr_n_estimators", 80, 160, step=40),
            learning_rate=trial.suggest_float("gbr_learning_rate", 0.01, 0.1, log=True),
            max_depth=trial.suggest_int("gbr_max_depth", 2, 5),
            subsample=trial.suggest_float("gbr_subsample", 0.7, 1.0),
            random_state=RANDOM_STATE,
        )

    else:  # XGBR
        model = XGBRegressor(
            n_estimators=trial.suggest_int("xgb_n_estimators", 200, 500, step=100),
            learning_rate=trial.suggest_float("xgb_learning_rate", 0.03, 0.15, log=True),
            max_depth=trial.suggest_int("xgb_max_depth", 3, 7),
            subsample=trial.suggest_float("xgb_subsample", 0.7, 1.0),
            colsample_bytree=trial.suggest_float("xgb_colsample", 0.7, 1.0),
            reg_alpha=trial.suggest_float("xgb_alpha", 0.0, 3.0),
            reg_lambda=trial.suggest_float("xgb_lambda", 0.5, 3.0),
            tree_method="hist",
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    return model_name, model


# def objective(trial):
#     model_name, model = make_model_from_trial(trial)

#     pipeline = Pipeline([
#         ("prep", preprocessor),
#         ("model", model),
#     ])

#     # 🔥 faster but stable
#     sample_size = min(50000, len(X_fit))
#     X_fit_sample = X_fit.sample(n=sample_size, random_state=42)
#     y_fit_sample = y_fit.loc[X_fit_sample.index]

#     pipeline.fit(X_fit_sample, y_fit_sample)
#     y_pred_valid = pipeline.predict(X_valid)

#     metrics = evaluate_metrics(y_valid, y_pred_valid)

#     return metrics["MAPE"]

# def objective(trial):
#     model_name, model = make_model_from_trial(trial)

#     pipeline = Pipeline([
#         ("prep", preprocessor),
#         ("model", model),
#     ])

#     sample_size = min(50000, len(X_fit))
#     X_fit_sample = X_fit.sample(n=sample_size, random_state=42)
#     y_fit_sample = y_fit.loc[X_fit_sample.index]

#     # ----------------------------✅ CHANGE 1: log transform
#     # y_log = np.log1p(y_fit_sample)

#     # try sqrt (EXPERIMENT)
#     y_log = np.sqrt(y_fit_sample)

#     pipeline.fit(X_fit_sample, y_log)

#     # ✅ CHANGE 2: reverse transform
#     y_pred_log = pipeline.predict(X_valid)
#     # y_pred = np.expm1(y_pred_log)
    
#     # ------------------------------------- try sqrt (EXPERIMENT)
#     y_pred = y_pred_log ** 2

#     # ✅ CHANGE 3: stable MAPE
#     return safe_mape(y_valid, y_pred)

def objective(trial):
    model_name, model = make_model_from_trial(trial)

    pipeline = Pipeline([
        ("prep", preprocessor),
        ("model", model),
    ])

    sample_size = min(50000, len(X_fit))
    X_fit_sample = X_fit.sample(n=sample_size, random_state=42)
    y_fit_sample = y_fit.loc[X_fit_sample.index]

    # 🔥 LOG TRANSFORM
    y_log = np.log1p(y_fit_sample)

    pipeline.fit(X_fit_sample, y_log)

    # predict
    y_pred_log = pipeline.predict(X_valid)
    y_pred = np.expm1(y_pred_log)

    return safe_mape(y_valid, y_pred)
# --------------------
# # Run Optuna model selection
# n_trials = 50

# study = optuna.create_study(
#     study_name="model_selection",
#     direction="minimize",
#     sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
# )

# if HAS_MLFLOW:
#     with mlflow.start_run(run_name="model_selection_optuna"):
#         study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True, catch=(ValueError,))
#         mlflow.log_params(study.best_params)
#         mlflow.log_metric("best_valid_MAPE", study.best_value)
# else:
#     study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True, catch=(ValueError,))

# print("Best validation MAPE:", round(study.best_value, 6))
# print("Best params:")
# study.best_params

# --------------------
# study.trials_dataframe().sort_values("value").head(10)
# --------------------
print("Final Features:", feature_cols)
print("Feature count:", len(feature_cols))

# 🔥 DEBUG CHECK (ADD THIS)
print("After FE:")
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
# --------------------
# # ================= SAVE BEST MODEL =================

# best_model_name = study.best_params["model_name"]

# # rebuild best model
# _, best_model = make_model_from_trial(optuna.trial.FixedTrial(study.best_params))

# final_pipeline = Pipeline([
#     ("prep", preprocessor),
#     ("model", best_model),
# ])

# # train on FULL training data
# final_pipeline.fit(X_train, y_train)

# # save
# joblib.dump(final_pipeline, "/kaggle/working/final_xgbr_0.179127.pkl")

# print("✅ Model saved successfully!")
# --------------------
import pandas as pd
import numpy as np

data_path = "/kaggle/input/notebooks/shubhamyadav74/notebookacca71d522/final_data.csv"
df = pd.read_csv(data_path)

def get_time_col(df):
    for c in ["pickup_slot", "tpep_pickup_datetime", "timestamp"]:
        if c in df.columns:
            return c
    return None

time_col = get_time_col(df)

df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)

split_date = df[time_col].max() - pd.Timedelta(days=15)

train_df = df[df[time_col] < split_date].copy()
test_df = df[df[time_col] >= split_date].copy()

print("Train:", train_df.shape)
print("Test:", test_df.shape)
# --------------------
target_candidates = ["total_pickups_model", "total_pickups_raw", "total_pickups"]
target_col = next((c for c in target_candidates if c in train_df.columns), None)

# TIME FEATURES
for df_ in [train_df, test_df]:
    df_["week_of_year"] = df_[time_col].dt.isocalendar().week.astype(int)
    df_["day_of_month"] = df_[time_col].dt.day
    df_["is_month_start"] = df_[time_col].dt.is_month_start.astype(int)
    df_["is_month_end"] = df_[time_col].dt.is_month_end.astype(int)

# LAG FEATURES
lag_steps = [1,2,3,6,12,24]
rolling_windows = [3,6]

def add_lag(df):
    for lag in lag_steps:
        df[f"lag_{lag}"] = df.groupby("region")[target_col].shift(lag)

    for w in rolling_windows:
        df[f"lag_roll_mean_{w}"] = df.groupby("region")[target_col].transform(
            lambda s: s.shift(1).rolling(w).mean()
        )
        df[f"lag_roll_std_{w}"] = df.groupby("region")[target_col].transform(
            lambda s: s.shift(1).rolling(w).std()
        )
    return df

train_df = add_lag(train_df)
lag_cols = [c for c in train_df.columns if c.startswith("lag_")]

# 🔥 SAFE DROP (avoid empty train)
train_df = train_df.dropna(subset=lag_cols)
if len(train_df) == 0:
    print("⚠️ fallback to fill")
    train_df = train_df.fillna(method="ffill").fillna(method="bfill")

train_df = train_df.reset_index(drop=True)

# TEST FIX
history = train_df.tail(200)
test_temp = pd.concat([history, test_df]).reset_index(drop=True)
test_temp = add_lag(test_temp)

test_df = test_temp.iloc[len(history):].copy()
test_df = test_df.fillna(method="ffill").fillna(method="bfill")

if len(test_df) == 0:
    raise ValueError("❌ Test empty after FE")

test_df = test_df.reset_index(drop=True)

print("After FE:", train_df.shape, test_df.shape)
# --------------------
train_df["trend_strength"] = train_df["lag_1"] - train_df["lag_roll_mean_3"]
test_df["trend_strength"] = test_df["lag_1"] - test_df["lag_roll_mean_3"]

train_df["region_hour"] = train_df["region"] * train_df["pickup_hour"]
test_df["region_hour"] = test_df["region"] * test_df["pickup_hour"]

global_mean = train_df[target_col].mean()
region_stats = train_df.groupby("region")[target_col].agg(["mean", "count"])

smooth = 20
region_map = (
    (region_stats["mean"] * region_stats["count"] + global_mean * smooth)
    / (region_stats["count"] + smooth)
)

train_df["region_mean"] = train_df["region"].map(region_map)
test_df["region_mean"] = test_df["region"].map(region_map)
# --------------------
feature_cols = [c for c in train_df.columns if c.startswith("lag_")] + [
    "region","pickup_hour","pickup_day_of_week",
    "week_of_year","day_of_month","is_month_start","is_month_end",
    "trend_strength","region_hour","region_mean"
]

# 🔥 SAFETY
feature_cols = [c for c in feature_cols if c in train_df.columns]

X_train = train_df[feature_cols]
y_train = train_df[target_col]

X_test = test_df[feature_cols]
y_test = test_df[target_col]

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
# --------------------
from xgboost import XGBRegressor

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.1026,
    max_depth=6,
    subsample=0.8295,
    colsample_bytree=0.9024,
    reg_alpha=1.2257,
    reg_lambda=0.8597,
    random_state=42,
    n_jobs=-1,
    tree_method="hist"
)

y_train_log = np.log1p(y_train)
model.fit(X_train, y_train_log)

y_pred_train = np.expm1(model.predict(X_train))
y_pred_test = np.expm1(model.predict(X_test))

y_pred_train = np.clip(y_pred_train, 0, None)
y_pred_test = np.clip(y_pred_test, 0, None)
# --------------------
predictions = pd.DataFrame({
    "actual_demand": y_test.values,
    "predicted_demand": y_pred_test,
}).reset_index(drop=True)

context_test = test_df.iloc[:len(predictions)].copy()

if time_col in context_test.columns:
    predictions["pickup_slot"] = pd.to_datetime(context_test[time_col].values)

if "region" in context_test.columns:
    predictions["region"] = context_test["region"].values

predictions["rolling_mean"] = predictions.groupby("region")["actual_demand"].transform("mean")
predictions["rolling_mean"] = predictions["rolling_mean"].fillna(predictions["rolling_mean"].median())

predictions["surge_threshold"] = predictions["rolling_mean"] * 1.3
predictions["surge_flag"] = predictions["predicted_demand"] > predictions["surge_threshold"]

ratio = predictions["predicted_demand"] / (predictions["surge_threshold"] + 1e-6)

predictions["surge_level"] = "none"
predictions.loc[(predictions["surge_flag"]) & (ratio <= 1.1), "surge_level"] = "low"
predictions.loc[(predictions["surge_flag"]) & (ratio > 1.1) & (ratio <= 1.25), "surge_level"] = "medium"
predictions.loc[(predictions["surge_flag"]) & (ratio > 1.25), "surge_level"] = "high"

predictions["rolling_std"] = predictions.groupby("region")["actual_demand"].transform("std").fillna(0)
predictions["risk_score"] = predictions["rolling_std"] / (predictions["rolling_mean"] + 1e-6)

predictions["risk_band"] = pd.cut(
    predictions["risk_score"],
    bins=[-np.inf, 0.35, 0.75, np.inf],
    labels=["Stable", "Moderate", "Volatile"]
).astype(str)

if "avg_fare_region_slot" in test_df.columns:
    predictions["avg_fare"] = context_test["avg_fare_region_slot"].values
elif "avg_fare_region_slot" in train_df.columns:
    predictions["avg_fare"] = train_df["avg_fare_region_slot"].median()
else:
    predictions["avg_fare"] = 10

predictions["expected_revenue"] = predictions["predicted_demand"] * predictions["avg_fare"]

baseline = predictions.groupby("region")["actual_demand"].transform("mean")
predictions["demand_pressure"] = predictions["predicted_demand"] / (baseline + 1e-6)

if "pickup_slot" in predictions.columns:
    predictions["pickup_hour"] = predictions["pickup_slot"].dt.hour
    predictions["pickup_day"] = predictions["pickup_slot"].dt.dayofweek

best_times = (
    predictions.groupby(["region", "pickup_hour"])["predicted_demand"]
    .mean().reset_index()
)

best_times = best_times.sort_values(["region","predicted_demand"],ascending=[True,False]).groupby("region").head(3)
best_times["rank"] = best_times.groupby("region").cumcount()+1

region_mean = predictions.groupby("region")["predicted_demand"].mean()
predictions["best_region"] = region_mean.idxmax()
predictions["relocation_gain"] = region_mean.max() - predictions["predicted_demand"]

print("✅ Intelligence layer built")
# --------------------
# ================= BUSINESS INTELLIGENCE =================

# ---- 1. SURGE DETECTION ----
predictions["baseline"] = predictions.groupby("region")["actual_demand"].transform("mean")
predictions["baseline"] = predictions["baseline"].fillna(predictions["baseline"].median())

predictions["surge_threshold"] = predictions["baseline"] * 1.3
predictions["surge_flag"] = predictions["predicted_demand"] > predictions["surge_threshold"]

ratio = predictions["predicted_demand"] / (predictions["surge_threshold"] + 1e-6)

predictions["surge_level"] = "none"
predictions.loc[(predictions["surge_flag"]) & (ratio <= 1.1), "surge_level"] = "low"
predictions.loc[(predictions["surge_flag"]) & (ratio > 1.1) & (ratio <= 1.25), "surge_level"] = "medium"
predictions.loc[(predictions["surge_flag"]) & (ratio > 1.25), "surge_level"] = "high"

# ---- 2. RISK SCORE ----
predictions["std"] = predictions.groupby("region")["actual_demand"].transform("std").fillna(0)

predictions["risk_score"] = predictions["std"] / (predictions["baseline"] + 1e-6)

predictions["risk_band"] = pd.cut(
    predictions["risk_score"],
    bins=[-np.inf, 0.35, 0.75, np.inf],
    labels=["Stable", "Moderate", "Volatile"]
)

# ---- 3. REVENUE ----
if "avg_fare_region_slot" in predictions.columns:
    predictions["avg_fare"] = predictions["avg_fare_region_slot"]
elif "avg_fare_region_slot" in train_df.columns:
    predictions["avg_fare"] = train_df["avg_fare_region_slot"].median()
else:
    predictions["avg_fare"] = 10

predictions["expected_revenue"] = predictions["predicted_demand"] * predictions["avg_fare"]

# ---- 4. DEMAND PRESSURE ----
predictions["demand_pressure"] = predictions["predicted_demand"] / (predictions["baseline"] + 1e-6)

print("✅ Business intelligence ready")
predictions.head()
# --------------------
# ================= DRIVER RELOCATION =================

recommendations = []

for _, row in predictions.iterrows():

    current_region = row["region"]
    current_time = row["pickup_slot"]
    current_demand = row["predicted_demand"]

    same_time = predictions[predictions["pickup_slot"] == current_time]

    better = same_time[same_time["predicted_demand"] > current_demand]

    if len(better) == 0:
        continue

    best_target = better.sort_values("predicted_demand", ascending=False).iloc[0]

    recommendations.append({
        "region": current_region,
        "pickup_slot": current_time,
        "recommended_region": best_target["region"],
        "expected_gain": best_target["predicted_demand"] - current_demand
    })

recommendations_df = pd.DataFrame(recommendations)

print("✅ Relocation ready")
recommendations_df.head()
# --------------------
# ================= BEST TIME =================

if np.issubdtype(predictions["pickup_slot"].dtype, np.datetime64):

    predictions["hour"] = predictions["pickup_slot"].dt.hour

    slot_profile = predictions.groupby(
        ["region", "hour"]
    )["predicted_demand"].mean().reset_index()

    best_times = (
        slot_profile.sort_values(
            ["region", "predicted_demand"],
            ascending=[True, False]
        )
        .groupby("region")
        .head(3)
    )

    best_times["rank"] = best_times.groupby("region").cumcount() + 1

    print("✅ Best time ready")
    best_times.head()
# --------------------
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def safe_mape(y_true, y_pred):
    y_true = np.maximum(y_true, 10)
    return np.mean(np.abs((y_true - y_pred) / y_true))

def smape(y_true, y_pred):
    denom = np.abs(y_true) + np.abs(y_pred)
    return np.mean(2*np.abs(y_true-y_pred)/np.where(denom==0,1,denom))

def evaluate_metrics(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE": float(safe_mape(y_true, y_pred)),
        "sMAPE": float(smape(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }

train_metrics = evaluate_metrics(y_train, y_pred_train)
test_metrics = evaluate_metrics(y_test, y_pred_test)

metrics_summary = pd.DataFrame([
    {"split":"train",**train_metrics},
    {"split":"test",**test_metrics},
])

print(metrics_summary)
# --------------------
from pathlib import Path
import joblib

base_path = Path("/kaggle/working/final_3")
base_path.mkdir(parents=True, exist_ok=True)

joblib.dump(model, base_path / "xgb_model.pkl")
metrics_summary.to_csv(base_path / "metrics_summary.csv", index=False)
predictions.to_csv(base_path / "predictions.csv", index=False)

ui_cols = [
    "pickup_slot","region","actual_demand","predicted_demand",
    "surge_flag","surge_level","risk_score","risk_band",
    "expected_revenue","demand_pressure",
]

predictions[[c for c in ui_cols if c in predictions.columns]].to_csv(
    base_path / "business_output.csv", index=False
)

if "recommendations_df" in globals() and not recommendations_df.empty:
    recommendations_df.to_csv(base_path / "driver_recommendations.csv", index=False)
else:
    pd.DataFrame().to_csv(base_path / "driver_recommendations.csv", index=False)

if "best_times" in globals() and not best_times.empty:
    best_times.to_csv(base_path / "best_times.csv", index=False)
else:
    pd.DataFrame().to_csv(base_path / "best_times.csv", index=False)

print("✅ All artifacts saved:", base_path)
# --------------------
import shutil

src = "/kaggle/input/notebooks/shubhamyadav74/notebookacca71d522/final_data.csv"
dst = "/kaggle/working/final_data.csv"

shutil.copy(src, dst)

print("✅ Copied to working directory")
# --------------------
