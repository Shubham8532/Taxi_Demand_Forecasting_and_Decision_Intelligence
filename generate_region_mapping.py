import json
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# ---------- PATHS ----------
PLOT_DATA = "data/external/plot_data.csv"
SHAPEFILE = "data/taxi_zones/taxi_zones.shp"
OLD_JSON = "region_mapping.json"
OUTPUT_JSON = "region_mapping_new.json"

# ---------- STEP 1 ----------
print("Loading plot_data...")

centroids = (
    pd.read_csv(
        PLOT_DATA,
        usecols=["pickup_latitude", "pickup_longitude", "region"]
    )
    .groupby("region", as_index=False)
    .agg({
        "pickup_latitude": "mean",
        "pickup_longitude": "mean"
    })
)

centroids.rename(columns={
    "pickup_latitude": "lat",
    "pickup_longitude": "lon"
}, inplace=True)

print(f"Computed {len(centroids)} centroids")

# ---------- STEP 2 ----------
print("Loading taxi zones...")

zones = gpd.read_file(SHAPEFILE)

print("Zones CRS:", zones.crs)

# Convert centroid dataframe to GeoDataFrame
points = gpd.GeoDataFrame(
    centroids,
    geometry=gpd.points_from_xy(
        centroids.lon,
        centroids.lat
    ),
    crs="EPSG:4326"
)

# Convert the points into the same CRS as the taxi zones
points = points.to_crs(zones.crs)


# ---------- STEP 3 ----------
print("Matching centroids to taxi zones...")

# First try polygon containment
joined = gpd.sjoin(
    points,
    zones[["zone", "borough", "geometry"]],
    how="left",
    predicate="within"
)

# For centroids not inside any polygon, use nearest zone
missing = joined["zone"].isna()

if missing.any():
    nearest = gpd.sjoin_nearest(
        points[missing],
        zones[["zone", "borough", "geometry"]],
        how="left"
    )

    joined.loc[missing, "zone"] = nearest["zone"].values
    joined.loc[missing, "borough"] = nearest["borough"].values

print("Matched", len(joined), "regions")

print(joined[["region", "lat", "lon", "zone", "borough"]])
print(zones[["LocationID", "zone", "borough"]].head(20))
print("Taxi zone bounds:", zones.total_bounds)
print("Point bounds:", points.total_bounds)

# ---------- STEP 4 ----------
print("Creating new mapping...")

with open(OLD_JSON, "r") as f:
    old = json.load(f)

new_mapping = {}

for _, row in joined.iterrows():

    rid = str(int(row["region"]))

    entry = old[rid].copy()

    lat = float(row["lat"])
    lon = float(row["lon"])

    # Update centroid
    entry["lat"] = lat
    entry["lon"] = lon

    # Update official zone info
    entry["name"] = f'{row["borough"]} - {row["zone"]}'
    entry["borough"] = row["borough"]
    entry["neighborhood"] = row["zone"]

    # Create new boundary around centroid
    d = 0.008

    entry["boundary"] = [
        [lat + d, lon],
        [lat + d * 0.7, lon + d * 0.7],
        [lat, lon + d],
        [lat - d * 0.7, lon + d * 0.7],
        [lat - d, lon],
        [lat - d * 0.7, lon - d * 0.7],
        [lat, lon - d],
        [lat + d * 0.7, lon - d * 0.7]
    ]

    new_mapping[rid] = entry

# ---------- STEP 5 ----------
with open(OUTPUT_JSON, "w") as f:
    json.dump(new_mapping, f, indent=2)

print("Done.")
print("Saved as:", OUTPUT_JSON)

