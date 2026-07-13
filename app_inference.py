"""Flask application for NYC Taxi Demand Prediction.

This is the Azure-deployable version that uses pre-computed region
stats instead of loading the full CSV at runtime.

Required files (in project root):
    - models/xgb_model.pkl              (~2.3 MB)
    - models/region_inference_stats.pkl  (~6 KB)
    - region_mapping.json               (~25 KB)

To generate the stats file, run:
    python scripts/build_inference_stats.py
"""

from __future__ import annotations

import json
import math
import traceback
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytz
from flask import Flask, jsonify, render_template, request

from src.inference.realtime import RealtimePredictor
from src.utils.geo import haversine_distance, validate_coordinates

warnings.filterwarnings("ignore")

app = Flask(__name__)

# ================= PATHS =================
ROOT_PATH = Path(__file__).parent
MODEL_PATH = ROOT_PATH / "models" / "xgb_model.pkl"
STATS_PATH = ROOT_PATH / "models" / "region_inference_stats.pkl"
REGION_MAPPING_PATH = ROOT_PATH / "region_mapping.json"

# ================= PREDICTOR (lazy-loaded, cached) =================
predictor = RealtimePredictor(
    model_path=MODEL_PATH,
    stats_path=STATS_PATH,
    region_mapping_path=REGION_MAPPING_PATH,
)

# ================= REGION COLORS =================
REGION_COLORS = [
    "#FF0000", "#FF4500", "#FF8C00", "#FFD700", "#ADFF2F",
    "#32CD32", "#008000", "#006400", "#00FF00", "#7CFC00",
    "#00FA9A", "#00FFFF", "#40E0D0", "#4682B4", "#1E90FF",
    "#0000FF", "#0000CD", "#8A2BE2", "#9932CC", "#BA55D3",
    "#FF00FF", "#FF1493", "#C71585", "#FF4500", "#FF6347",
    "#FFA07A", "#FFDAB9", "#FFE4B5", "#F5DEB3", "#EEE8AA",
]


def get_demand_color(region_id: int) -> str:
    """Return a consistent color for a region ID."""
    return REGION_COLORS[region_id % len(REGION_COLORS)]


# ================= ROUTES =================

@app.route("/")
def index():
    """Serve the main dashboard."""
    return render_template("index.html")


@app.route("/get_regions", methods=["GET"])
def get_regions():
    """Return all region centroids and names."""
    try:
        regions = predictor.region_mapping
        return jsonify({"success": True, "regions": regions})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/get_scatter_points", methods=["GET"])
def get_scatter_points():
    """Return scatter points for map visualisation.

    Uses region centroids with slight random jitter to create
    a scatter effect without needing the full plot_data CSV.
    """
    try:
        regions = predictor.region_mapping
        scatter_points = []
        rng = np.random.default_rng(42)

        for rid_str, info in regions.items():
            rid = int(rid_str)
            # Generate ~60 jittered points per region for visualisation
            n_points = 60
            lats = info["lat"] + rng.normal(0, 0.008, n_points)
            lons = info["lon"] + rng.normal(0, 0.008, n_points)

            for lat, lon in zip(lats, lons):
                scatter_points.append({
                    "lat": float(lat),
                    "lon": float(lon),
                    "region": rid,
                    "color": get_demand_color(rid),
                })

        # Sample down if too many
        if len(scatter_points) > 2000:
            indices = rng.choice(len(scatter_points), 2000, replace=False)
            scatter_points = [scatter_points[i] for i in indices]

        return jsonify({"success": True, "points": scatter_points})

    except Exception as e:
        print("Scatter error:", str(e))
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/find_region", methods=["POST"])
def find_region():
    """Find the nearest region to a given lat/lon coordinate."""
    try:
        data = request.get_json()
        latitude = float(data["latitude"])
        longitude = float(data["longitude"])

        validate_coordinates(latitude, longitude)

        regions = predictor.region_mapping
        closest_region = None
        min_dist = float("inf")

        for rid_str, info in regions.items():
            dist = haversine_distance(
                latitude, longitude, info["lat"], info["lon"]
            )
            if dist < min_dist:
                min_dist = dist
                closest_region = rid_str

        region_info = regions[closest_region]
        return jsonify({
            "success": True,
            "region_id": closest_region,
            "region_name": region_info["name"],
            "latitude": latitude,
            "longitude": longitude,
        })

    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        print("Find region error:", str(e))
        return jsonify({"success": False, "error": "Error finding region"}), 500


@app.route("/get_available_times", methods=["POST"])
def get_available_times():
    """Generate 15-minute time slots starting from the next quarter-hour."""
    try:
        ist = pytz.timezone("Asia/Kolkata")
        now = datetime.now(ist)

        minutes = (now.minute // 15 + 1) * 15
        next_slot = now.replace(minute=0, second=0, microsecond=0) + timedelta(
            minutes=minutes
        )

        times = [
            (next_slot + timedelta(minutes=15 * i)).strftime("%H:%M")
            for i in range(50)
        ]
        return jsonify({"success": True, "times": times})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    """Predict demand for a specific region, date, and time.

    This is the main prediction endpoint. It uses the
    RealtimePredictor which needs NO CSV — only the model
    and pre-computed stats.
    """
    try:
        data = request.get_json()
        region_id = int(data["region_id"])
        date_str = data["date"]
        time_str = data["time"]
        timestamp = pd.Timestamp(f"{date_str} {time_str}")

        regions = predictor.region_mapping

        # Predict ALL regions at this timestamp
        all_results = predictor.predict_all_regions(timestamp)

        # Enrich with colors and selection flag
        all_regions_info = []
        for r in all_results:
            r["color"] = get_demand_color(r["region_id"])
            r["actual_demand"] = r["predicted_demand"]  # no actual at inference
            r["is_selected"] = (r["region_id"] == region_id)
            all_regions_info.append(r)

        # Sort for top/bottom zones
        sorted_regions = sorted(
            all_regions_info, key=lambda x: x["predicted_demand"], reverse=True
        )
        top_5_zones = sorted_regions[:5]
        bottom_5_zones = sorted_regions[-5:]

        # Selected region
        selected = next(
            (r for r in all_regions_info if r["region_id"] == region_id),
            sorted_regions[0],
        )

        # Build lag-based features for trend info
        selected_demand = selected["predicted_demand"]

        # Get lag values from stats for the selected region
        stats = predictor._stats  # safe after _ensure_loaded
        region_info = stats.get(region_id, {})
        last_vals = region_info.get("last_values", [0, 0, 0, 0])

        lag_1 = int(last_vals[-1]) if len(last_vals) >= 1 else 0
        lag_2 = int(last_vals[-2]) if len(last_vals) >= 2 else 0
        lag_3 = int(last_vals[-3]) if len(last_vals) >= 3 else 0
        lag_6 = int(last_vals[-6]) if len(last_vals) >= 6 else 0

        # Trend
        if lag_1 > lag_6:
            trend, trend_icon = "increasing", "↑"
        elif lag_1 < lag_6:
            trend, trend_icon = "decreasing", "↓"
        else:
            trend, trend_icon = "stable", "→"

        # Confidence
        error = abs(selected_demand - lag_1) if lag_1 > 0 else 0
        error_percent = round((error / (lag_1 + 20)) * 100, 1)

        if error_percent < 10:
            confidence = "High"
        elif error_percent < 20:
            confidence = "Medium"
        else:
            confidence = "Low"

        next_timestamp = timestamp + timedelta(minutes=15)

        # Time labels for chart
        time_labels = [
            (timestamp - timedelta(minutes=m)).strftime("%H:%M")
            for m in [60, 45, 30, 15]
        ]
        last_hour_values = [
            int(last_vals[-4]) if len(last_vals) >= 4 else 0,
            int(last_vals[-3]) if len(last_vals) >= 3 else 0,
            int(last_vals[-2]) if len(last_vals) >= 2 else 0,
            int(last_vals[-1]) if len(last_vals) >= 1 else 0,
        ]

        # Best time window
        peak_time = region_info.get("best_time_window", None)
        peak_demand = selected_demand

        # Recommendations: nearby zones with higher demand
        current_info = regions.get(str(region_id), {"lat": 0, "lon": 0})
        current_lat = current_info.get("lat", 0)
        current_lon = current_info.get("lon", 0)

        better_zones = []
        for zone in all_regions_info:
            if zone["region_id"] == region_id:
                continue
            if zone["predicted_demand"] > selected_demand:
                zone_geo = regions.get(
                    str(zone["region_id"]), {"lat": 0, "lon": 0}
                )
                distance = haversine_distance(
                    current_lat, current_lon,
                    zone_geo.get("lat", 0), zone_geo.get("lon", 0),
                )
                gain = zone["predicted_demand"] - selected_demand

                # remove
                print("Current:", current_region_info)

                print("Zone:", zone_info)

                print(
                    zone["name"],
                    zone_info["lat"],
                    zone_info["lon"],
                    distance
                ) 
                #               
                better_zones.append({
                    "region_id": zone["region_id"],
                    "name": zone["name"],
                    "lat": zone_geo.get("lat", 0),
                    "lon": zone_geo.get("lon", 0),
                    "predicted_demand": zone["predicted_demand"],
                    "distance_km": round(distance, 2),
                    "expected_gain": gain,
                    "color": zone["color"],
                })

        better_zones.sort(key=lambda x: (-x["expected_gain"], x["distance_km"]))
        recommendations = better_zones[:5]

        # Build response (same structure as original app.py)
        response = {
            "success": True,
            "region_id": region_id,
            "region_name": selected["name"],
            "prediction": {
                "predicted_demand": selected_demand,
                "actual_demand": lag_1,  # use last known as proxy
                "error": error,
                "error_percent": error_percent,
                "confidence": confidence,
                "color": selected["color"],
                "next_interval": next_timestamp.strftime("%I:%M %p"),
                "current_time": timestamp.strftime("%I:%M %p"),
            },
            "features": {
                "lag_values": [lag_1, lag_2, lag_3, lag_6],
                "avg_pickups": lag_1,
                "day_of_week": "",
                "trend": trend,
                "trend_icon": trend_icon,
            },
            "peak_hours": {
                "time": peak_time,
                "demand": peak_demand,
            },
            "time_labels": time_labels,
            "last_hour_values": last_hour_values,
            "all_regions": all_regions_info,
            "top_zones": top_5_zones,
            "low_zones": bottom_5_zones,
            "recommendations": recommendations,
        }

        return jsonify(response)

    except Exception as e:
        print("ERROR:", str(e))
        print(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/predict_all_regions", methods=["POST"])
def predict_all_regions():
    """Get predictions for all regions at a given time."""
    try:
        data = request.get_json()
        date_str = data["date"]
        time_str = data["time"]
        timestamp = pd.Timestamp(f"{date_str} {time_str}")

        results = predictor.predict_all_regions(timestamp)

        for r in results:
            r["color"] = get_demand_color(r["region_id"])
            r["actual_demand"] = r["predicted_demand"]

        return jsonify({"success": True, "regions": results})

    except Exception as e:
        print("ERROR:", str(e))
        print(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


# ================= RUN =================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)