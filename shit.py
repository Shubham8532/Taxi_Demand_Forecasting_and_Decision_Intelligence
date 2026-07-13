### 1. 2 hours ago
import joblib
import pandas as pd
import json
from flask import Flask, render_template, request, jsonify
from pathlib import Path
import datetime as dt
import traceback
import math
import warnings
import numpy as np

warnings.filterwarnings('ignore')

app = Flask(__name__)

# ================= GLOBAL CACHE =================
model_cache = None
data_cache = None
region_mapping = None

# ================= FEATURE LIST (🔥 CRITICAL) =================
# FEATURE_COLS = [
#     'region', 'pickup_hour', 'pickup_day_of_week',
#     'is_weekend', 'rush_hour', 'is_night',
#     'hour_sin', 'hour_cos',
#     'week_of_year', 'day_of_month',
#     'is_month_start', 'is_month_end',
#     'lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12', 'lag_24',
#     'lag_roll_mean_3', 'lag_roll_std_3',
#     'lag_roll_mean_6', 'lag_roll_std_6',
#     'trend_strength', 'is_peak',
#     'region_hour', 'region_mean',
#     'hour_mean', 'dow_mean'
# ]

FEATURE_COLS = [
    'lag_1','lag_2','lag_3','lag_6','lag_12','lag_24',
    'lag_roll_mean_3','lag_roll_std_3','lag_roll_mean_6','lag_roll_std_6',
    'region','pickup_hour','pickup_day_of_week',
    'week_of_year','day_of_month','is_month_start','is_month_end',
    'trend_strength','region_hour','region_mean'
]

# ================= REGION MAPPING =================
def load_region_mapping():
    global region_mapping
    if region_mapping is None:
        root_path = Path(__file__).parent
        with open(root_path / "region_mapping.json", 'r') as f:
            region_mapping = json.load(f)

        # convert keys to int
        region_mapping = {int(k): v for k, v in region_mapping.items()}

    return region_mapping

# ================= LOAD MODEL =================
def load_model():
    global model_cache
    if model_cache is None:
        root_path = Path(__file__).parent

        model_path = root_path / "models/xgb_model.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"❌ Model not found at {model_path}")

        model_cache = joblib.load(model_path)

        print("✅ Model loaded")

    return model_cache

# ================= LOAD DATA =================
def load_data():
    global data_cache

    if data_cache is None:
        root_path = Path(__file__).parent

        # plot data (for map)
        df_plot = pd.read_csv(root_path / "data/external/plot_data.csv")

        # main dataset
        df = pd.read_csv(root_path / "data/processed/final_data.csv")

        # detect time column
        time_col = None
        for c in ["pickup_slot", "tpep_pickup_datetime", "pickup_datetime", "timestamp"]:
            if c in df.columns:
                time_col = c
                break

        if time_col is None:
            raise ValueError(f"❌ No time column found. Columns: {df.columns}")

        # convert datetime
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.dropna(subset=[time_col])

        # sort + index
        df = df.sort_values(time_col).set_index(time_col)

        print("✅ Data loaded")
        # print("Time column:", time_col)
        # print("Columns:", list(df.columns))

        data_cache = (df_plot, df)

    return data_cache

# ================= COLORS =================
REGION_COLORS = [
    "#FF0000", "#FF4500", "#FF8C00", "#FFD700", "#ADFF2F",
    "#32CD32", "#008000", "#006400", "#00FF00", "#7CFC00",
    "#00FA9A", "#00FFFF", "#40E0D0", "#4682B4", "#1E90FF",
    "#0000FF", "#0000CD", "#8A2BE2", "#9932CC", "#BA55D3",
    "#FF00FF", "#FF1493", "#C71585", "#FF4500", "#FF6347",
    "#FFA07A", "#FFDAB9", "#FFE4B5", "#F5DEB3", "#EEE8AA"
]

def get_demand_color(region_id):
    return REGION_COLORS[region_id % len(REGION_COLORS)]

# ================= DISTANCE =================
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371  # km

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)

    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    return R * c

# ===================CALCULATE SPEED ================
AVG_SPEED_KMPH = 25

def calculate_eta(distance_km):
    """
    Estimate travel time between two demand zones.
    """
    eta_minutes = (distance_km / AVG_SPEED_KMPH) * 60

    return max(1, round(eta_minutes))

# ================= VALIDATION =================
def validate_coordinates(latitude, longitude):
    if not (40.60 <= latitude <= 40.85):
        raise ValueError("Latitude out of NYC bounds")

    if not (-74.05 <= longitude <= -73.70):
        raise ValueError("Longitude out of NYC bounds")

@app.route('/')
def index():
    return render_template('index.html')

# ================= GET REGIONS =================
@app.route('/get_regions', methods=['GET'])
def get_regions():
    try:
        regions = load_region_mapping()
        return jsonify({'success': True, 'regions': regions})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ================= SCATTER MAP =================
@app.route('/get_scatter_points', methods=['GET'])
def get_scatter_points():
    try:
        df_plot, _ = load_data()

        if len(df_plot) > 2000:
            df_sample = df_plot.sample(n=2000, random_state=42)
        else:
            df_sample = df_plot

        scatter_points = []
        for _, row in df_sample.iterrows():
            region_id = int(row['region'])

            scatter_points.append({
                'lat': float(row['pickup_latitude']),
                'lon': float(row['pickup_longitude']),
                'region': region_id,
                'color': get_demand_color(region_id)
            })

        return jsonify({'success': True, 'points': scatter_points})

    except Exception as e:
        print("Scatter error:", str(e))
        return jsonify({'success': False, 'error': str(e)}), 500

# ================= FIND REGION (UPDATED — NO KMEANS) =================
@app.route('/find_region', methods=['POST'])
def find_region():
    try:
        data = request.get_json()

        latitude = float(data['latitude'])
        longitude = float(data['longitude'])

        validate_coordinates(latitude, longitude)

        # 🔥 NEW: find nearest region from mapping
        regions = load_region_mapping()

        closest_region = None
        min_dist = float('inf')

        for rid, info in regions.items():
            dist = calculate_distance(
                latitude, longitude,
                info['lat'], info['lon']
            )

            if dist < min_dist:
                min_dist = dist
                closest_region = rid

        region_info = regions[closest_region]

        return jsonify({
            'success': True,
            'region_id': closest_region,
            'region_name': region_info['name'],
            'latitude': latitude,
            'longitude': longitude
        })

    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400

    except Exception as e:
        print("Find region error:", str(e))
        return jsonify({'success': False, 'error': 'Error finding region'}), 500

# @app.route('/get_available_times', methods=['POST'])
# def get_available_times():
#     try:
#         data = request.get_json()
#         date_str = data['date']

#         _, df = load_data()

#         date_obj = pd.to_datetime(date_str).date()

#         # 🔥 FIX: use index safely
#         df_temp = df.copy()
#         df_temp["date"] = df_temp.index.date
#         df_temp["time"] = df_temp.index.strftime('%H:%M')

#         available_times = df_temp[df_temp["date"] == date_obj]["time"].unique()

#         available_times = sorted(available_times)

#         return jsonify({
#             'success': True,
#             'times': list(available_times)
#         })

#     except Exception as e:
#         return jsonify({'success': False, 'error': str(e)}), 500
#     except Exception as e:
#         return jsonify({'success': False, 'error': str(e)}), 500

# temporary
# @app.route('/get_available_times', methods=['POST'])
# def get_available_times():
#     try:
#         _, df = load_data()

#         # get unique times
#         times = sorted(list(set(df.index.strftime("%H:%M"))))

#         return jsonify({'success': True, 'times': times[:50]})

#     except Exception as e:
#         return jsonify({'success': False, 'error': str(e)}), 500

# 

from datetime import timedelta
import pytz
from datetime import datetime

@app.route('/get_available_times', methods=['POST'])
def get_available_times():
    try:
        ist = pytz.timezone('Asia/Kolkata')
        now = datetime.now(ist)

        # round UP to next 15-min
        minutes = (now.minute // 15 + 1) * 15
        next_slot = now.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=minutes)

        times = []
        for i in range(50):
            slot = next_slot + timedelta(minutes=15 * i)
            times.append(slot.strftime("%H:%M"))

        return jsonify({'success': True, 'times': times})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ================= FEATURE ENGINE =================
def build_features_for_timestamp(df, timestamp):

    if timestamp not in df.index:
        raise ValueError("Timestamp not found")

    # STEP 1: past data
    df_past = df[df.index <= timestamp].copy()

    # STEP 2: sort
    df_past = df_past.sort_values(["region", df_past.index.name])

    # ========================
    # 🔥 ONLY REQUIRED FEATURES
    # ========================

    # LAGS
    for lag in [1,2,3,6,12,24]:
        df_past[f"lag_{lag}"] = df_past.groupby("region")["total_pickups_raw"].shift(lag)

    # ROLLING
    for w in [3,6]:
        df_past[f"lag_roll_mean_{w}"] = df_past.groupby("region")["total_pickups_raw"].transform(
            lambda s: s.shift(1).rolling(w).mean()
        )
        df_past[f"lag_roll_std_{w}"] = df_past.groupby("region")["total_pickups_raw"].transform(
            lambda s: s.shift(1).rolling(w).std()
        )

    # EXTRA (USED IN TRAINING)
    df_past["trend_strength"] = df_past["lag_1"] - df_past["lag_roll_mean_3"]
    df_past["region_hour"] = df_past["region"] * df_past["pickup_hour"]

    # REGION MEAN
    region_mean = df_past.groupby("region")["total_pickups_raw"].mean()
    df_past["region_mean"] = df_past["region"].map(region_mean)

    # TIME FEATURES
    df_past["week_of_year"] = df_past.index.isocalendar().week.astype(int)
    df_past["day_of_month"] = df_past.index.day
    df_past["is_month_start"] = df_past.index.is_month_start.astype(int)
    df_past["is_month_end"] = df_past.index.is_month_end.astype(int)

    # ========================
    # ❌ REMOVE THESE (NOT TRAINED)
    # ========================
    # is_peak ❌
    # hour_sin ❌
    # hour_cos ❌
    # hour_mean ❌
    # dow_mean ❌

    # STEP 4: latest per region
    current = df_past.groupby("region").tail(1).copy()

    # STEP 5: reset index (important)
    current = current.reset_index(drop=True)

    # STEP 6: EXACT FEATURE ALIGNMENT
    X = current[FEATURE_COLS].copy()
    X = X.fillna(0)

    return X, current


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        region_id = int(data['region_id'])
        date_str = data['date']
        time_str = data['time']

        timestamp = pd.Timestamp(f"{date_str} {time_str}")

        # ---------- LOAD ----------
        _, df = load_data()
        model = load_model()
        regions = load_region_mapping()

        # ---------- BUILD FEATURES ----------
        X_all, current = build_features_for_timestamp(df, timestamp)

        # ---------- PREDICT ----------
        y_pred_log = model.predict(X_all)
        all_predictions = np.expm1(y_pred_log)
        all_predictions = np.clip(all_predictions, 0, None)

        # ---------- BUILD REGION LIST ----------
        all_regions_info = []

        for idx, row in current.iterrows():
            rid = int(row["region"])

            predicted = int(all_predictions[idx])
            actual = int(row.get("total_pickups_raw", 0))

            info = regions.get(rid, {"name": f"Region {rid}", "lat": 0, "lon": 0})

            all_regions_info.append({
                'region_id': rid,
                'name': info['name'],
                'lat': info['lat'],
                'lon': info['lon'],
                'predicted_demand': predicted,
                'actual_demand': actual,
                'color': get_demand_color(rid),
                'is_selected': (rid == region_id)
            })

        # ---------- SORT ----------
        sorted_regions = sorted(all_regions_info, key=lambda x: x['predicted_demand'], reverse=True)
        top_5_zones = sorted_regions[:5]
        bottom_5_zones = sorted_regions[-5:]

        # ---------- SELECTED ----------
        selected = next(r for r in all_regions_info if r['region_id'] == region_id)
        row_sel = current[current['region'] == region_id].iloc[0]

        # ---------- LAGS ----------
        lag_1 = int(row_sel.get('lag_1', 0))
        lag_2 = int(row_sel.get('lag_2', 0))
        lag_3 = int(row_sel.get('lag_3', 0))
        lag_4 = int(row_sel.get('lag_6', 0))  # use lag_6 as "older" proxy

        # ---------- TREND ----------
        if lag_1 > lag_4:
            trend = "increasing"
            trend_icon = "↑"
        elif lag_1 < lag_4:
            trend = "decreasing"
            trend_icon = "↓"
        else:
            trend = "stable"
            trend_icon = "→"

        # ---------- CONFIDENCE ----------
        error = abs(selected['predicted_demand'] - selected['actual_demand'])
        error_percent = round((error / (selected['actual_demand'] + 20) * 100), 1)

        if error_percent < 10:
            confidence = "High"
        elif error_percent < 20:
            confidence = "Medium"
        else:
            confidence = "Low"

        next_timestamp = timestamp + dt.timedelta(minutes=15)

        # ==============================
        # 🔥 LAST HOUR PATTERN FIX
        # ==============================
        time_labels = [
            (timestamp - dt.timedelta(minutes=60)).strftime('%H:%M'),
            (timestamp - dt.timedelta(minutes=45)).strftime('%H:%M'),
            (timestamp - dt.timedelta(minutes=30)).strftime('%H:%M'),
            (timestamp - dt.timedelta(minutes=15)).strftime('%H:%M')
        ]

        last_hour_values = [
            int(row_sel.get('lag_4', 0)),
            int(row_sel.get('lag_3', 0)),
            int(row_sel.get('lag_2', 0)),
            int(row_sel.get('lag_1', 0))
        ]

        # ==============================
        # 🔥 PEAK TIME FIX
        # ==============================
        peak_time = row_sel.get('best_time_window', None)
        peak_demand = selected['predicted_demand']

        # ==============================
        # 🔥 MULTI-RECOMMENDATION FIX
        # ==============================
        current_region_info = regions[region_id]
        current_lat = current_region_info['lat']
        current_lon = current_region_info['lon']

        selected_demand = selected['predicted_demand']
        better_zones = []

        for zone in all_regions_info:
            if zone['region_id'] == region_id:
                continue

            if zone['predicted_demand'] > selected_demand:
                zone_info = regions[zone['region_id']]

                distance = calculate_distance(
                    current_lat, current_lon,
                    zone_info['lat'], zone_info['lon']
                )

                gain = zone['predicted_demand'] - selected_demand



                better_zones.append({
                    "region_id": zone['region_id'],
                    "name": zone['name'],
                    "lat": zone_info['lat'],
                    "lon": zone_info['lon'],
                    "predicted_demand": zone['predicted_demand'],
                    "distance_km": round(distance, 2),
                    "eta_minutes": calculate_eta(distance),
                    "expected_gain": gain,
                    "recommendation_score": 0
                })

        # better_zones.sort(key=lambda x: (-x['expected_gain'], x['distance_km']))
        # recommendations = better_zones[:5]

        # ===========================================
        # SMART RECOMMENDATION ENGINE
        # ===========================================

        if better_zones:

            max_demand = max(z["predicted_demand"] for z in better_zones)
            min_demand = min(z["predicted_demand"] for z in better_zones)

            max_eta = max(z["eta_minutes"] for z in better_zones)
            min_eta = min(z["eta_minutes"] for z in better_zones)

            max_distance = max(z["distance_km"] for z in better_zones)
            min_distance = min(z["distance_km"] for z in better_zones)

            for zone in better_zones:

                # --------------------
                # Demand (higher better)
                # --------------------
                if max_demand == min_demand:
                    demand_score = 100
                else:
                    demand_score = (
                        (zone["predicted_demand"] - min_demand)
                        / (max_demand - min_demand)
                    ) * 100

                # --------------------
                # ETA (lower better)
                # --------------------
                if max_eta == min_eta:
                    eta_score = 100
                else:
                    eta_score = (
                        (max_eta - zone["eta_minutes"])
                        / (max_eta - min_eta)
                    ) * 100

                # --------------------
                # Distance (lower better)
                # --------------------
                if max_distance == min_distance:
                    distance_score = 100
                else:
                    distance_score = (
                        (max_distance - zone["distance_km"])
                        / (max_distance - min_distance)
                    ) * 100

                # --------------------
                # Final Recommendation Score
                # --------------------
                zone["recommendation_score"] = round(

                    demand_score * 0.60 +

                    eta_score * 0.25 +

                    distance_score * 0.15

                ,1)

        better_zones.sort(
            key=lambda x: x["recommendation_score"],
            reverse=True
        )

        recommendations = better_zones[:5]

        # ========= RECOMMENDATION BADGE =========
        badge_names = [
            "Best Choice",
            "Recommended",
            "Good Option",
            "Alternative",
            "Consider"
        ]

        for i, zone in enumerate(recommendations):
            if i < len(badge_names):
                zone["badge"] = badge_names[i]
            else:
                zone["badge"] = "📌 Consider"

        # ==============================
        # ---------- FINAL RESPONSE ----------
        # ==============================
        response = {
            'success': True,
            'region_id': region_id,
            'region_name': selected['name'],

            'prediction': {
                'predicted_demand': selected['predicted_demand'],
                'actual_demand': selected['actual_demand'],
                'error': error,
                'error_percent': error_percent,
                'confidence': confidence,
                'color': selected['color'],
                'next_interval': next_timestamp.strftime('%I:%M %p'),
                'current_time': timestamp.strftime('%I:%M %p')
            },

            'features': {
                'lag_values': [lag_1, lag_2, lag_3, lag_4],
                'avg_pickups': selected['actual_demand'],
                'day_of_week': "",
                'trend': trend,
                'trend_icon': trend_icon
            },

            'peak_hours': {
                'time': peak_time,
                'demand': peak_demand
            },

            'time_labels': time_labels,
            'last_hour_values': last_hour_values,

            'all_regions': all_regions_info,
            'top_zones': top_5_zones,
            'low_zones': bottom_5_zones,
            'recommendations': recommendations
        }

        # print("🚀 RESPONSE:", response)

        return jsonify(response)

    except Exception as e:
        print("🔥 ERROR:", str(e))
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500
        # DEBUG
        # print("🚀 RESPONSE:", response)

        return jsonify(response)

    except Exception as e:
        print("🔥 ERROR:", str(e))
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/predict_all_regions', methods=['POST'])
def predict_all_regions():
    """Get predictions for all regions"""
    try:
        data = request.get_json()
        date_str = data['date']
        time_str = data['time']

        timestamp = pd.Timestamp(f"{date_str} {time_str}")

        # load
        _, df = load_data()
        model = load_model()
        regions = load_region_mapping()

        # ---------- BUILD FEATURES ----------
        X_all, current = build_features_for_timestamp(df, timestamp)

        # ---------- PREDICT ----------
        y_pred_log = model.predict(X_all)
        preds = np.expm1(y_pred_log)
        preds = np.clip(preds, 0, None)

        # ---------- BUILD RESPONSE ----------
        results = []

        for idx, row in current.iterrows():
            region_id = int(row["region"])

            predicted = int(preds[idx])
            actual = int(row.get("total_pickups_raw", 0))

            info = regions.get(region_id, {
                "name": f"Region {region_id}",
                "lat": 0,
                "lon": 0
            })

            results.append({
                'region_id': region_id,
                'name': info['name'],
                'lat': info['lat'],
                'lon': info['lon'],
                'predicted_demand': predicted,
                'actual_demand': actual,
                'color': get_demand_color(region_id)
            })

        return jsonify({'success': True, 'regions': results})

    except Exception as e:
        print("🔥 ERROR:", str(e))
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500


# ================= RUN =================
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=False)