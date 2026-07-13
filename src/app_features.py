import pandas as pd
from src.app_config import FEATURE_COLS

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