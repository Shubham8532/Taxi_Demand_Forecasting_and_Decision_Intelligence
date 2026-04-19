import pandas as pd

df = pd.read_csv("data/processed/final_data_csv.csv")

# detect time column
time_col = None
for c in ["pickup_slot", "tpep_pickup_datetime", "timestamp"]:
    if c in df.columns:
        time_col = c
        break

df[time_col] = pd.to_datetime(df[time_col])

# SIMPLE dataset (like old project)
ui_df = pd.DataFrame({
    "tpep_pickup_datetime": df[time_col],
    "region": df["region"],
    "lag_1": df.get("lag_1", 0),
    "lag_2": df.get("lag_2", 0),
    "lag_3": df.get("lag_3", 0),
    "lag_4": df.get("lag_6", 0),  # substitute
    "total_pickups": df.get("total_pickups_raw", 0),
    "avg_pickups": df.get("avg_pickups", 0),
    "day_of_week": df.get("pickup_day_of_week", 0)
})

ui_df = ui_df.dropna().sort_values("tpep_pickup_datetime")

ui_df.to_csv("data/processed/ui_dataset.csv", index=False)

print("✅ ui_dataset created")