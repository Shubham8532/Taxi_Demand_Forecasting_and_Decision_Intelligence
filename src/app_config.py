# ================= FEATURE LIST =================
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

from turtle import pd


FEATURE_COLS = [
    'lag_1','lag_2','lag_3','lag_6','lag_12','lag_24',
    'lag_roll_mean_3','lag_roll_std_3','lag_roll_mean_6','lag_roll_std_6',
    'region','pickup_hour','pickup_day_of_week',
    'week_of_year','day_of_month','is_month_start','is_month_end',
    'trend_strength','region_hour','region_mean'
]

# ================= COLORS =================
REGION_COLORS = [
    "#FF0000", "#FF4500", "#FF8C00", "#FFD700", "#ADFF2F",
    "#32CD32", "#008000", "#006400", "#00FF00", "#7CFC00",
    "#00FA9A", "#00FFFF", "#40E0D0", "#4682B4", "#1E90FF",
    "#0000FF", "#0000CD", "#8A2BE2", "#9932CC", "#BA55D3",
    "#FF00FF", "#FF1493", "#C71585", "#FF4500", "#FF6347",
    "#FFA07A", "#FFDAB9", "#FFE4B5", "#F5DEB3", "#EEE8AA"
]


# ===================CALCULATE SPEED ================
AVG_SPEED_KMPH = 25