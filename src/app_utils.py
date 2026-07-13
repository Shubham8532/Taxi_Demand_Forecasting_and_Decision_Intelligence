import math
from src.app_config import REGION_COLORS, AVG_SPEED_KMPH

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