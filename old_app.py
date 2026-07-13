import geopandas as gpd

zones = gpd.read_file("data/taxi_zones/taxi_zones.shp")
print(zones.columns)