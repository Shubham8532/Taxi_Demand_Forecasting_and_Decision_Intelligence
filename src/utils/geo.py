"""Geo-spatial utility functions.

Haversine distance, coordinate validation, and NYC boundary checks
used by clustering (notebook 3) and the Flask app's region lookup.
"""

from __future__ import annotations

import math

import numpy as np

# NYC bounding box used across notebooks 2 & 3
NYC_BOUNDS = {
    "min_lat": 40.55,
    "max_lat": 40.95,
    "min_lon": -74.10,
    "max_lon": -73.65,
}

# Tighter bounds used by the app for input validation
APP_BOUNDS = {
    "min_lat": 40.60,
    "max_lat": 40.85,
    "min_lon": -74.05,
    "max_lon": -73.70,
}


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance in **kilometres** between two points.

    Uses the Haversine formula — identical to notebook 3's
    ``haversine_distance`` and app.py's ``calculate_distance``.

    Parameters
    ----------
    lat1, lon1 : float
        Latitude/longitude of point A in degrees.
    lat2, lon2 : float
        Latitude/longitude of point B in degrees.

    Returns
    -------
    float
        Distance in kilometres.
    """
    R = 6371.0088  # Earth radius in km (same constant as notebook 3)

    lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
    lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)

    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def haversine_distance_vectorised(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """Vectorised Haversine for numpy arrays (used in clustering notebook 3).

    Parameters
    ----------
    lat1, lon1, lat2, lon2 : np.ndarray
        Coordinates in degrees.

    Returns
    -------
    np.ndarray
        Distances in kilometres.
    """
    R = 6371.0088
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return R * c


def pairwise_haversine_matrix(centroids_lat_lon: np.ndarray) -> np.ndarray:
    """N×N pairwise distance matrix from an (N, 2) array of [lat, lon].

    Reproduces notebook 3's ``pairwise_haversine_matrix``.

    Parameters
    ----------
    centroids_lat_lon : np.ndarray, shape (N, 2)
        Each row is ``[latitude, longitude]`` in degrees.

    Returns
    -------
    np.ndarray, shape (N, N)
        Symmetric distance matrix in kilometres.
    """
    lat = np.radians(centroids_lat_lon[:, 0])
    lon = np.radians(centroids_lat_lon[:, 1])

    lat1 = lat[:, None]
    lat2 = lat[None, :]
    lon1 = lon[:, None]
    lon2 = lon[None, :]

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return 6371.0088 * c


def validate_coordinates(latitude: float, longitude: float) -> None:
    """Raise ``ValueError`` if coordinates fall outside NYC app bounds.

    Parameters
    ----------
    latitude : float
    longitude : float

    Raises
    ------
    ValueError
        If the point is outside the bounding box.
    """
    if not (APP_BOUNDS["min_lat"] <= latitude <= APP_BOUNDS["max_lat"]):
        raise ValueError(
            f"Latitude {latitude} out of NYC bounds "
            f"[{APP_BOUNDS['min_lat']}, {APP_BOUNDS['max_lat']}]"
        )
    if not (APP_BOUNDS["min_lon"] <= longitude <= APP_BOUNDS["max_lon"]):
        raise ValueError(
            f"Longitude {longitude} out of NYC bounds "
            f"[{APP_BOUNDS['min_lon']}, {APP_BOUNDS['max_lon']}]"
        )
