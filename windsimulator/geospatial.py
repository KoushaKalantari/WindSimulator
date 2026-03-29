from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import FICTIONAL_NEIGHBORHOODS

KM_PER_DEG_LAT = 110.574


@dataclass(frozen=True)
class GeoNeighborhood:
    name: str
    latitude: float
    longitude: float


def km_per_deg_lon(latitude: float) -> float:
    return 111.320 * np.cos(np.deg2rad(latitude))


def latlon_to_local_km(
    latitude,
    longitude,
    origin_latitude: float,
    origin_longitude: float,
):
    x_km = (np.asarray(longitude) - origin_longitude) * km_per_deg_lon(origin_latitude)
    y_km = (np.asarray(latitude) - origin_latitude) * KM_PER_DEG_LAT
    return x_km, y_km


def local_km_to_latlon(
    x_km,
    y_km,
    origin_latitude: float,
    origin_longitude: float,
):
    latitude = origin_latitude + (np.asarray(y_km) / KM_PER_DEG_LAT)
    longitude = origin_longitude + (np.asarray(x_km) / km_per_deg_lon(origin_latitude))
    return latitude, longitude


def make_latlon_grid(
    origin_latitude: float,
    origin_longitude: float,
    x_km,
    y_km,
):
    latitude_grid, longitude_grid = local_km_to_latlon(
        x_km,
        y_km,
        origin_latitude=origin_latitude,
        origin_longitude=origin_longitude,
    )
    return latitude_grid, longitude_grid


def anchor_demo_neighborhoods(
    origin_latitude: float,
    origin_longitude: float,
    offsets_km: dict[str, tuple[float, float]] = FICTIONAL_NEIGHBORHOODS,
) -> dict[str, tuple[float, float]]:
    neighborhoods: dict[str, tuple[float, float]] = {}
    for name, (x_km, y_km) in offsets_km.items():
        latitude, longitude = local_km_to_latlon(
            x_km,
            y_km,
            origin_latitude=origin_latitude,
            origin_longitude=origin_longitude,
        )
        neighborhoods[name] = (float(latitude), float(longitude))
    return neighborhoods


def normalize_geo_neighborhoods(
    neighborhoods: dict[str, tuple[float, float]] | None,
    origin_latitude: float,
    origin_longitude: float,
) -> dict[str, tuple[float, float]]:
    if neighborhoods is None:
        return anchor_demo_neighborhoods(origin_latitude, origin_longitude)
    return neighborhoods
