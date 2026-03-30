from __future__ import annotations

import json
import os
import ssl
import time
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from urllib.parse import urlencode
from urllib.request import Request, urlopen

try:
    import certifi
except Exception:
    certifi = None


NOMINATIM_REVERSE_URL = "https://nominatim.openstreetmap.org/reverse"
DEFAULT_USER_AGENT = "WindSimulator/0.1 (prototype hazard demo)"
_LAST_NOMINATIM_REQUEST_TS = 0.0


@dataclass(frozen=True)
class ReverseGeocodeResult:
    latitude: float
    longitude: float
    display_name: str
    road: str | None
    neighborhood: str | None
    city: str | None
    postcode: str | None
    state: str | None
    country: str | None


@dataclass(frozen=True)
class ReferenceLabel:
    label: str
    latitude: float
    longitude: float
    kind: str


def _ssl_context():
    return ssl.create_default_context(cafile=certifi.where() if certifi is not None else None)


def _nominatim_headers() -> dict[str, str]:
    return {
        "User-Agent": os.environ.get("NOMINATIM_USER_AGENT", DEFAULT_USER_AGENT),
        "Accept-Language": "en",
    }


def _respect_nominatim_rate_limit() -> None:
    global _LAST_NOMINATIM_REQUEST_TS
    now = time.time()
    wait_s = 1.0 - (now - _LAST_NOMINATIM_REQUEST_TS)
    if wait_s > 0:
        time.sleep(wait_s)
    _LAST_NOMINATIM_REQUEST_TS = time.time()


def _pick_first(address: dict, *keys: str) -> str | None:
    for key in keys:
        value = address.get(key)
        if value:
            return str(value)
    return None


def _normalize_reference_name(name: str) -> str:
    return " ".join(str(name).strip().lower().split())


def _dedupe_reference_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    deduped: list[tuple[float, float]] = []
    for latitude, longitude in points:
        if any(
            abs(latitude - existing_latitude) < 0.01 and abs(longitude - existing_longitude) < 0.01
            for existing_latitude, existing_longitude in deduped
        ):
            continue
        deduped.append((latitude, longitude))
    return deduped


def _reference_sample_points(
    *,
    latitude_min: float,
    longitude_min: float,
    latitude_max: float,
    longitude_max: float,
    focus_points: tuple[tuple[float, float], ...] = (),
) -> list[tuple[float, float]]:
    latitude_span = max(latitude_max - latitude_min, 0.001)
    longitude_span = max(longitude_max - longitude_min, 0.001)
    aspect_ratio = max(longitude_span / latitude_span, 1e-6)
    if aspect_ratio >= 1.4:
        rows, cols = 3, 4
    elif aspect_ratio <= 0.72:
        rows, cols = 4, 3
    else:
        rows, cols = 3, 3

    latitude_values = np.linspace(latitude_min + 0.10 * latitude_span, latitude_max - 0.10 * latitude_span, rows)
    longitude_values = np.linspace(longitude_min + 0.10 * longitude_span, longitude_max - 0.10 * longitude_span, cols)
    grid_points = [
        (float(latitude), float(longitude))
        for latitude in latitude_values
        for longitude in longitude_values
    ]
    center_latitude = 0.5 * (latitude_min + latitude_max)
    center_longitude = 0.5 * (longitude_min + longitude_max)
    points = [(center_latitude, center_longitude)]
    points.extend((float(latitude), float(longitude)) for latitude, longitude in focus_points[:10])
    points.extend(grid_points)
    clipped_points = [
        (
            min(max(float(latitude), float(latitude_min)), float(latitude_max)),
            min(max(float(longitude), float(longitude_min)), float(longitude_max)),
        )
        for latitude, longitude in points
    ]
    return _dedupe_reference_points(clipped_points)


@lru_cache(maxsize=256)
def reverse_geocode(latitude: float, longitude: float) -> ReverseGeocodeResult:
    params = {
        "lat": f"{latitude:.6f}",
        "lon": f"{longitude:.6f}",
        "format": "jsonv2",
        "addressdetails": 1,
        "zoom": 18,
    }
    email = os.environ.get("NOMINATIM_EMAIL")
    if email:
        params["email"] = email
    url = f"{NOMINATIM_REVERSE_URL}?{urlencode(params)}"
    request = Request(url, headers=_nominatim_headers())
    _respect_nominatim_rate_limit()
    with urlopen(request, context=_ssl_context()) as response:
        payload = json.loads(response.read().decode("utf-8"))

    address = payload.get("address", {})
    return ReverseGeocodeResult(
        latitude=latitude,
        longitude=longitude,
        display_name=str(payload.get("display_name", "")),
        road=_pick_first(address, "road", "pedestrian", "footway", "path", "cycleway", "residential", "highway"),
        neighborhood=_pick_first(address, "neighbourhood", "suburb", "quarter", "hamlet"),
        city=_pick_first(address, "city", "town", "municipality", "village", "county"),
        postcode=_pick_first(address, "postcode"),
        state=_pick_first(address, "state"),
        country=_pick_first(address, "country"),
    )


def reference_labels_for_bounds(
    *,
    latitude_min: float,
    longitude_min: float,
    latitude_max: float,
    longitude_max: float,
    focus_points: tuple[tuple[float, float], ...] = (),
    max_city_labels: int = 8,
    max_neighborhood_labels: int = 10,
    max_road_labels: int = 8,
) -> list[ReferenceLabel]:
    labels: list[ReferenceLabel] = []
    seen_cities: set[str] = set()
    seen_neighborhoods: set[str] = set()
    seen_roads: set[str] = set()
    sample_points = _reference_sample_points(
        latitude_min=latitude_min,
        longitude_min=longitude_min,
        latitude_max=latitude_max,
        longitude_max=longitude_max,
        focus_points=focus_points,
    )
    for latitude, longitude in sample_points:
        try:
            result = reverse_geocode(round(float(latitude), 5), round(float(longitude), 5))
        except Exception:
            continue

        if result.city and len(seen_cities) < max_city_labels:
            normalized_city = _normalize_reference_name(result.city)
            if normalized_city and normalized_city not in seen_cities:
                labels.append(
                    ReferenceLabel(
                        label=result.city,
                        latitude=float(latitude),
                        longitude=float(longitude),
                        kind="city",
                    )
                )
                seen_cities.add(normalized_city)

        if result.neighborhood and len(seen_neighborhoods) < max_neighborhood_labels:
            normalized_neighborhood = _normalize_reference_name(result.neighborhood)
            if (
                normalized_neighborhood
                and normalized_neighborhood not in seen_neighborhoods
                and normalized_neighborhood not in seen_cities
            ):
                labels.append(
                    ReferenceLabel(
                        label=result.neighborhood,
                        latitude=float(latitude),
                        longitude=float(longitude),
                        kind="neighborhood",
                    )
                )
                seen_neighborhoods.add(normalized_neighborhood)

        if result.road and len(seen_roads) < max_road_labels:
            normalized_road = _normalize_reference_name(result.road)
            if (
                normalized_road
                and normalized_road not in seen_roads
                and normalized_road not in seen_cities
                and normalized_road not in seen_neighborhoods
            ):
                labels.append(
                    ReferenceLabel(
                        label=result.road,
                        latitude=float(latitude),
                        longitude=float(longitude),
                        kind="road",
                    )
                )
                seen_roads.add(normalized_road)
    return labels
