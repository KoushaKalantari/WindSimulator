from __future__ import annotations

import json
import ssl
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd


OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ELEVATION_URL = "https://api.open-meteo.com/v1/elevation"

try:
    import certifi
except Exception:
    certifi = None


@dataclass(frozen=True)
class WeatherStep:
    timestamp: datetime
    wind_speed_mps: float
    wind_from_deg: float
    temperature_c: float | None = None


@lru_cache(maxsize=256)
def _fetch_json_by_url(url: str) -> dict:
    ssl_context = ssl.create_default_context(
        cafile=certifi.where() if certifi is not None else None
    )
    with urlopen(url, context=ssl_context) as response:
        return json.loads(response.read().decode("utf-8"))


def _fetch_json(base_url: str, params: dict[str, str | int | float]) -> dict:
    query = urlencode(params)
    return _fetch_json_by_url(f"{base_url}?{query}")


def parse_incident_time(incident_time: str | datetime | None) -> datetime | None:
    if incident_time is None:
        return None
    if isinstance(incident_time, datetime):
        dt = incident_time
    else:
        normalized = incident_time.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)


def default_incident_time_utc() -> datetime:
    now = datetime.now(timezone.utc)
    rounded = now.replace(minute=0, second=0, microsecond=0)
    if rounded < now:
        rounded = rounded + timedelta(hours=1)
    return rounded


def fetch_hourly_weather_forecast(
    latitude: float,
    longitude: float,
    forecast_hours: int = 48,
) -> pd.DataFrame:
    payload = _fetch_json(
        OPEN_METEO_FORECAST_URL,
        {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": "wind_speed_10m,wind_direction_10m,temperature_2m",
            "forecast_hours": forecast_hours,
            "timezone": "UTC",
        },
    )
    hourly = payload["hourly"]
    timestamps = [
        datetime.fromisoformat(f"{timestamp}+00:00")
        for timestamp in hourly["time"]
    ]
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "wind_speed_mps": hourly["wind_speed_10m"],
            "wind_from_deg": hourly["wind_direction_10m"],
            "temperature_c": hourly.get("temperature_2m"),
        }
    )


def select_weather_window(
    weather_df: pd.DataFrame,
    incident_time: str | datetime | None,
    duration_hours: int,
) -> pd.DataFrame:
    start_time = parse_incident_time(incident_time) or default_incident_time_utc()
    if weather_df.empty:
        raise ValueError("Weather forecast response did not contain hourly data.")

    nearest_idx = (weather_df["timestamp"] - start_time).abs().idxmin()
    aligned_start = weather_df.loc[nearest_idx, "timestamp"]
    aligned_end = aligned_start + timedelta(hours=duration_hours - 1)
    window = weather_df[
        (weather_df["timestamp"] >= aligned_start)
        & (weather_df["timestamp"] <= aligned_end)
    ].reset_index(drop=True)
    if window.empty:
        raise ValueError("No forecast data was available for the requested incident time window.")
    return window


def fetch_elevations(
    latitudes,
    longitudes,
    chunk_size: int = 100,
):
    latitudes = np.asarray(latitudes, dtype=float)
    longitudes = np.asarray(longitudes, dtype=float)
    if latitudes.shape != longitudes.shape:
        raise ValueError("latitudes and longitudes must have the same shape")

    flat_latitudes = latitudes.ravel()
    flat_longitudes = longitudes.ravel()
    elevations: list[float] = []
    for start in range(0, len(flat_latitudes), chunk_size):
        end = start + chunk_size
        payload = _fetch_json(
            OPEN_METEO_ELEVATION_URL,
            {
                "latitude": ",".join(f"{value:.6f}" for value in flat_latitudes[start:end]),
                "longitude": ",".join(f"{value:.6f}" for value in flat_longitudes[start:end]),
            },
        )
        elevations.extend(payload["elevation"])
    return np.asarray(elevations, dtype=float).reshape(latitudes.shape)
