from __future__ import annotations

from collections import deque
from hashlib import sha256
import json
from pathlib import Path
import ssl
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Callable
from urllib.parse import urlencode
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np
import pandas as pd


OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ELEVATION_URL = "https://api.open-meteo.com/v1/elevation"
OPEN_METEO_ELEVATION_MAX_COORDINATES_PER_REQUEST = 100
HTTP_RATE_LIMIT_RETRIES = 4
HTTP_RATE_LIMIT_BACKOFF_SECONDS = 1.5
WEATHER_FORECAST_CACHE_TTL_S = 6 * 3600
ELEVATION_CACHE_TTL_S = 30 * 86400
# Open-Meteo's public free tier lists 600/minute, 5000/hour, 10000/day,
# and 300000/month. We also smooth requests with a stricter minimum interval
# to avoid bursty traffic when a notebook asks for high terrain resolution.
OPEN_METEO_REQUEST_LIMITS = (
    (60.0, 600),
    (3600.0, 5000),
    (86400.0, 10000),
    (30.0 * 86400.0, 300000),
)
OPEN_METEO_MIN_REQUEST_INTERVAL_S = 0.25

try:
    import certifi
except Exception:
    certifi = None


_OPEN_METEO_REQUEST_LOCK = Lock()
_OPEN_METEO_LAST_REQUEST_AT = 0.0
_OPEN_METEO_REQUEST_WINDOWS = {
    window_s: deque() for window_s, _ in OPEN_METEO_REQUEST_LIMITS
}
_OPEN_METEO_CACHE_DIR = Path(__file__).resolve().parent.parent / ".windsimulator_cache" / "open_meteo"


@dataclass(frozen=True)
class WeatherStep:
    timestamp: datetime
    wind_speed_mps: float
    wind_from_deg: float
    temperature_c: float | None = None


def _emit_progress(progress_callback: Callable[[str], None] | None, message: str) -> None:
    if progress_callback is not None:
        progress_callback(message)


def _cache_ttl_for_url(url: str) -> float | None:
    if url.startswith(OPEN_METEO_ELEVATION_URL):
        return ELEVATION_CACHE_TTL_S
    if url.startswith(OPEN_METEO_FORECAST_URL):
        return WEATHER_FORECAST_CACHE_TTL_S
    return None


def _cache_path_for_url(url: str) -> Path:
    return _OPEN_METEO_CACHE_DIR / f"{sha256(url.encode('utf-8')).hexdigest()}.json"


def _load_cached_json(url: str) -> dict | None:
    ttl_s = _cache_ttl_for_url(url)
    if ttl_s is None:
        return None

    cache_path = _cache_path_for_url(url)
    if not cache_path.exists():
        return None

    try:
        cache_record = json.loads(cache_path.read_text(encoding="utf-8"))
        fetched_at_s = float(cache_record["fetched_at_s"])
        cached_url = str(cache_record["url"])
        payload = cache_record["payload"]
    except Exception:
        try:
            cache_path.unlink()
        except OSError:
            pass
        return None

    if cached_url != url or (time.time() - fetched_at_s) > ttl_s:
        try:
            cache_path.unlink()
        except OSError:
            pass
        return None
    return payload


def _store_cached_json(url: str, payload: dict) -> None:
    if _cache_ttl_for_url(url) is None:
        return

    cache_path = _cache_path_for_url(url)
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = cache_path.with_suffix(".tmp")
        temp_path.write_text(
            json.dumps(
                {
                    "url": url,
                    "fetched_at_s": time.time(),
                    "payload": payload,
                }
            ),
            encoding="utf-8",
        )
        temp_path.replace(cache_path)
    except OSError:
        # Cache writes are a performance optimization; failures should not block requests.
        pass


def _wait_for_open_meteo_request_slot(
    progress_callback: Callable[[str], None] | None = None,
    request_label: str | None = None,
) -> None:
    global _OPEN_METEO_LAST_REQUEST_AT

    wait_message_sent = False
    while True:
        with _OPEN_METEO_REQUEST_LOCK:
            now = time.monotonic()
            wait_s = max(0.0, _OPEN_METEO_LAST_REQUEST_AT + OPEN_METEO_MIN_REQUEST_INTERVAL_S - now)
            for window_s, limit in OPEN_METEO_REQUEST_LIMITS:
                timestamps = _OPEN_METEO_REQUEST_WINDOWS[window_s]
                cutoff = now - window_s
                while timestamps and timestamps[0] <= cutoff:
                    timestamps.popleft()
                if len(timestamps) >= limit:
                    wait_s = max(wait_s, timestamps[0] + window_s - now)
            if wait_s <= 0.0:
                request_time = time.monotonic()
                _OPEN_METEO_LAST_REQUEST_AT = request_time
                for window_s, _ in OPEN_METEO_REQUEST_LIMITS:
                    _OPEN_METEO_REQUEST_WINDOWS[window_s].append(request_time)
                return
        if wait_s >= 1.0 and not wait_message_sent:
            label = request_label or "Open-Meteo request"
            _emit_progress(
                progress_callback,
                f"{label}: waiting {wait_s:.1f}s to stay within the provider rate limit.",
            )
            wait_message_sent = True
        time.sleep(max(wait_s, 0.05))


def _fetch_json_by_url(
    url: str,
    *,
    progress_callback: Callable[[str], None] | None = None,
    request_label: str | None = None,
) -> dict:
    label = request_label or "Open-Meteo request"
    cached_payload = _load_cached_json(url)
    if cached_payload is not None:
        _emit_progress(progress_callback, f"{label}: using cached response.")
        return cached_payload

    _emit_progress(progress_callback, f"{label}: requesting fresh data from Open-Meteo.")
    ssl_context = ssl.create_default_context(
        cafile=certifi.where() if certifi is not None else None
    )
    for attempt in range(HTTP_RATE_LIMIT_RETRIES):
        try:
            _wait_for_open_meteo_request_slot(
                progress_callback=progress_callback,
                request_label=label,
            )
            with urlopen(url, context=ssl_context) as response:
                payload = json.loads(response.read().decode("utf-8"))
                _store_cached_json(url, payload)
                return payload
        except HTTPError as exc:
            if exc.code != 429 or attempt == HTTP_RATE_LIMIT_RETRIES - 1:
                raise
            retry_after = exc.headers.get("Retry-After") if exc.headers is not None else None
            try:
                delay_s = float(retry_after) if retry_after is not None else 0.0
            except ValueError:
                delay_s = 0.0
            if delay_s <= 0.0:
                delay_s = HTTP_RATE_LIMIT_BACKOFF_SECONDS * (2**attempt)
            _emit_progress(
                progress_callback,
                f"{label}: provider returned HTTP 429, retrying in {delay_s:.1f}s "
                f"(attempt {attempt + 2}/{HTTP_RATE_LIMIT_RETRIES}).",
            )
            time.sleep(delay_s)
    raise RuntimeError("unreachable")


def _fetch_json(
    base_url: str,
    params: dict[str, str | int | float],
    *,
    progress_callback: Callable[[str], None] | None = None,
    request_label: str | None = None,
) -> dict:
    query = urlencode(params)
    return _fetch_json_by_url(
        f"{base_url}?{query}",
        progress_callback=progress_callback,
        request_label=request_label,
    )


def parse_incident_time(incident_time: str | datetime | None) -> datetime | None:
    if incident_time is None:
        return None
    if isinstance(incident_time, datetime):
        dt = incident_time
    else:
        normalized_input = incident_time.strip().lower()
        if normalized_input in {"auto", "next_hour", "next_forecast_hour"}:
            return None
        normalized = incident_time.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).replace(second=0, microsecond=0)


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
    progress_callback: Callable[[str], None] | None = None,
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
        progress_callback=progress_callback,
        request_label="1/5 [weather] Hourly forecast request",
    )
    hourly = payload["hourly"]
    timestamps = [
        datetime.fromisoformat(f"{timestamp}+00:00")
        for timestamp in hourly["time"]
    ]
    if timestamps:
        _emit_progress(
            progress_callback,
            "1/5 [weather] Forecast received: "
            f"{len(timestamps)} hourly rows from "
            f"{timestamps[0].strftime('%Y-%m-%d %H:%M UTC')} to "
            f"{timestamps[-1].strftime('%Y-%m-%d %H:%M UTC')}.",
        )
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
    frame_interval_minutes: int = 60,
) -> pd.DataFrame:
    if duration_hours <= 0:
        raise ValueError("duration_hours must be positive.")
    if frame_interval_minutes <= 0:
        raise ValueError("frame_interval_minutes must be positive.")

    start_time = parse_incident_time(incident_time) or default_incident_time_utc()
    if weather_df.empty:
        raise ValueError("Weather forecast response did not contain hourly data.")

    hourly = weather_df.copy()
    hourly["timestamp"] = pd.to_datetime(hourly["timestamp"], utc=True)
    hourly = hourly.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

    start_timestamp = pd.Timestamp(start_time).tz_convert("UTC")
    frame_frequency = pd.Timedelta(minutes=frame_interval_minutes)
    end_timestamp = start_timestamp + pd.Timedelta(hours=duration_hours)
    frame_timestamps = pd.date_range(
        start=start_timestamp,
        end=end_timestamp,
        freq=frame_frequency,
    )
    if frame_timestamps.empty:
        raise ValueError("No forecast timestamps were generated for the requested time window.")

    available_start = hourly["timestamp"].iloc[0]
    available_end = hourly["timestamp"].iloc[-1]
    if frame_timestamps[0] < available_start or frame_timestamps[-1] > available_end:
        raise ValueError(
            "No forecast data was available for the requested incident time window. "
            f"Requested {frame_timestamps[0].strftime('%Y-%m-%d %H:%M UTC')} to "
            f"{frame_timestamps[-1].strftime('%Y-%m-%d %H:%M UTC')}, but forecast data covers "
            f"{available_start.strftime('%Y-%m-%d %H:%M UTC')} to "
            f"{available_end.strftime('%Y-%m-%d %H:%M UTC')}."
        )

    hourly_seconds = hourly["timestamp"].astype("int64").to_numpy(dtype=np.float64) / 1_000_000_000.0
    frame_seconds = frame_timestamps.astype("int64").to_numpy(dtype=np.float64) / 1_000_000_000.0

    wind_speed_mps = np.interp(
        frame_seconds,
        hourly_seconds,
        hourly["wind_speed_mps"].to_numpy(dtype=np.float64),
    )
    wind_radians = np.deg2rad(hourly["wind_from_deg"].to_numpy(dtype=np.float64) % 360.0)
    wind_from_deg = (
        np.rad2deg(
            np.arctan2(
                np.interp(frame_seconds, hourly_seconds, np.sin(wind_radians)),
                np.interp(frame_seconds, hourly_seconds, np.cos(wind_radians)),
            )
        )
        + 360.0
    ) % 360.0

    temperature_series = hourly["temperature_c"]
    if temperature_series.isnull().all():
        temperature_c = np.full(len(frame_timestamps), np.nan)
    else:
        temperature_source = temperature_series.astype(float).interpolate(limit_direction="both")
        temperature_c = np.interp(
            frame_seconds,
            hourly_seconds,
            temperature_source.to_numpy(dtype=np.float64),
        )

    return pd.DataFrame(
        {
            "timestamp": frame_timestamps,
            "wind_speed_mps": wind_speed_mps,
            "wind_from_deg": wind_from_deg,
            "temperature_c": temperature_c,
        }
    )


def fetch_elevations(
    latitudes,
    longitudes,
    chunk_size: int = OPEN_METEO_ELEVATION_MAX_COORDINATES_PER_REQUEST,
    progress_callback: Callable[[str], None] | None = None,
    progress_label: str = "Elevation request",
):
    latitudes = np.asarray(latitudes, dtype=float)
    longitudes = np.asarray(longitudes, dtype=float)
    if latitudes.shape != longitudes.shape:
        raise ValueError("latitudes and longitudes must have the same shape")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    flat_latitudes = latitudes.ravel()
    flat_longitudes = longitudes.ravel()
    elevations: list[float] = []
    effective_chunk_size = min(int(chunk_size), OPEN_METEO_ELEVATION_MAX_COORDINATES_PER_REQUEST)
    total_points = len(flat_latitudes)
    total_batches = max(1, (total_points + effective_chunk_size - 1) // effective_chunk_size)
    if total_batches > 1:
        _emit_progress(
            progress_callback,
            f"{progress_label}: retrieving {total_points} elevation samples in {total_batches} API batches.",
        )
    for start in range(0, len(flat_latitudes), effective_chunk_size):
        end = start + effective_chunk_size
        batch_index = (start // effective_chunk_size) + 1
        batch_points = len(flat_latitudes[start:end])
        request_label = progress_label
        if total_batches > 1:
            request_label = f"{progress_label} [{batch_index}/{total_batches}, {batch_points} pts]"
        elif total_points > 1:
            request_label = f"{progress_label} [{batch_points} pts]"
        payload = _fetch_json(
            OPEN_METEO_ELEVATION_URL,
            {
                "latitude": ",".join(f"{value:.6f}" for value in flat_latitudes[start:end]),
                "longitude": ",".join(f"{value:.6f}" for value in flat_longitudes[start:end]),
            },
            progress_callback=progress_callback,
            request_label=request_label,
        )
        elevations.extend(payload["elevation"])
    if total_batches > 1:
        _emit_progress(progress_callback, f"{progress_label}: elevation sampling complete.")
    return np.asarray(elevations, dtype=float).reshape(latitudes.shape)
