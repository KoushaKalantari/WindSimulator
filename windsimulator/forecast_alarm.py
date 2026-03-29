from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .analysis import copy_config, sample_grid_value
from .config import DEFAULT_CONFIG, EMERGENCY_ALERT_THRESHOLDS, PlumeConfig
from .core import compute_concentration, make_grid
from .data_sources import fetch_elevations, fetch_hourly_weather_forecast, select_weather_window
from .emergency import (
    build_notice_payloads,
    broadcast_recommended_for_band,
    emergency_band_for_concentration,
    incident_to_config,
    recommended_action_for_band,
    severity_multiplier,
)
from .geocoding import ReverseGeocodeResult, reverse_geocode
from .geospatial import make_latlon_grid, normalize_geo_neighborhoods, latlon_to_local_km
from .emergency import HazardIncident


BAND_PRIORITY = {"MINIMAL": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}
DEFAULT_TOP_LOCATION_COUNT = 8
DEFAULT_TOP_LOCATION_SPACING_KM = 2.5

FRAME_REPORT_COLUMNS = [
    "Neighborhood",
    "Latitude",
    "Longitude",
    "X_km",
    "Y_km",
    "Concentration",
    "Band",
    "RecommendedAction",
    "BroadcastRecommended",
    "DraftNotice",
]

TOP_LOCATION_COLUMNS = [
    "LocationLabel",
    "Neighborhood",
    "City",
    "PostalCode",
    "State",
    "Country",
    "DisplayName",
    "Latitude",
    "Longitude",
    "X_km",
    "Y_km",
    "Concentration",
    "PeakConcentration",
    "Band",
    "PeakBand",
    "PeakTime",
    "FirstImpactedTime",
    "RecommendedAction",
    "BroadcastRecommended",
    "DraftNotice",
]


@dataclass
class TerrainSurface:
    x_km: np.ndarray
    y_km: np.ndarray
    latitude_grid: np.ndarray
    longitude_grid: np.ndarray
    elevation_m: np.ndarray
    source_elevation_m: float


@dataclass
class ForecastAlarmFrame:
    timestamp: pd.Timestamp
    cfg: PlumeConfig
    concentration: np.ndarray
    neighborhood_report: pd.DataFrame
    weather: dict[str, Any]


@dataclass
class ForecastAlarmSimulation:
    incident: HazardIncident
    source_latitude: float
    source_longitude: float
    terrain: TerrainSurface
    weather_window: pd.DataFrame
    frames: list[ForecastAlarmFrame]
    neighborhood_summary: pd.DataFrame
    impacted_neighborhoods: pd.DataFrame
    top_locations: pd.DataFrame
    top_neighborhoods: pd.DataFrame
    top_cities: pd.DataFrame
    top_postal_codes: pd.DataFrame
    notice_payloads: list[dict[str, Any]]
    animation: Any | None = None


def terrain_modifier(surface: TerrainSurface) -> np.ndarray:
    dx_km = float(np.mean(np.diff(surface.x_km[0, :]))) if surface.x_km.shape[1] > 1 else 1.0
    dy_km = float(np.mean(np.diff(surface.y_km[:, 0]))) if surface.y_km.shape[0] > 1 else 1.0
    grad_y, grad_x = np.gradient(surface.elevation_m, dy_km, dx_km)
    slope_mag = np.hypot(grad_x, grad_y)
    relative_elevation = surface.source_elevation_m - surface.elevation_m
    modifier = 1.0 + np.clip(relative_elevation / 400.0, -0.25, 0.35)
    modifier *= 1.0 - np.clip(slope_mag / 2000.0, 0.0, 0.2)
    return np.clip(modifier, 0.65, 1.45)


def build_geo_notice_message(
    neighborhood: str,
    band: str,
    action: str,
    incident_name: str,
    incident_type: str,
    source_latitude: float,
    source_longitude: float,
    timestamp: pd.Timestamp | None = None,
) -> str:
    time_text = ""
    if timestamp is not None:
        time_text = f" around {timestamp.strftime('%Y-%m-%d %H:%M UTC')}"
    return (
        f"Emergency notice for {neighborhood}: "
        f"A simulated {incident_type} event ({incident_name}) near "
        f"({source_latitude:.4f}, {source_longitude:.4f}) may impact your area"
        f"{time_text}. Predicted impact level: {band}. "
        f"Recommended action: {action}. Follow official emergency instructions."
    )


def empty_frame_report() -> pd.DataFrame:
    return pd.DataFrame(columns=FRAME_REPORT_COLUMNS)


def empty_top_location_report() -> pd.DataFrame:
    return pd.DataFrame(columns=TOP_LOCATION_COLUMNS)


def build_terrain_surface(
    source_latitude: float,
    source_longitude: float,
    cfg: PlumeConfig,
) -> TerrainSurface:
    x_km, y_km = make_grid(cfg)
    latitude_grid, longitude_grid = make_latlon_grid(
        origin_latitude=source_latitude,
        origin_longitude=source_longitude,
        x_km=x_km,
        y_km=y_km,
    )
    elevation_m = fetch_elevations(latitude_grid, longitude_grid)
    source_elevation_m = float(fetch_elevations(np.array([source_latitude]), np.array([source_longitude]))[0])
    return TerrainSurface(
        x_km=x_km,
        y_km=y_km,
        latitude_grid=latitude_grid,
        longitude_grid=longitude_grid,
        elevation_m=elevation_m,
        source_elevation_m=source_elevation_m,
    )


def build_geo_neighborhood_report(
    incident_name: str,
    incident_type: str,
    source_latitude: float,
    source_longitude: float,
    neighborhoods: dict[str, tuple[float, float]],
    concentration_grid,
    x_grid,
    y_grid,
    timestamp: pd.Timestamp,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, (latitude, longitude) in neighborhoods.items():
        x_km, y_km = latlon_to_local_km(
            latitude,
            longitude,
            origin_latitude=source_latitude,
            origin_longitude=source_longitude,
        )
        concentration = sample_grid_value(
            x_grid=x_grid,
            y_grid=y_grid,
            values=concentration_grid,
            x_km=float(x_km),
            y_km=float(y_km),
        )
        band = emergency_band_for_concentration(concentration)
        action = recommended_action_for_band(band)
        rows.append(
            {
                "Neighborhood": name,
                "Latitude": float(latitude),
                "Longitude": float(longitude),
                "X_km": float(x_km),
                "Y_km": float(y_km),
                "Concentration": concentration,
                "Band": band,
                "RecommendedAction": action,
                "BroadcastRecommended": broadcast_recommended_for_band(band),
                "DraftNotice": build_geo_notice_message(
                    neighborhood=name,
                    band=band,
                    action=action,
                    incident_name=incident_name,
                    incident_type=incident_type,
                    source_latitude=source_latitude,
                    source_longitude=source_longitude,
                    timestamp=timestamp,
                ),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("Concentration", ascending=False)
        .reset_index(drop=True)
    )


def summarize_forecast_impacts(
    incident_name: str,
    incident_type: str,
    source_latitude: float,
    source_longitude: float,
    frame_reports: list[pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    all_reports = pd.concat(frame_reports, ignore_index=True)
    summary_rows: list[dict[str, Any]] = []
    for neighborhood, group in all_reports.groupby("Neighborhood", sort=False):
        peak_idx = group["Concentration"].idxmax()
        peak_row = group.loc[peak_idx]
        impacted_group = group[group["BroadcastRecommended"]].sort_values("Timestamp")
        first_impacted_time = (
            impacted_group.iloc[0]["Timestamp"] if not impacted_group.empty else pd.NaT
        )
        summary_rows.append(
            {
                "Neighborhood": neighborhood,
                "Latitude": peak_row["Latitude"],
                "Longitude": peak_row["Longitude"],
                "PeakConcentration": peak_row["Concentration"],
                "PeakBand": peak_row["Band"],
                "RecommendedAction": peak_row["RecommendedAction"],
                "FirstImpactedTime": first_impacted_time,
                "BroadcastRecommended": peak_row["BroadcastRecommended"],
                "DraftNotice": build_geo_notice_message(
                    neighborhood=neighborhood,
                    band=peak_row["Band"],
                    action=peak_row["RecommendedAction"],
                    incident_name=incident_name,
                    incident_type=incident_type,
                    source_latitude=source_latitude,
                    source_longitude=source_longitude,
                    timestamp=first_impacted_time if pd.notna(first_impacted_time) else None,
                ),
            }
        )

    summary = pd.DataFrame(summary_rows)
    if summary.empty:
        return summary, summary, []

    summary["BandPriority"] = summary["PeakBand"].map(BAND_PRIORITY)
    summary = summary.sort_values(
        ["BandPriority", "PeakConcentration"],
        ascending=[False, False],
    ).reset_index(drop=True)
    impacted = summary[summary["BroadcastRecommended"]].drop(columns=["BandPriority"]).reset_index(drop=True)
    summary = summary.drop(columns=["BandPriority"])
    notice_payloads = build_notice_payloads(
        impacted.rename(
            columns={
                "PeakBand": "Band",
                "PeakConcentration": "Concentration",
            }
        )
    )
    return summary, impacted, notice_payloads


def safe_reverse_geocode(latitude: float, longitude: float) -> ReverseGeocodeResult:
    try:
        return reverse_geocode(round(float(latitude), 5), round(float(longitude), 5))
    except Exception:
        return ReverseGeocodeResult(
            latitude=float(latitude),
            longitude=float(longitude),
            display_name="",
            neighborhood=None,
            city=None,
            postcode=None,
            state=None,
            country=None,
        )


def location_label_for_result(
    result: ReverseGeocodeResult,
    latitude: float,
    longitude: float,
) -> str:
    label_parts: list[str] = []
    for value in (result.neighborhood, result.city, result.postcode):
        if value and value not in label_parts:
            label_parts.append(value)
    if label_parts:
        return ", ".join(label_parts)
    return f"{float(latitude):.4f}, {float(longitude):.4f}"


def select_top_hotspot_indices(
    terrain: TerrainSurface,
    peak_concentration: np.ndarray,
    max_locations: int = DEFAULT_TOP_LOCATION_COUNT,
    min_spacing_km: float = DEFAULT_TOP_LOCATION_SPACING_KM,
) -> list[tuple[int, int]]:
    threshold = EMERGENCY_ALERT_THRESHOLDS["LOW"]
    flat_order = np.argsort(peak_concentration.ravel())[::-1]
    selected: list[tuple[int, int]] = []
    selected_xy: list[tuple[float, float]] = []

    for flat_idx in flat_order:
        row_idx, col_idx = np.unravel_index(int(flat_idx), peak_concentration.shape)
        concentration = float(peak_concentration[row_idx, col_idx])
        if concentration < threshold and selected:
            break
        candidate_xy = (
            float(terrain.x_km[row_idx, col_idx]),
            float(terrain.y_km[row_idx, col_idx]),
        )
        if any(
            np.hypot(candidate_xy[0] - x_km, candidate_xy[1] - y_km) < min_spacing_km
            for x_km, y_km in selected_xy
        ):
            continue
        selected.append((row_idx, col_idx))
        selected_xy.append(candidate_xy)
        if len(selected) >= max_locations:
            break

    if not selected and flat_order.size:
        selected.append(np.unravel_index(int(flat_order[0]), peak_concentration.shape))
    return selected


def build_top_location_report(
    incident_name: str,
    incident_type: str,
    source_latitude: float,
    source_longitude: float,
    terrain: TerrainSurface,
    frames: list[ForecastAlarmFrame],
    max_locations: int = DEFAULT_TOP_LOCATION_COUNT,
    min_spacing_km: float = DEFAULT_TOP_LOCATION_SPACING_KM,
) -> pd.DataFrame:
    if not frames:
        return empty_top_location_report()

    concentration_stack = np.stack([frame.concentration for frame in frames], axis=0)
    peak_frame_indices = np.argmax(concentration_stack, axis=0)
    peak_concentration = np.max(concentration_stack, axis=0)
    hotspot_indices = select_top_hotspot_indices(
        terrain=terrain,
        peak_concentration=peak_concentration,
        max_locations=max_locations,
        min_spacing_km=min_spacing_km,
    )

    rows: list[dict[str, Any]] = []
    for row_idx, col_idx in hotspot_indices:
        latitude = float(terrain.latitude_grid[row_idx, col_idx])
        longitude = float(terrain.longitude_grid[row_idx, col_idx])
        series = concentration_stack[:, row_idx, col_idx]
        peak_value = float(peak_concentration[row_idx, col_idx])
        peak_frame_idx = int(peak_frame_indices[row_idx, col_idx])
        peak_time = frames[peak_frame_idx].timestamp
        impacted_steps = np.flatnonzero(series >= EMERGENCY_ALERT_THRESHOLDS["LOW"])
        first_impacted_time = (
            frames[int(impacted_steps[0])].timestamp
            if impacted_steps.size
            else pd.NaT
        )
        geocode = safe_reverse_geocode(latitude, longitude)
        label = location_label_for_result(geocode, latitude, longitude)
        band = emergency_band_for_concentration(peak_value)
        action = recommended_action_for_band(band)
        rows.append(
            {
                "LocationLabel": label,
                "Neighborhood": geocode.neighborhood,
                "City": geocode.city,
                "PostalCode": geocode.postcode,
                "State": geocode.state,
                "Country": geocode.country,
                "DisplayName": geocode.display_name,
                "Latitude": latitude,
                "Longitude": longitude,
                "X_km": float(terrain.x_km[row_idx, col_idx]),
                "Y_km": float(terrain.y_km[row_idx, col_idx]),
                "Concentration": peak_value,
                "PeakConcentration": peak_value,
                "Band": band,
                "PeakBand": band,
                "PeakTime": peak_time,
                "FirstImpactedTime": first_impacted_time,
                "RecommendedAction": action,
                "BroadcastRecommended": broadcast_recommended_for_band(band),
                "DraftNotice": build_geo_notice_message(
                    neighborhood=label,
                    band=band,
                    action=action,
                    incident_name=incident_name,
                    incident_type=incident_type,
                    source_latitude=source_latitude,
                    source_longitude=source_longitude,
                    timestamp=first_impacted_time if pd.notna(first_impacted_time) else peak_time,
                ),
            }
        )

    if not rows:
        return empty_top_location_report()

    report = pd.DataFrame(rows)
    report["BandPriority"] = report["PeakBand"].map(BAND_PRIORITY)
    report = report.sort_values(
        ["BandPriority", "PeakConcentration"],
        ascending=[False, False],
    )
    report = report.drop_duplicates(subset=["LocationLabel"], keep="first")
    report = report.drop(columns=["BandPriority"]).reset_index(drop=True)
    return report


def summarize_top_locations(
    top_locations: pd.DataFrame,
    key_column: str,
    extra_columns: list[str] | None = None,
) -> pd.DataFrame:
    extra_columns = extra_columns or []
    if top_locations.empty or key_column not in top_locations.columns:
        return pd.DataFrame(
            columns=[
                key_column,
                *extra_columns,
                "PeakBand",
                "PeakConcentration",
                "FirstImpactedTime",
                "HotspotCount",
                "RecommendedAction",
                "BroadcastRecommended",
            ]
        )

    valid = top_locations[
        top_locations[key_column].notna()
        & (top_locations[key_column].astype(str).str.strip() != "")
    ].copy()
    if valid.empty:
        return pd.DataFrame(
            columns=[
                key_column,
                *extra_columns,
                "PeakBand",
                "PeakConcentration",
                "FirstImpactedTime",
                "HotspotCount",
                "RecommendedAction",
                "BroadcastRecommended",
            ]
        )

    rows: list[dict[str, Any]] = []
    for key_value, group in valid.groupby(key_column, sort=False):
        peak_idx = group["PeakConcentration"].idxmax()
        peak_row = group.loc[peak_idx]
        first_impacted_values = group["FirstImpactedTime"].dropna()
        row = {
            key_column: key_value,
            "PeakBand": peak_row["PeakBand"],
            "PeakConcentration": peak_row["PeakConcentration"],
            "FirstImpactedTime": (
                first_impacted_values.min() if not first_impacted_values.empty else pd.NaT
            ),
            "HotspotCount": int(len(group)),
            "RecommendedAction": peak_row["RecommendedAction"],
            "BroadcastRecommended": bool(group["BroadcastRecommended"].any()),
        }
        for extra_column in extra_columns:
            row[extra_column] = peak_row.get(extra_column)
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary["BandPriority"] = summary["PeakBand"].map(BAND_PRIORITY)
    summary = summary.sort_values(
        ["BandPriority", "PeakConcentration"],
        ascending=[False, False],
    ).drop(columns=["BandPriority"]).reset_index(drop=True)
    return summary


def build_geo_notice_payloads(top_locations: pd.DataFrame) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    if top_locations.empty:
        return payloads

    impacted = top_locations[top_locations["BroadcastRecommended"]].drop_duplicates(
        subset=["LocationLabel"],
        keep="first",
    )
    for row in impacted.to_dict(orient="records"):
        payloads.append(
            {
                "location_label": row["LocationLabel"],
                "neighborhood": row["Neighborhood"],
                "city": row["City"],
                "postal_code": row["PostalCode"],
                "band": row["PeakBand"],
                "recommended_action": row["RecommendedAction"],
                "peak_time": row["PeakTime"],
                "first_impacted_time": row["FirstImpactedTime"],
                "message": row["DraftNotice"],
            }
        )
    return payloads


def simulate_forecast_alarm(
    source_latitude: float,
    source_longitude: float,
    incident_time: str | None,
    severity: str | float,
    incident_type: str,
    name: str,
    neighborhoods: dict[str, tuple[float, float]] | None = None,
    base_cfg: PlumeConfig = DEFAULT_CONFIG,
    duration_hours: int = 12,
    forecast_hours: int = 48,
    simulation_resolution: int = 40,
    stability_class: str | None = None,
    release_height_m: float | None = None,
) -> ForecastAlarmSimulation:
    incident = HazardIncident(
        name=name,
        source_x_km=0.0,
        source_y_km=0.0,
        severity=severity,
        incident_type=incident_type,
        stability_class=stability_class,
        release_height_m=release_height_m,
    )
    geo_neighborhoods = (
        normalize_geo_neighborhoods(
            neighborhoods,
            origin_latitude=source_latitude,
            origin_longitude=source_longitude,
        )
        if neighborhoods is not None
        else None
    )
    weather_df = fetch_hourly_weather_forecast(
        latitude=source_latitude,
        longitude=source_longitude,
        forecast_hours=forecast_hours,
    )
    weather_window = select_weather_window(
        weather_df=weather_df,
        incident_time=incident_time,
        duration_hours=duration_hours,
    )

    terrain_cfg = copy_config(
        base_cfg,
        source_x_km=0.0,
        source_y_km=0.0,
        resolution=simulation_resolution,
    )
    terrain = build_terrain_surface(
        source_latitude=source_latitude,
        source_longitude=source_longitude,
        cfg=terrain_cfg,
    )
    terrain_factor = terrain_modifier(terrain)

    frames: list[ForecastAlarmFrame] = []
    frame_reports: list[pd.DataFrame] = []
    for weather_row in weather_window.to_dict(orient="records"):
        frame_incident = HazardIncident(
            name=name,
            source_x_km=0.0,
            source_y_km=0.0,
            severity=severity,
            incident_type=incident_type,
            wind_from_deg=float(weather_row["wind_from_deg"]),
            wind_speed_mps=float(weather_row["wind_speed_mps"]),
            stability_class=stability_class,
            release_height_m=release_height_m,
        )
        cfg = incident_to_config(frame_incident, base_cfg=terrain_cfg)
        _, _, concentration = compute_concentration(cfg)
        concentration = concentration * terrain_factor
        if geo_neighborhoods is not None:
            report = build_geo_neighborhood_report(
                incident_name=name,
                incident_type=incident_type,
                source_latitude=source_latitude,
                source_longitude=source_longitude,
                neighborhoods=geo_neighborhoods,
                concentration_grid=concentration,
                x_grid=terrain.x_km,
                y_grid=terrain.y_km,
                timestamp=pd.Timestamp(weather_row["timestamp"]),
            )
        else:
            report = empty_frame_report()
        report["Timestamp"] = pd.Timestamp(weather_row["timestamp"])
        report["SeverityMultiplier"] = severity_multiplier(severity)
        report["IncidentType"] = incident_type
        frame_reports.append(report)
        frames.append(
            ForecastAlarmFrame(
                timestamp=pd.Timestamp(weather_row["timestamp"]),
                cfg=cfg,
                concentration=concentration,
                neighborhood_report=report,
                weather=weather_row,
            )
        )

    if geo_neighborhoods is not None:
        neighborhood_summary, impacted_neighborhoods, _ = summarize_forecast_impacts(
            incident_name=name,
            incident_type=incident_type,
            source_latitude=source_latitude,
            source_longitude=source_longitude,
            frame_reports=frame_reports,
        )
    else:
        neighborhood_summary = pd.DataFrame()
        impacted_neighborhoods = pd.DataFrame()

    top_locations = build_top_location_report(
        incident_name=name,
        incident_type=incident_type,
        source_latitude=source_latitude,
        source_longitude=source_longitude,
        terrain=terrain,
        frames=frames,
    )
    top_neighborhoods = summarize_top_locations(
        top_locations,
        key_column="Neighborhood",
        extra_columns=["City", "PostalCode"],
    )
    top_cities = summarize_top_locations(
        top_locations,
        key_column="City",
        extra_columns=["State", "Country"],
    )
    top_postal_codes = summarize_top_locations(
        top_locations,
        key_column="PostalCode",
        extra_columns=["City", "State"],
    )
    notice_payloads = build_geo_notice_payloads(top_locations)

    if top_neighborhoods.empty:
        top_neighborhoods = neighborhood_summary.copy()
    if impacted_neighborhoods.empty and not top_neighborhoods.empty:
        impacted_neighborhoods = top_neighborhoods[top_neighborhoods["BroadcastRecommended"]].reset_index(
            drop=True
        )

    return ForecastAlarmSimulation(
        incident=incident,
        source_latitude=source_latitude,
        source_longitude=source_longitude,
        terrain=terrain,
        weather_window=weather_window,
        frames=frames,
        neighborhood_summary=top_neighborhoods,
        impacted_neighborhoods=impacted_neighborhoods,
        top_locations=top_locations,
        top_neighborhoods=top_neighborhoods,
        top_cities=top_cities,
        top_postal_codes=top_postal_codes,
        notice_payloads=notice_payloads,
    )
