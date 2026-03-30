from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
from urllib.error import HTTPError

from matplotlib.figure import Figure
from matplotlib.path import Path
import numpy as np
import pandas as pd

from .analysis import copy_config, sample_grid_value
from .config import (
    DEFAULT_CONFIG,
    DEFAULT_FORECAST_FRAME_INTERVAL_MINUTES,
    DEFAULT_INTERNAL_FORECAST_RESOLUTION,
    DEFAULT_TERRAIN_SURFACE_RESOLUTION,
    EMERGENCY_ALERT_THRESHOLDS,
    MAX_TERRAIN_PROVIDER_FETCH_RESOLUTION,
    MIN_TERRAIN_PROVIDER_FETCH_RESOLUTION,
    UNCERTAINTY_SCENARIOS,
    UncertaintyScenarioPreset,
    PlumeConfig,
)
from .core import gaussian_puff_ground_level_centered, make_grid, wind_to_math_angle_rad
from .data_sources import (
    default_incident_time_utc,
    fetch_elevations,
    fetch_hourly_weather_forecast,
    parse_incident_time,
    select_weather_window,
)
from .emergency import (
    CAP_URGENCY_PRIORITY,
    NOTICE_LEVEL_PRIORITY,
    ResolvedMaterialFate,
    ResolvedSourceTerm,
    resolve_incident_source_term,
    resolve_material_fate,
    build_notice_payloads,
    broadcast_decision,
    emergency_band_for_concentration,
    incident_to_config,
    rank_broadcast_areas,
    recommended_action_for_band,
    severity_multiplier,
)
from .geocoding import ReverseGeocodeResult, reverse_geocode
from .geospatial import (
    latlon_to_local_km,
    local_km_to_latlon,
    make_latlon_grid,
    normalize_geo_neighborhoods,
)
from .emergency import HazardIncident


BAND_PRIORITY = {"MINIMAL": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}
DEFAULT_TOP_LOCATION_COUNT = 5
DEFAULT_TOP_LOCATION_SPACING_KM = 2.5
PUFF_TRANSPORT_SUBSTEP_SECONDS = 90.0
TERRAIN_STEERING_SCALE = 6.0
TERRAIN_BLOCKING_SCALE = 3.0
MIN_EFFECTIVE_TRANSPORT_SPEED_MPS = 0.35
SCENARIO_PRIORITY = {"Likely": 0, "Conservative": 1, "Worst Reasonable": 2}
ACTION_POLYGON_MIN_AREA_SQKM = 0.15
MAX_DYNAMIC_FORECAST_GRID_RESOLUTION = 180
AUTO_DURATION_MARKERS = {"auto", "adaptive", "material_auto"}
AUTO_DURATION_MIN_HOURS = 4.0
AUTO_DURATION_MAX_HOURS = 48.0
AUTO_DURATION_BUFFER_HOURS = 1.0
AUTO_DURATION_TARGET_AIRBORNE_FRACTION = 0.03
AUTO_DURATION_MATERIAL_SEARCH_MAX_HOURS = 240.0
AUTO_DURATION_WEATHER_LOOKAHEAD_HOURS = 6.0

FRAME_REPORT_COLUMNS = [
    "Neighborhood",
    "Latitude",
    "Longitude",
    "X_km",
    "Y_km",
    "Concentration",
    "Band",
    "NoticeLevel",
    "CAPSeverity",
    "CAPUrgency",
    "MinutesToImpact",
    "RecommendedAction",
    "BroadcastRecommended",
    "BroadcastBasis",
    "BroadcastPriorityRank",
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
    "NoticeLevel",
    "CAPSeverity",
    "CAPUrgency",
    "MinutesToImpact",
    "RecommendedAction",
    "BroadcastRecommended",
    "BroadcastBasis",
    "BroadcastPriorityRank",
    "DraftNotice",
]

ACTION_POLYGON_COLUMNS = [
    "BroadcastPriorityRank",
    "AreaLabel",
    "ScenarioName",
    "ScenarioDescription",
    "Aggregation",
    "MemberCount",
    "Band",
    "NoticeLevel",
    "CAPSeverity",
    "CAPUrgency",
    "MinutesToImpact",
    "FirstImpactedTime",
    "PeakConcentration",
    "AreaSqKm",
    "CentroidLatitude",
    "CentroidLongitude",
    "RecommendedAction",
    "BroadcastRecommended",
    "BroadcastBasis",
    "GeoJSONCoordinates",
    "GeoJSONFeature",
]

UNCERTAINTY_SUMMARY_COLUMNS = [
    "ScenarioName",
    "ScenarioDescription",
    "Aggregation",
    "MemberCount",
    "DirectionOffsetsDeg",
    "WindSpeedScale",
    "SourceEmissionScale",
    "ReleaseDurationScale",
    "StabilityClass",
    "PeakConcentrationMax",
    "LowAreaSqKm",
    "MediumAreaSqKm",
    "HighAreaSqKm",
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
    source_term: ResolvedSourceTerm
    material_fate: ResolvedMaterialFate
    resolved_duration_hours: float
    duration_basis: str
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
    uncertainty_summary: pd.DataFrame
    action_polygons: pd.DataFrame
    action_geojson: dict[str, Any]
    notice_payloads: list[dict[str, Any]]
    animation: Any | None = None


@dataclass
class ScenarioAggregate:
    scenario_name: str
    scenario_description: str
    aggregation: str
    member_count: int
    direction_offsets_deg: tuple[float, ...]
    wind_speed_scale: float
    source_emission_scale: float
    release_duration_scale: float
    stability_class: str
    peak_concentration: np.ndarray
    first_impact_minutes_by_band: dict[str, np.ndarray]


@dataclass
class TerrainFlowField:
    x_coords_km: np.ndarray
    y_coords_km: np.ndarray
    slope_x: np.ndarray
    slope_y: np.ndarray
    slope_magnitude: np.ndarray
    curvature: np.ndarray


@dataclass
class TransportPuff:
    center_x_km: float
    center_y_km: float
    mass: float
    age_s: float
    path_length_m: float
    heading_rad: float
    speed_mps: float
    meander_index: float
    terrain_confinement: float
    terrain_roughness: float
    turning_accum_rad: float
    terrain_exposure: float
    orographic_lift_m: float


@dataclass
class TerrainVelocityStep:
    vx_mps: float
    vy_mps: float
    slope_x: float
    slope_y: float
    slope_magnitude: float
    channeling_strength: float
    drainage_speed_mps: float


def terrain_modifier(surface: TerrainSurface) -> np.ndarray:
    dx_km = float(np.mean(np.diff(surface.x_km[0, :]))) if surface.x_km.shape[1] > 1 else 1.0
    dy_km = float(np.mean(np.diff(surface.y_km[:, 0]))) if surface.y_km.shape[0] > 1 else 1.0
    grad_y, grad_x = np.gradient(surface.elevation_m, dy_km, dx_km)
    slope_mag = np.hypot(grad_x, grad_y)
    relative_elevation = surface.source_elevation_m - surface.elevation_m
    modifier = 1.0 + np.clip(relative_elevation / 400.0, -0.25, 0.35)
    modifier *= 1.0 - np.clip(slope_mag / 2000.0, 0.0, 0.2)
    return np.clip(modifier, 0.65, 1.45)


def build_terrain_flow_field(surface: TerrainSurface) -> TerrainFlowField:
    dx_km = float(np.mean(np.diff(surface.x_km[0, :]))) if surface.x_km.shape[1] > 1 else 1.0
    dy_km = float(np.mean(np.diff(surface.y_km[:, 0]))) if surface.y_km.shape[0] > 1 else 1.0
    grad_y_m_per_km, grad_x_m_per_km = np.gradient(surface.elevation_m, dy_km, dx_km)
    curvature_x = np.gradient(grad_x_m_per_km, dx_km, axis=1)
    curvature_y = np.gradient(grad_y_m_per_km, dy_km, axis=0)
    slope_x = grad_x_m_per_km / 1000.0
    slope_y = grad_y_m_per_km / 1000.0
    return TerrainFlowField(
        x_coords_km=surface.x_km[0, :].astype(float, copy=False),
        y_coords_km=surface.y_km[:, 0].astype(float, copy=False),
        slope_x=slope_x.astype(float, copy=False),
        slope_y=slope_y.astype(float, copy=False),
        slope_magnitude=np.hypot(slope_x, slope_y).astype(float, copy=False),
        curvature=(curvature_x + curvature_y).astype(float, copy=False),
    )


def _sample_regular_grid(values: np.ndarray, x_coords: np.ndarray, y_coords: np.ndarray, x_km: float, y_km: float) -> float:
    if values.ndim != 2 or len(x_coords) < 1 or len(y_coords) < 1:
        raise ValueError("values must be a 2D regular grid")
    if len(x_coords) == 1 or len(y_coords) == 1:
        return float(values[0, 0])

    x0 = float(x_coords[0])
    y0 = float(y_coords[0])
    dx = float(x_coords[1] - x_coords[0])
    dy = float(y_coords[1] - y_coords[0])
    if abs(dx) < 1e-12 or abs(dy) < 1e-12:
        return float(values[0, 0])

    col = np.clip((x_km - x0) / dx, 0.0, len(x_coords) - 1.0)
    row = np.clip((y_km - y0) / dy, 0.0, len(y_coords) - 1.0)
    col0 = int(np.floor(col))
    row0 = int(np.floor(row))
    col1 = min(col0 + 1, len(x_coords) - 1)
    row1 = min(row0 + 1, len(y_coords) - 1)
    tx = float(col - col0)
    ty = float(row - row0)

    v00 = float(values[row0, col0])
    v01 = float(values[row0, col1])
    v10 = float(values[row1, col0])
    v11 = float(values[row1, col1])
    return (
        (1.0 - tx) * (1.0 - ty) * v00
        + tx * (1.0 - ty) * v01
        + (1.0 - tx) * ty * v10
        + tx * ty * v11
    )


def _wind_velocity_components(wind_from_deg: float, wind_speed_mps: float) -> tuple[float, float]:
    heading_rad = wind_to_math_angle_rad(float(wind_from_deg))
    speed_mps = max(float(wind_speed_mps), MIN_EFFECTIVE_TRANSPORT_SPEED_MPS)
    return float(np.cos(heading_rad) * speed_mps), float(np.sin(heading_rad) * speed_mps)


def _angle_delta_rad(a_rad: float, b_rad: float) -> float:
    return float(np.arctan2(np.sin(a_rad - b_rad), np.cos(a_rad - b_rad)))


def _terrain_aware_velocity(
    *,
    base_speed_mps: float,
    base_heading_rad: float,
    x_km: float,
    y_km: float,
    flow_field: TerrainFlowField,
) -> tuple[float, float]:
    base_speed = max(float(base_speed_mps), MIN_EFFECTIVE_TRANSPORT_SPEED_MPS)
    base_unit = np.array([np.cos(base_heading_rad), np.sin(base_heading_rad)], dtype=float)

    slope_x = _sample_regular_grid(flow_field.slope_x, flow_field.x_coords_km, flow_field.y_coords_km, x_km, y_km)
    slope_y = _sample_regular_grid(flow_field.slope_y, flow_field.x_coords_km, flow_field.y_coords_km, x_km, y_km)
    slope_mag = _sample_regular_grid(
        flow_field.slope_magnitude,
        flow_field.x_coords_km,
        flow_field.y_coords_km,
        x_km,
        y_km,
    )
    curvature = _sample_regular_grid(
        flow_field.curvature,
        flow_field.x_coords_km,
        flow_field.y_coords_km,
        x_km,
        y_km,
    )
    if slope_mag <= 1e-9:
        return tuple(base_unit * base_speed)

    gradient_unit = np.array([slope_x, slope_y], dtype=float) / max(float(slope_mag), 1e-9)
    contour_unit = np.array([-gradient_unit[1], gradient_unit[0]], dtype=float)
    if np.dot(contour_unit, base_unit) < 0.0:
        contour_unit = -contour_unit

    wind_relaxation = float(np.clip(1.5 / max(base_speed, 0.75), 0.15, 1.0))
    valley_channel_factor = float(np.clip(curvature / 140.0, 0.0, 1.0))
    steering_strength = float(np.clip(slope_mag * TERRAIN_STEERING_SCALE, 0.0, 0.42))
    channeling_strength = float(
        np.clip((steering_strength * 0.65 + valley_channel_factor * 0.35) * wind_relaxation, 0.0, 0.58)
    )
    drainage_speed = float(
        np.clip(slope_mag * 900.0 * wind_relaxation * (0.45 + valley_channel_factor), 0.0, 1.4)
    )
    effective_vector = base_unit * base_speed
    effective_vector += contour_unit * (base_speed * channeling_strength)
    effective_vector += -gradient_unit * drainage_speed
    effective_norm = float(np.hypot(effective_vector[0], effective_vector[1]))
    if effective_norm <= 1e-9:
        effective_unit = base_unit
    else:
        effective_unit = effective_vector / effective_norm

    upslope_alignment = max(float(np.dot(effective_unit, gradient_unit)), 0.0)
    downslope_alignment = max(float(-np.dot(effective_unit, gradient_unit)), 0.0)
    blocking_strength = float(np.clip(slope_mag * TERRAIN_BLOCKING_SCALE, 0.0, 0.55))
    speed_factor = 1.0 - 0.50 * blocking_strength * upslope_alignment
    speed_factor *= 1.0 + 0.12 * blocking_strength * downslope_alignment
    speed_factor *= 1.0 + 0.10 * channeling_strength * (1.0 - upslope_alignment)
    speed_factor *= 1.0 + 0.08 * valley_channel_factor * wind_relaxation
    speed = max(base_speed * speed_factor, MIN_EFFECTIVE_TRANSPORT_SPEED_MPS)
    velocity = effective_unit * speed
    return float(velocity[0]), float(velocity[1])


def _advance_puff(
    puff: TransportPuff,
    *,
    dt_s: float,
    start_vx_mps: float,
    start_vy_mps: float,
    end_vx_mps: float,
    end_vy_mps: float,
    flow_field: TerrainFlowField,
) -> None:
    if dt_s <= 0.0:
        return

    substeps = max(1, int(np.ceil(float(dt_s) / PUFF_TRANSPORT_SUBSTEP_SECONDS)))
    substep_s = float(dt_s) / substeps
    for substep_index in range(substeps):
        interpolation = (substep_index + 0.5) / substeps
        ambient_vx_mps = float(start_vx_mps + (end_vx_mps - start_vx_mps) * interpolation)
        ambient_vy_mps = float(start_vy_mps + (end_vy_mps - start_vy_mps) * interpolation)
        ambient_speed_mps = max(float(np.hypot(ambient_vx_mps, ambient_vy_mps)), MIN_EFFECTIVE_TRANSPORT_SPEED_MPS)
        ambient_heading_rad = float(np.arctan2(ambient_vy_mps, ambient_vx_mps))
        previous_heading = float(puff.heading_rad)
        slope_x = _sample_regular_grid(
            flow_field.slope_x,
            flow_field.x_coords_km,
            flow_field.y_coords_km,
            puff.center_x_km,
            puff.center_y_km,
        )
        slope_y = _sample_regular_grid(
            flow_field.slope_y,
            flow_field.x_coords_km,
            flow_field.y_coords_km,
            puff.center_x_km,
            puff.center_y_km,
        )
        local_slope_mag = _sample_regular_grid(
            flow_field.slope_magnitude,
            flow_field.x_coords_km,
            flow_field.y_coords_km,
            puff.center_x_km,
            puff.center_y_km,
        )
        local_curvature = _sample_regular_grid(
            flow_field.curvature,
            flow_field.x_coords_km,
            flow_field.y_coords_km,
            puff.center_x_km,
            puff.center_y_km,
        )
        vx_mps, vy_mps = _terrain_aware_velocity(
            base_speed_mps=ambient_speed_mps,
            base_heading_rad=ambient_heading_rad,
            x_km=puff.center_x_km,
            y_km=puff.center_y_km,
            flow_field=flow_field,
        )
        puff.center_x_km += (vx_mps * substep_s) / 1000.0
        puff.center_y_km += (vy_mps * substep_s) / 1000.0
        puff.speed_mps = float(np.hypot(vx_mps, vy_mps))
        if puff.speed_mps > 1e-9:
            puff.heading_rad = float(np.arctan2(vy_mps, vx_mps))
        turn_delta = abs(_angle_delta_rad(puff.heading_rad, previous_heading))
        puff.turning_accum_rad = float(np.clip(puff.turning_accum_rad + turn_delta, 0.0, 4.0 * np.pi))

        if puff.speed_mps > 1e-9:
            step_unit_x = vx_mps / puff.speed_mps
            step_unit_y = vy_mps / puff.speed_mps
        else:
            step_unit_x = np.cos(previous_heading)
            step_unit_y = np.sin(previous_heading)
        horizontal_step_m = puff.speed_mps * substep_s
        slope_along_path = float(slope_x * step_unit_x + slope_y * step_unit_y)
        terrain_rise_m = max(slope_along_path, 0.0) * horizontal_step_m
        terrain_drop_m = max(-slope_along_path, 0.0) * horizontal_step_m
        valley_channel_factor = float(np.clip(local_curvature / 140.0, 0.0, 1.0))
        puff.orographic_lift_m = float(
            np.clip(
                puff.orographic_lift_m * 0.975
                + terrain_rise_m * (0.28 + 0.10 * valley_channel_factor)
                - terrain_drop_m * 0.10,
                0.0,
                240.0,
            )
        )
        puff.terrain_exposure = float(
            np.clip(
                puff.terrain_exposure * 0.985
                + local_slope_mag * horizontal_step_m * (1.0 + 0.65 * valley_channel_factor),
                0.0,
                25000.0,
            )
        )
        puff.meander_index = float(
            np.clip(puff.meander_index * 0.84 + turn_delta / np.deg2rad(18.0), 0.0, 2.1)
        )
        puff.terrain_confinement = float(
            np.clip(
                puff.terrain_confinement * 0.89
                + local_slope_mag * 16.0
                + valley_channel_factor * 0.08,
                0.0,
                0.95,
            )
        )
        puff.terrain_roughness = float(
            np.clip(
                puff.terrain_roughness * 0.88
                + local_slope_mag * 10.0
                + turn_delta / np.deg2rad(60.0) * 0.05,
                0.0,
                0.95,
            )
        )
        puff.path_length_m += puff.speed_mps * substep_s
        puff.age_s += substep_s


def _transport_domain_padding_km(surface: TerrainSurface) -> tuple[float, float]:
    x_span = float(np.max(surface.x_km) - np.min(surface.x_km))
    y_span = float(np.max(surface.y_km) - np.min(surface.y_km))
    return max(5.0, x_span * 0.6), max(5.0, y_span * 0.6)


def _prune_transport_puffs(puffs: list[TransportPuff], surface: TerrainSurface) -> list[TransportPuff]:
    x_min = float(np.min(surface.x_km))
    x_max = float(np.max(surface.x_km))
    y_min = float(np.min(surface.y_km))
    y_max = float(np.max(surface.y_km))
    x_pad_km, y_pad_km = _transport_domain_padding_km(surface)
    max_age_s = 48.0 * 3600.0
    return [
        puff
        for puff in puffs
        if (x_min - x_pad_km) <= puff.center_x_km <= (x_max + x_pad_km)
        and (y_min - y_pad_km) <= puff.center_y_km <= (y_max + y_pad_km)
        and puff.age_s <= max_age_s
        and puff.mass > 0.0
    ]


def _release_transport_puff(
    *,
    cfg: PlumeConfig,
    base_heading_rad: float,
    release_mass: float,
) -> TransportPuff:
    base_speed = max(float(cfg.wind_speed_mps), MIN_EFFECTIVE_TRANSPORT_SPEED_MPS)
    return TransportPuff(
        center_x_km=float(cfg.source_x_km),
        center_y_km=float(cfg.source_y_km),
        mass=float(release_mass),
        age_s=0.0,
        path_length_m=0.0,
        heading_rad=float(base_heading_rad),
        speed_mps=base_speed,
        meander_index=0.0,
        terrain_confinement=0.0,
        terrain_roughness=0.0,
        turning_accum_rad=0.0,
        terrain_exposure=0.0,
        orographic_lift_m=0.0,
    )


def compute_transport_concentration(
    *,
    surface: TerrainSurface,
    cfg: PlumeConfig,
    puffs: list[TransportPuff],
    material_fate: ResolvedMaterialFate,
) -> np.ndarray:
    concentration = np.zeros_like(surface.x_km, dtype=float)
    if not puffs:
        return concentration

    for puff in puffs:
        airborne_fraction = _material_airborne_fraction(puff.age_s, material_fate)
        effective_mass = float(puff.mass) * airborne_fraction
        if effective_mass <= 1e-12:
            continue
        effective_speed = max(float(puff.speed_mps), MIN_EFFECTIVE_TRANSPORT_SPEED_MPS)
        minimum_spread_distance_m = max(cfg.min_downwind_m, effective_speed * 60.0)
        travel_distance_m = max(
            float(puff.path_length_m),
            minimum_spread_distance_m,
            effective_speed * max(float(puff.age_s), 60.0),
        )
        turning_factor = float(np.clip(puff.turning_accum_rad / (np.pi / 2.0), 0.0, 1.8))
        terrain_exposure_factor = float(
            np.clip(puff.terrain_exposure / max(float(puff.path_length_m), 1.0) * 18.0, 0.0, 1.2)
        )
        orographic_height_boost_m = float(np.clip(puff.orographic_lift_m * 0.22, 0.0, 120.0))
        meander_scale = float(np.clip(1.0 + 0.55 * puff.meander_index, 1.0, 2.4))
        confinement = float(np.clip(puff.terrain_confinement, 0.0, 0.85))
        roughness = float(np.clip(puff.terrain_roughness, 0.0, 0.8))
        concentration += gaussian_puff_ground_level_centered(
            x_km=surface.x_km,
            y_km=surface.y_km,
            center_x_km=puff.center_x_km,
            center_y_km=puff.center_y_km,
            heading_rad=puff.heading_rad,
            q_total=effective_mass,
            travel_distance_m=travel_distance_m,
            release_height_m=float(cfg.release_height_m + orographic_height_boost_m),
            stability_class=cfg.stability_class,
            receptor_elevation_m=surface.elevation_m,
            source_elevation_m=surface.source_elevation_m,
            min_spread_distance_m=minimum_spread_distance_m,
            sigma_x_scale=float(
                np.clip(
                    1.0
                    + 0.38 * confinement
                    + 0.22 * puff.meander_index
                    + 0.18 * turning_factor
                    + 0.12 * terrain_exposure_factor,
                    0.9,
                    2.5,
                )
            ),
            sigma_y_scale=float(
                np.clip(
                    meander_scale
                    * (1.0 - 0.30 * confinement)
                    * (1.0 + 0.18 * turning_factor + 0.16 * terrain_exposure_factor),
                    0.7,
                    3.0,
                )
            ),
            sigma_z_scale=float(
                np.clip(
                    1.0
                    + 0.26 * roughness
                    + 0.10 * puff.meander_index
                    + 0.22 * turning_factor
                    + 0.18 * terrain_exposure_factor
                    + 0.22 * (orographic_height_boost_m / max(float(cfg.release_height_m), 10.0)),
                    0.9,
                    2.2,
                )
            ),
        )
    return concentration


def _half_life_factor(age_minutes: float, half_life_minutes: float | None) -> float:
    if half_life_minutes is None:
        return 1.0
    effective_half_life = max(float(half_life_minutes), 1e-6)
    return float(np.exp(-np.log(2.0) * max(float(age_minutes), 0.0) / effective_half_life))


def _material_airborne_fraction(
    age_s: float,
    material_fate: ResolvedMaterialFate,
) -> float:
    age_minutes = max(float(age_s) / 60.0, 0.0)
    reactive_factor = _half_life_factor(
        age_minutes,
        material_fate.reactive_airborne_half_life_minutes,
    )
    deposition_factor = _half_life_factor(
        age_minutes,
        material_fate.deposition_half_life_minutes,
    )
    residual_fraction = float(np.clip(material_fate.residual_airborne_fraction, 0.0, 1.0))
    airborne_fraction = deposition_factor * (
        residual_fraction + (1.0 - residual_fraction) * reactive_factor
    )
    return float(np.clip(airborne_fraction, 0.0, 1.0))


def _duration_hours_is_auto(duration_hours: int | float | str | None) -> bool:
    if duration_hours is None:
        return True
    if isinstance(duration_hours, str):
        return duration_hours.strip().lower() in AUTO_DURATION_MARKERS
    return False


def _manual_duration_hours(duration_hours: int | float | str | None) -> float:
    if _duration_hours_is_auto(duration_hours):
        raise ValueError("duration_hours must be numeric when not using auto duration.")
    resolved = float(duration_hours)
    if resolved <= 0.0:
        raise ValueError("duration_hours must be positive.")
    return resolved


def _incident_start_timestamp_utc(incident_time: str | None) -> pd.Timestamp:
    start_dt = parse_incident_time(incident_time) or default_incident_time_utc()
    return pd.Timestamp(start_dt).tz_convert("UTC")


def _material_airborne_cutoff_hours(
    material_fate: ResolvedMaterialFate,
    *,
    target_fraction: float = AUTO_DURATION_TARGET_AIRBORNE_FRACTION,
    max_hours: float = AUTO_DURATION_MATERIAL_SEARCH_MAX_HOURS,
) -> tuple[float, bool]:
    step_minutes = 15.0
    age_minutes = 0.0
    while age_minutes <= max_hours * 60.0:
        airborne_fraction = _material_airborne_fraction(age_minutes * 60.0, material_fate)
        if airborne_fraction <= target_fraction:
            return age_minutes / 60.0, False
        age_minutes += step_minutes
    return max_hours, True


def _weather_duration_factor(
    *,
    average_wind_speed_mps: float,
    material_fate: ResolvedMaterialFate,
    stability_class: str | None,
) -> float:
    wind_speed = max(float(average_wind_speed_mps), MIN_EFFECTIVE_TRANSPORT_SPEED_MPS)
    base_hours = 4.0 + 18.0 / max(wind_speed, 0.75)
    normalized_stability = str(stability_class or DEFAULT_CONFIG.stability_class).strip().upper()
    stability_factor = {
        "A": 0.78,
        "B": 0.84,
        "C": 0.92,
        "D": 1.0,
        "E": 1.16,
        "F": 1.28,
    }.get(normalized_stability, 1.0)

    persistence_factor = 1.0
    if material_fate.residual_airborne_fraction >= 0.8:
        persistence_factor *= 2.2
    elif material_fate.residual_airborne_fraction >= 0.5:
        persistence_factor *= 1.5
    if material_fate.reactive_airborne_half_life_minutes is not None:
        if material_fate.reactive_airborne_half_life_minutes <= 60.0:
            persistence_factor *= 0.55
        elif material_fate.reactive_airborne_half_life_minutes <= 180.0:
            persistence_factor *= 0.75
    if material_fate.deposition_half_life_minutes is not None:
        if material_fate.deposition_half_life_minutes >= 6000.0:
            persistence_factor *= 1.35
        elif material_fate.deposition_half_life_minutes >= 3000.0:
            persistence_factor *= 1.15

    return float(base_hours * stability_factor * persistence_factor)


def _resolve_auto_duration_hours(
    *,
    weather_df: pd.DataFrame,
    incident_time: str | None,
    source_term: ResolvedSourceTerm,
    material_fate: ResolvedMaterialFate,
    stability_class: str | None,
    frame_interval_minutes: int,
) -> tuple[float, str]:
    hourly = weather_df.copy()
    hourly["timestamp"] = pd.to_datetime(hourly["timestamp"], utc=True)
    hourly = hourly.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    if hourly.empty:
        raise ValueError("Weather forecast response did not contain hourly data.")

    requested_start = _incident_start_timestamp_utc(incident_time)
    ventilation_window = hourly[hourly["timestamp"] >= requested_start].copy()
    if ventilation_window.empty:
        ventilation_window = hourly.copy()
    first_weather_timestamp = pd.Timestamp(ventilation_window.iloc[0]["timestamp"])
    ventilation_end = first_weather_timestamp + pd.Timedelta(hours=AUTO_DURATION_WEATHER_LOOKAHEAD_HOURS)
    ventilation_window = ventilation_window[ventilation_window["timestamp"] <= ventilation_end].reset_index(drop=True)
    average_wind_speed_mps = float(ventilation_window["wind_speed_mps"].mean())

    material_cutoff_hours, material_cutoff_capped = _material_airborne_cutoff_hours(material_fate)
    weather_duration_hours = _weather_duration_factor(
        average_wind_speed_mps=average_wind_speed_mps,
        material_fate=material_fate,
        stability_class=stability_class,
    )
    release_duration_hours = max(float(source_term.release_duration_minutes) / 60.0, 0.0)
    clearance_hours = min(material_cutoff_hours, weather_duration_hours)
    resolved_duration_hours = max(
        AUTO_DURATION_MIN_HOURS,
        release_duration_hours + clearance_hours + AUTO_DURATION_BUFFER_HOURS,
    )
    resolved_duration_hours = min(resolved_duration_hours, AUTO_DURATION_MAX_HOURS)

    available_end = pd.Timestamp(hourly["timestamp"].iloc[-1])
    available_duration_hours = max(
        (available_end - requested_start).total_seconds() / 3600.0,
        0.0,
    )
    availability_capped = False
    if available_duration_hours > 0.0 and resolved_duration_hours > available_duration_hours:
        resolved_duration_hours = available_duration_hours
        availability_capped = True

    frame_hours = max(float(frame_interval_minutes) / 60.0, 1.0 / 60.0)
    resolved_duration_hours = max(
        frame_hours,
        np.ceil(resolved_duration_hours / frame_hours) * frame_hours,
    )

    basis_parts = [
        f"material={material_fate.profile_label}",
        f"avg wind={average_wind_speed_mps:.1f} m/s",
        f"stability={str(stability_class or DEFAULT_CONFIG.stability_class).strip().upper()}",
        f"weather clearance={weather_duration_hours:.1f} h",
        f"airborne cutoff={material_cutoff_hours:.1f} h",
    ]
    if material_cutoff_capped:
        basis_parts.append("airborne cutoff exceeded search horizon")
    if availability_capped:
        basis_parts.append("limited by available weather horizon")
    if np.isclose(resolved_duration_hours, AUTO_DURATION_MAX_HOURS):
        basis_parts.append("limited by operational max horizon")
    return float(resolved_duration_hours), "; ".join(basis_parts)


def build_geo_notice_message(
    neighborhood: str,
    band: str,
    action: str,
    incident_name: str,
    incident_type: str,
    source_latitude: float,
    source_longitude: float,
    notice_level: str | None = None,
    cap_urgency: str | None = None,
    timestamp: pd.Timestamp | None = None,
) -> str:
    time_text = ""
    if timestamp is not None:
        time_text = f" around {timestamp.strftime('%Y-%m-%d %H:%M UTC')}"
    notice_text = f" Recommended notice level: {notice_level}." if notice_level else ""
    urgency_text = ""
    if cap_urgency and cap_urgency != "Unknown":
        urgency_text = f" Urgency: {cap_urgency}."
    return (
        f"Emergency notice for {neighborhood}: "
        f"A simulated {incident_type} event ({incident_name}) near "
        f"({source_latitude:.4f}, {source_longitude:.4f}) may impact your area"
        f"{time_text}. Predicted impact level: {band}.{notice_text}{urgency_text} "
        f"Recommended action: {action}. Follow official emergency instructions."
    )


def empty_frame_report() -> pd.DataFrame:
    return pd.DataFrame(columns=FRAME_REPORT_COLUMNS)


def empty_top_location_report() -> pd.DataFrame:
    return pd.DataFrame(columns=TOP_LOCATION_COLUMNS)


def empty_action_polygon_report() -> pd.DataFrame:
    return pd.DataFrame(columns=ACTION_POLYGON_COLUMNS)


def empty_uncertainty_summary() -> pd.DataFrame:
    return pd.DataFrame(columns=UNCERTAINTY_SUMMARY_COLUMNS)


def _band_threshold_items() -> list[tuple[str, float]]:
    return [
        ("LOW", EMERGENCY_ALERT_THRESHOLDS["LOW"]),
        ("MEDIUM", EMERGENCY_ALERT_THRESHOLDS["MEDIUM"]),
        ("HIGH", EMERGENCY_ALERT_THRESHOLDS["HIGH"]),
    ]


def _shift_stability_class(stability_class: str | None, shift: int) -> str:
    classes = ["A", "B", "C", "D", "E", "F"]
    normalized = str(stability_class or DEFAULT_CONFIG.stability_class).strip().upper()
    if normalized not in classes:
        normalized = DEFAULT_CONFIG.stability_class
    base_index = classes.index(normalized)
    shifted_index = int(np.clip(base_index + int(shift), 0, len(classes) - 1))
    return classes[shifted_index]


def _scaled_source_term(
    source_term: ResolvedSourceTerm,
    *,
    emission_scale: float = 1.0,
    duration_scale: float = 1.0,
) -> ResolvedSourceTerm:
    emission_rate = max(float(source_term.emission_rate) * float(emission_scale), 0.0)
    release_duration_minutes = max(float(source_term.release_duration_minutes) * float(duration_scale), 0.0)
    initial_pulse_minutes = max(float(source_term.initial_pulse_minutes) * float(duration_scale), 0.0)
    return ResolvedSourceTerm(
        profile_key=source_term.profile_key,
        profile_label=source_term.profile_label,
        emission_rate=emission_rate,
        release_duration_minutes=release_duration_minutes,
        initial_pulse_minutes=initial_pulse_minutes,
        initial_pulse_mass=emission_rate * initial_pulse_minutes * 60.0,
        release_height_m=source_term.release_height_m,
        description=source_term.description,
    )


def _released_mass_over_interval(
    source_term: ResolvedSourceTerm,
    start_elapsed_s: float,
    end_elapsed_s: float,
) -> float:
    interval_start_s = max(float(start_elapsed_s), 0.0)
    interval_end_s = max(float(end_elapsed_s), interval_start_s)
    if interval_end_s <= interval_start_s:
        return 0.0

    released_mass = 0.0
    if interval_start_s <= 0.0 < interval_end_s + 1e-9:
        released_mass += float(source_term.initial_pulse_mass)

    release_end_s = max(float(source_term.release_duration_minutes) * 60.0, 0.0)
    steady_start_s = min(interval_start_s, release_end_s)
    steady_end_s = min(interval_end_s, release_end_s)
    released_mass += float(source_term.emission_rate) * max(steady_end_s - steady_start_s, 0.0)
    return float(released_mass)


def _grid_cell_area_sqkm(surface: TerrainSurface) -> float:
    dx_km = float(np.mean(np.diff(surface.x_km[0, :]))) if surface.x_km.shape[1] > 1 else 1.0
    dy_km = float(np.mean(np.diff(surface.y_km[:, 0]))) if surface.y_km.shape[0] > 1 else 1.0
    return abs(dx_km * dy_km)


def _empty_first_impact_minutes(shape: tuple[int, int]) -> dict[str, np.ndarray]:
    return {
        band: np.full(shape, np.nan, dtype=float)
        for band, _ in _band_threshold_items()
    }


def _update_first_impact_minutes(
    first_impact_minutes_by_band: dict[str, np.ndarray],
    concentration: np.ndarray,
    elapsed_minutes: float,
) -> None:
    for band, threshold in _band_threshold_items():
        first_minutes = first_impact_minutes_by_band[band]
        impacted_mask = concentration >= threshold
        new_impacts = impacted_mask & np.isnan(first_minutes)
        first_minutes[new_impacts] = float(elapsed_minutes)


def _nanmin_stack(values: list[np.ndarray]) -> np.ndarray:
    if not values:
        raise ValueError("values must contain at least one array")
    stack = np.stack(values, axis=0).astype(float, copy=False)
    stack = np.where(np.isnan(stack), np.inf, stack)
    reduced = np.min(stack, axis=0)
    reduced[np.isinf(reduced)] = np.nan
    return reduced


def _closed_ring(points: np.ndarray) -> np.ndarray:
    ring = np.asarray(points, dtype=float)
    if ring.ndim != 2 or ring.shape[0] < 3:
        return ring
    if not np.allclose(ring[0], ring[-1]):
        ring = np.vstack([ring, ring[0]])
    return ring


def _polygon_area_sqkm(points: np.ndarray) -> float:
    ring = _closed_ring(points)
    if ring.shape[0] < 4:
        return 0.0
    x_coords = ring[:, 0]
    y_coords = ring[:, 1]
    cross = x_coords[:-1] * y_coords[1:] - x_coords[1:] * y_coords[:-1]
    return abs(float(np.sum(cross)) * 0.5)


def _polygon_centroid(points: np.ndarray) -> tuple[float, float]:
    ring = _closed_ring(points)
    if ring.shape[0] < 4:
        return float(np.mean(ring[:, 0])), float(np.mean(ring[:, 1]))
    x_coords = ring[:, 0]
    y_coords = ring[:, 1]
    cross = x_coords[:-1] * y_coords[1:] - x_coords[1:] * y_coords[:-1]
    twice_area = float(np.sum(cross))
    if abs(twice_area) < 1e-9:
        return float(np.mean(ring[:-1, 0])), float(np.mean(ring[:-1, 1]))
    centroid_x = float(np.sum((x_coords[:-1] + x_coords[1:]) * cross) / (3.0 * twice_area))
    centroid_y = float(np.sum((y_coords[:-1] + y_coords[1:]) * cross) / (3.0 * twice_area))
    return centroid_x, centroid_y


def _extract_threshold_polygons(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    values: np.ndarray,
    threshold: float,
) -> list[np.ndarray]:
    if values.size == 0 or float(np.nanmax(values)) < float(threshold):
        return []

    figure = Figure(figsize=(1, 1))
    ax = figure.subplots()
    contour_set = ax.contour(x_grid, y_grid, values, levels=[float(threshold)])
    segments = contour_set.allsegs[0] if contour_set.allsegs else []
    figure.clear()

    polygons: list[np.ndarray] = []
    for segment in segments:
        ring = _closed_ring(segment)
        if ring.shape[0] < 4:
            continue
        if _polygon_area_sqkm(ring) < ACTION_POLYGON_MIN_AREA_SQKM:
            continue
        polygons.append(ring.astype(float, copy=False))
    return polygons

def _resample_grid(values: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    if values.shape == target_shape:
        return values.astype(float, copy=False)

    source_rows = np.linspace(0.0, 1.0, values.shape[0])
    source_cols = np.linspace(0.0, 1.0, values.shape[1])
    target_rows = np.linspace(0.0, 1.0, target_shape[0])
    target_cols = np.linspace(0.0, 1.0, target_shape[1])

    row_resampled = np.vstack(
        [np.interp(target_cols, source_cols, row) for row in values]
    )
    col_resampled = np.vstack(
        [
            np.interp(target_rows, source_rows, row_resampled[:, col_idx])
            for col_idx in range(row_resampled.shape[1])
        ]
    ).T
    return col_resampled.astype(float, copy=False)


def _terrain_provider_resolution_candidates(target_resolution: int) -> list[int]:
    target_resolution = int(target_resolution)
    if target_resolution <= MIN_TERRAIN_PROVIDER_FETCH_RESOLUTION:
        return [target_resolution]

    current = min(target_resolution, MAX_TERRAIN_PROVIDER_FETCH_RESOLUTION)
    candidates: list[int] = []
    while current >= MIN_TERRAIN_PROVIDER_FETCH_RESOLUTION:
        if current not in candidates:
            candidates.append(current)
        if current == MIN_TERRAIN_PROVIDER_FETCH_RESOLUTION:
            break
        next_resolution = max(
            MIN_TERRAIN_PROVIDER_FETCH_RESOLUTION,
            int(np.ceil(current * 0.75)),
        )
        if next_resolution >= current:
            break
        current = next_resolution
    return candidates


def build_terrain_surface(
    source_latitude: float,
    source_longitude: float,
    cfg: PlumeConfig,
    progress_callback: Callable[[str], None] | None = None,
) -> TerrainSurface:
    _emit_progress(
        progress_callback,
        "2/6 [terrain] Mapping the simulation grid into latitude/longitude coordinates.",
    )
    x_km, y_km = make_grid(cfg)
    latitude_grid, longitude_grid = make_latlon_grid(
        origin_latitude=source_latitude,
        origin_longitude=source_longitude,
        x_km=x_km,
        y_km=y_km,
    )
    target_resolution = int(cfg.resolution)
    provider_resolution_candidates = _terrain_provider_resolution_candidates(target_resolution)
    elevation_m: np.ndarray | None = None
    for candidate_index, provider_resolution in enumerate(provider_resolution_candidates, start=1):
        if provider_resolution < target_resolution:
            _emit_progress(
                progress_callback,
                "2/6 [terrain] Provider-safe fetch mode: sampling elevation at "
                f"{provider_resolution}x{provider_resolution} and interpolating to "
                f"{target_resolution}x{target_resolution}.",
            )
            sample_cfg = copy_config(cfg, resolution=provider_resolution)
            sample_x_km, sample_y_km = make_grid(sample_cfg)
            sample_latitude_grid, sample_longitude_grid = make_latlon_grid(
                origin_latitude=source_latitude,
                origin_longitude=source_longitude,
                x_km=sample_x_km,
                y_km=sample_y_km,
            )
        else:
            sample_latitude_grid = latitude_grid
            sample_longitude_grid = longitude_grid
        try:
            sampled_elevation_m = fetch_elevations(
                sample_latitude_grid,
                sample_longitude_grid,
                progress_callback=progress_callback,
                progress_label="2/6 [terrain] Surface elevation batches",
            )
            if sampled_elevation_m.shape != latitude_grid.shape:
                elevation_m = _resample_grid(sampled_elevation_m, latitude_grid.shape)
                _emit_progress(
                    progress_callback,
                    "2/6 [terrain] Upscaled provider-safe elevation samples onto the full simulation grid.",
                )
            else:
                elevation_m = sampled_elevation_m
            break
        except HTTPError as exc:
            is_last_candidate = candidate_index == len(provider_resolution_candidates)
            if exc.code != 429 or is_last_candidate:
                raise
            next_resolution = provider_resolution_candidates[candidate_index]
            _emit_progress(
                progress_callback,
                "2/6 [terrain] Elevation provider throttled this fetch; retrying with a lighter "
                f"{next_resolution}x{next_resolution} terrain sample grid.",
            )
    if elevation_m is None:
        raise RuntimeError("Terrain elevation sampling did not produce a grid.")
    _emit_progress(
        progress_callback,
        "2/6 [terrain] Sampling the source-point elevation for terrain normalization.",
    )
    source_elevation_m = float(
        fetch_elevations(
            np.array([source_latitude]),
            np.array([source_longitude]),
            progress_callback=progress_callback,
            progress_label="2/6 [terrain] Source elevation",
        )[0]
    )
    _emit_progress(progress_callback, "2/6 [terrain] Terrain surface is ready.")
    return TerrainSurface(
        x_km=x_km,
        y_km=y_km,
        latitude_grid=latitude_grid,
        longitude_grid=longitude_grid,
        elevation_m=elevation_m,
        source_elevation_m=source_elevation_m,
    )


def _simulate_transport_member(
    *,
    incident: HazardIncident,
    source_latitude: float,
    source_longitude: float,
    terrain: TerrainSurface,
    terrain_cfg: PlumeConfig,
    terrain_flow_field: TerrainFlowField,
    weather_window: pd.DataFrame,
    source_term: ResolvedSourceTerm,
    material_fate: ResolvedMaterialFate,
    geo_neighborhoods: dict[str, tuple[float, float]] | None = None,
    wind_direction_offset_deg: float = 0.0,
    wind_speed_scale: float = 1.0,
    stability_class: str | None = None,
    frame_interval_minutes: int = DEFAULT_FORECAST_FRAME_INTERVAL_MINUTES,
    progress_callback: Callable[[str], None] | None = None,
    progress_prefix: str | None = None,
    progress_enabled: bool = False,
    store_frames: bool = True,
    store_reports: bool = True,
) -> tuple[list[ForecastAlarmFrame], list[pd.DataFrame], np.ndarray, dict[str, np.ndarray]]:
    if weather_window.empty:
        return [], [], np.zeros_like(terrain.x_km, dtype=float), _empty_first_impact_minutes(terrain.x_km.shape)

    start_timestamp = pd.Timestamp(weather_window.iloc[0]["timestamp"])
    peak_concentration = np.zeros_like(terrain.x_km, dtype=float)
    first_impact_minutes_by_band = _empty_first_impact_minutes(terrain.x_km.shape)
    frames: list[ForecastAlarmFrame] = []
    frame_reports: list[pd.DataFrame] = []
    active_puffs: list[TransportPuff] = []
    previous_timestamp: pd.Timestamp | None = None
    previous_wind_velocity: tuple[float, float] | None = None

    total_frames = len(weather_window)
    for frame_index, weather_row in enumerate(weather_window.to_dict(orient="records"), start=1):
        frame_timestamp = pd.Timestamp(weather_row["timestamp"])
        adjusted_wind_from_deg = (float(weather_row["wind_from_deg"]) + float(wind_direction_offset_deg)) % 360.0
        adjusted_wind_speed_mps = max(
            float(weather_row["wind_speed_mps"]) * float(wind_speed_scale),
            MIN_EFFECTIVE_TRANSPORT_SPEED_MPS,
        )
        if progress_enabled and progress_prefix:
            _emit_progress(
                progress_callback,
                f"{progress_prefix} {frame_index}/{total_frames}] "
                f"{frame_timestamp.strftime('%Y-%m-%d %H:%M UTC')} | "
                f"wind {adjusted_wind_speed_mps:.1f} m/s from {adjusted_wind_from_deg:.0f} degrees.",
            )

        cfg = copy_config(
            terrain_cfg,
            emission_rate=float(source_term.emission_rate),
            wind_from_deg=adjusted_wind_from_deg,
            wind_speed_mps=adjusted_wind_speed_mps,
            stability_class=stability_class or terrain_cfg.stability_class,
            release_height_m=float(source_term.release_height_m),
        )
        base_heading_rad = wind_to_math_angle_rad(adjusted_wind_from_deg)
        current_wind_velocity = _wind_velocity_components(
            adjusted_wind_from_deg,
            adjusted_wind_speed_mps,
        )
        if previous_timestamp is None:
            dt_s = 0.0
            release_interval_end_s = max(float(frame_interval_minutes) * 60.0 * 0.5, 60.0)
            release_mass = _released_mass_over_interval(source_term, 0.0, release_interval_end_s)
            elapsed_end_s = 0.0
        else:
            dt_s = max((frame_timestamp - previous_timestamp).total_seconds(), 0.0)
            elapsed_start_s = max((previous_timestamp - start_timestamp).total_seconds(), 0.0)
            elapsed_end_s = max((frame_timestamp - start_timestamp).total_seconds(), 0.0)
            release_mass = _released_mass_over_interval(source_term, elapsed_start_s, elapsed_end_s)

        if dt_s > 0.0:
            start_vx_mps, start_vy_mps = previous_wind_velocity or current_wind_velocity
            end_vx_mps, end_vy_mps = current_wind_velocity
            mid_vx_mps = float(start_vx_mps + 0.5 * (end_vx_mps - start_vx_mps))
            mid_vy_mps = float(start_vy_mps + 0.5 * (end_vy_mps - start_vy_mps))
            for puff in active_puffs:
                _advance_puff(
                    puff,
                    dt_s=dt_s,
                    start_vx_mps=start_vx_mps,
                    start_vy_mps=start_vy_mps,
                    end_vx_mps=end_vx_mps,
                    end_vy_mps=end_vy_mps,
                    flow_field=terrain_flow_field,
                )
        else:
            end_vx_mps, end_vy_mps = current_wind_velocity
            mid_vx_mps, mid_vy_mps = current_wind_velocity

        if release_mass > 0.0:
            new_puff = _release_transport_puff(
                cfg=cfg,
                base_heading_rad=base_heading_rad,
                release_mass=release_mass,
            )
            if dt_s > 0.0:
                _advance_puff(
                    new_puff,
                    dt_s=dt_s * 0.5,
                    start_vx_mps=mid_vx_mps,
                    start_vy_mps=mid_vy_mps,
                    end_vx_mps=end_vx_mps,
                    end_vy_mps=end_vy_mps,
                    flow_field=terrain_flow_field,
                )
            active_puffs.append(new_puff)

        active_puffs = _prune_transport_puffs(active_puffs, terrain)
        concentration = compute_transport_concentration(
            surface=terrain,
            cfg=cfg,
            puffs=active_puffs,
            material_fate=material_fate,
        )
        peak_concentration = np.maximum(peak_concentration, concentration)
        _update_first_impact_minutes(
            first_impact_minutes_by_band,
            concentration,
            elapsed_minutes=elapsed_end_s / 60.0,
        )

        report = empty_frame_report()
        if store_reports and geo_neighborhoods is not None:
            report = build_geo_neighborhood_report(
                incident_name=incident.name,
                incident_type=incident.incident_type,
                source_latitude=source_latitude,
                source_longitude=source_longitude,
                neighborhoods=geo_neighborhoods,
                concentration_grid=concentration,
                x_grid=terrain.x_km,
                y_grid=terrain.y_km,
                timestamp=frame_timestamp,
            )
            report["Timestamp"] = frame_timestamp
            report["SeverityMultiplier"] = severity_multiplier(incident.severity)
            report["IncidentType"] = incident.incident_type
            frame_reports.append(report)

        if store_frames:
            frame_weather = dict(weather_row)
            frame_weather["wind_from_deg"] = adjusted_wind_from_deg
            frame_weather["wind_speed_mps"] = adjusted_wind_speed_mps
            frames.append(
                ForecastAlarmFrame(
                    timestamp=frame_timestamp,
                    cfg=cfg,
                    concentration=concentration,
                    neighborhood_report=report,
                    weather=frame_weather,
                )
            )

        previous_timestamp = frame_timestamp
        previous_wind_velocity = current_wind_velocity

    return frames, frame_reports, peak_concentration, first_impact_minutes_by_band


def _aggregate_uncertainty_scenario(
    *,
    preset: UncertaintyScenarioPreset,
    incident: HazardIncident,
    source_latitude: float,
    source_longitude: float,
    terrain: TerrainSurface,
    terrain_cfg: PlumeConfig,
    terrain_flow_field: TerrainFlowField,
    weather_window: pd.DataFrame,
    base_source_term: ResolvedSourceTerm,
    material_fate: ResolvedMaterialFate,
    frame_interval_minutes: int,
    progress_callback: Callable[[str], None] | None = None,
) -> ScenarioAggregate:
    scenario_source_term = _scaled_source_term(
        base_source_term,
        emission_scale=preset.source_emission_scale,
        duration_scale=preset.release_duration_scale,
    )
    scenario_stability_class = _shift_stability_class(
        incident.stability_class or terrain_cfg.stability_class,
        preset.stability_shift,
    )
    _emit_progress(
        progress_callback,
        f"4/6 [uncertainty] {preset.label}: running {len(preset.direction_offsets_deg)} member(s) with "
        f"wind speed scale {preset.wind_speed_scale:.2f}, source scale {preset.source_emission_scale:.2f}, "
        f"duration scale {preset.release_duration_scale:.2f}, stability {scenario_stability_class}.",
    )

    member_peaks: list[np.ndarray] = []
    member_first_impact_minutes: list[dict[str, np.ndarray]] = []
    for member_index, direction_offset_deg in enumerate(preset.direction_offsets_deg, start=1):
        _emit_progress(
            progress_callback,
            f"4/6 [uncertainty {preset.label} {member_index}/{len(preset.direction_offsets_deg)}] "
            f"wind direction offset {direction_offset_deg:+.0f} degrees.",
        )
        _, _, member_peak, member_first_impact = _simulate_transport_member(
            incident=incident,
            source_latitude=source_latitude,
            source_longitude=source_longitude,
            terrain=terrain,
            terrain_cfg=terrain_cfg,
            terrain_flow_field=terrain_flow_field,
            weather_window=weather_window,
            source_term=scenario_source_term,
            material_fate=material_fate,
            geo_neighborhoods=None,
            wind_direction_offset_deg=direction_offset_deg,
            wind_speed_scale=preset.wind_speed_scale,
            stability_class=scenario_stability_class,
            frame_interval_minutes=frame_interval_minutes,
            progress_callback=progress_callback,
            store_frames=False,
            store_reports=False,
        )
        member_peaks.append(member_peak)
        member_first_impact_minutes.append(member_first_impact)

    if preset.aggregation == "baseline":
        peak_concentration = member_peaks[0]
    else:
        peak_concentration = np.max(np.stack(member_peaks, axis=0), axis=0)

    first_impact_minutes_by_band = {
        band: _nanmin_stack([member[band] for member in member_first_impact_minutes])
        for band, _ in _band_threshold_items()
    }
    return ScenarioAggregate(
        scenario_name=preset.label,
        scenario_description=preset.description,
        aggregation=preset.aggregation,
        member_count=len(preset.direction_offsets_deg),
        direction_offsets_deg=tuple(float(value) for value in preset.direction_offsets_deg),
        wind_speed_scale=float(preset.wind_speed_scale),
        source_emission_scale=float(preset.source_emission_scale),
        release_duration_scale=float(preset.release_duration_scale),
        stability_class=scenario_stability_class,
        peak_concentration=peak_concentration,
        first_impact_minutes_by_band=first_impact_minutes_by_band,
    )


def _uncertainty_summary_row(
    aggregate: ScenarioAggregate,
    terrain: TerrainSurface,
) -> dict[str, Any]:
    cell_area_sqkm = _grid_cell_area_sqkm(terrain)
    return {
        "ScenarioName": aggregate.scenario_name,
        "ScenarioDescription": aggregate.scenario_description,
        "Aggregation": aggregate.aggregation,
        "MemberCount": aggregate.member_count,
        "DirectionOffsetsDeg": ", ".join(f"{offset:+.0f}" for offset in aggregate.direction_offsets_deg),
        "WindSpeedScale": aggregate.wind_speed_scale,
        "SourceEmissionScale": aggregate.source_emission_scale,
        "ReleaseDurationScale": aggregate.release_duration_scale,
        "StabilityClass": aggregate.stability_class,
        "PeakConcentrationMax": float(np.nanmax(aggregate.peak_concentration)),
        "LowAreaSqKm": float(np.count_nonzero(aggregate.peak_concentration >= EMERGENCY_ALERT_THRESHOLDS["LOW"]) * cell_area_sqkm),
        "MediumAreaSqKm": float(np.count_nonzero(aggregate.peak_concentration >= EMERGENCY_ALERT_THRESHOLDS["MEDIUM"]) * cell_area_sqkm),
        "HighAreaSqKm": float(np.count_nonzero(aggregate.peak_concentration >= EMERGENCY_ALERT_THRESHOLDS["HIGH"]) * cell_area_sqkm),
    }


def build_uncertainty_summary(
    aggregates: list[ScenarioAggregate],
    terrain: TerrainSurface,
) -> pd.DataFrame:
    if not aggregates:
        return empty_uncertainty_summary()
    return pd.DataFrame(
        [_uncertainty_summary_row(aggregate, terrain) for aggregate in aggregates],
        columns=UNCERTAINTY_SUMMARY_COLUMNS,
    )


def _rank_action_polygons(action_polygons: pd.DataFrame) -> pd.DataFrame:
    if action_polygons.empty:
        return action_polygons

    ranked = action_polygons.copy()
    ranked["_NoticePriority"] = ranked["NoticeLevel"].map(NOTICE_LEVEL_PRIORITY).fillna(0)
    ranked["_UrgencyPriority"] = ranked["CAPUrgency"].map(CAP_URGENCY_PRIORITY).fillna(0)
    ranked["_ScenarioPriority"] = ranked["ScenarioName"].map(SCENARIO_PRIORITY).fillna(99)
    ranked["_FirstImpactSort"] = pd.to_datetime(
        ranked["FirstImpactedTime"],
        utc=True,
        errors="coerce",
    ).fillna(pd.Timestamp("2262-01-01T00:00:00Z"))
    ranked = ranked.sort_values(
        [
            "_NoticePriority",
            "_UrgencyPriority",
            "_ScenarioPriority",
            "_FirstImpactSort",
            "PeakConcentration",
            "AreaSqKm",
        ],
        ascending=[False, False, True, True, False, False],
    ).reset_index(drop=True)
    ranked["BroadcastPriorityRank"] = np.arange(1, len(ranked) + 1)
    return ranked.drop(
        columns=[
            "_NoticePriority",
            "_UrgencyPriority",
            "_ScenarioPriority",
            "_FirstImpactSort",
        ]
    )


def build_action_polygon_report(
    *,
    source_latitude: float,
    source_longitude: float,
    terrain: TerrainSurface,
    weather_window: pd.DataFrame,
    aggregates: list[ScenarioAggregate],
    progress_callback: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    if not aggregates or weather_window.empty:
        return empty_action_polygon_report()

    start_timestamp = pd.Timestamp(weather_window.iloc[0]["timestamp"])
    flat_points = np.column_stack([terrain.x_km.ravel(), terrain.y_km.ravel()])
    band_levels = [
        ("HIGH", EMERGENCY_ALERT_THRESHOLDS["HIGH"]),
        ("MEDIUM", EMERGENCY_ALERT_THRESHOLDS["MEDIUM"]),
        ("LOW", EMERGENCY_ALERT_THRESHOLDS["LOW"]),
    ]
    rows: list[dict[str, Any]] = []
    _emit_progress(
        progress_callback,
        "5/6 [polygons] Deriving layered action polygons from the likely, conservative, and worst-reasonable envelopes.",
    )
    for aggregate in aggregates:
        _emit_progress(
            progress_callback,
            f"5/6 [polygons] {aggregate.scenario_name}: extracting threshold polygons.",
        )
        for band, threshold in band_levels:
            polygons = _extract_threshold_polygons(
                terrain.x_km,
                terrain.y_km,
                aggregate.peak_concentration,
                threshold,
            )
            for polygon_index, polygon_xy in enumerate(polygons, start=1):
                polygon_path = Path(polygon_xy)
                polygon_mask = polygon_path.contains_points(flat_points, radius=1e-9).reshape(terrain.x_km.shape)
                if not np.any(polygon_mask):
                    centroid_x_km, centroid_y_km = _polygon_centroid(polygon_xy)
                    nearest_idx = np.unravel_index(
                        int(
                            np.argmin(
                                (terrain.x_km - centroid_x_km) ** 2
                                + (terrain.y_km - centroid_y_km) ** 2
                            )
                        ),
                        terrain.x_km.shape,
                    )
                    polygon_mask[nearest_idx] = True
                peak_concentration = float(np.nanmax(aggregate.peak_concentration[polygon_mask]))
                first_minutes_values = aggregate.first_impact_minutes_by_band[band][polygon_mask]
                valid_first_minutes = first_minutes_values[~np.isnan(first_minutes_values)]
                first_minutes_to_impact = (
                    float(np.min(valid_first_minutes)) if valid_first_minutes.size else None
                )
                first_impacted_time = (
                    start_timestamp + pd.Timedelta(minutes=first_minutes_to_impact)
                    if first_minutes_to_impact is not None
                    else pd.NaT
                )
                decision = broadcast_decision(
                    band,
                    first_impacted_time=first_impacted_time,
                    reference_time=start_timestamp,
                    peak_concentration=peak_concentration,
                    minutes_until_impact=first_minutes_to_impact,
                )
                area_sqkm = _polygon_area_sqkm(polygon_xy)
                centroid_x_km, centroid_y_km = _polygon_centroid(polygon_xy)
                centroid_latitude, centroid_longitude = local_km_to_latlon(
                    centroid_x_km,
                    centroid_y_km,
                    origin_latitude=source_latitude,
                    origin_longitude=source_longitude,
                )
                ring_latitude, ring_longitude = local_km_to_latlon(
                    polygon_xy[:, 0],
                    polygon_xy[:, 1],
                    origin_latitude=source_latitude,
                    origin_longitude=source_longitude,
                )
                coordinates = [
                    [
                        [float(lon), float(lat)]
                        for lon, lat in zip(ring_longitude, ring_latitude)
                    ]
                ]
                area_label = f"{aggregate.scenario_name} {decision['NoticeLevel']} polygon {polygon_index}"
                feature_id = (
                    f"{aggregate.scenario_name.lower().replace(' ', '-')}-"
                    f"{band.lower()}-{polygon_index}"
                )
                geojson_feature = {
                    "type": "Feature",
                    "id": feature_id,
                    "properties": {
                        "area_label": area_label,
                        "scenario_name": aggregate.scenario_name,
                        "scenario_description": aggregate.scenario_description,
                        "aggregation": aggregate.aggregation,
                        "member_count": aggregate.member_count,
                        "band": band,
                        "notice_level": decision["NoticeLevel"],
                        "cap_severity": decision["CAPSeverity"],
                        "cap_urgency": decision["CAPUrgency"],
                        "minutes_to_impact": first_minutes_to_impact,
                        "first_impacted_time": (
                            first_impacted_time.isoformat() if pd.notna(first_impacted_time) else None
                        ),
                        "peak_concentration": peak_concentration,
                        "area_sqkm": area_sqkm,
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": coordinates,
                    },
                }
                rows.append(
                    {
                        "AreaLabel": area_label,
                        "ScenarioName": aggregate.scenario_name,
                        "ScenarioDescription": aggregate.scenario_description,
                        "Aggregation": aggregate.aggregation,
                        "MemberCount": aggregate.member_count,
                        "Band": band,
                        "FirstImpactedTime": first_impacted_time,
                        "PeakConcentration": peak_concentration,
                        "AreaSqKm": area_sqkm,
                        "CentroidLatitude": float(centroid_latitude),
                        "CentroidLongitude": float(centroid_longitude),
                        "RecommendedAction": recommended_action_for_band(band),
                        "GeoJSONCoordinates": coordinates,
                        "GeoJSONFeature": geojson_feature,
                        **decision,
                    }
                )

    if not rows:
        return empty_action_polygon_report()

    return _rank_action_polygons(pd.DataFrame(rows, columns=ACTION_POLYGON_COLUMNS))


def build_action_polygon_geojson(action_polygons: pd.DataFrame) -> dict[str, Any]:
    if action_polygons.empty:
        return {"type": "FeatureCollection", "features": []}
    return {
        "type": "FeatureCollection",
        "features": [
            row["GeoJSONFeature"]
            for row in action_polygons.to_dict(orient="records")
            if row.get("GeoJSONFeature") is not None
        ],
    }


def build_action_polygon_payloads(action_polygons: pd.DataFrame) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    if action_polygons.empty:
        return payloads

    impacted = action_polygons[action_polygons["BroadcastRecommended"]].reset_index(drop=True)
    for row in impacted.to_dict(orient="records"):
        payloads.append(
            {
                "area_label": row["AreaLabel"],
                "scenario_name": row["ScenarioName"],
                "band": row["Band"],
                "notice_level": row.get("NoticeLevel"),
                "cap_severity": row.get("CAPSeverity"),
                "cap_urgency": row.get("CAPUrgency"),
                "broadcast_priority_rank": row.get("BroadcastPriorityRank"),
                "first_impacted_time": row.get("FirstImpactedTime"),
                "minutes_to_impact": row.get("MinutesToImpact"),
                "peak_concentration": row.get("PeakConcentration"),
                "area_sqkm": row.get("AreaSqKm"),
                "centroid_latitude": row.get("CentroidLatitude"),
                "centroid_longitude": row.get("CentroidLongitude"),
                "recommended_action": row.get("RecommendedAction"),
                "broadcast_basis": row.get("BroadcastBasis"),
                "geojson": row.get("GeoJSONFeature"),
            }
        )
    return payloads


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
        decision = broadcast_decision(
            band,
            first_impacted_time=timestamp,
            reference_time=timestamp,
            peak_concentration=concentration,
        )
        rows.append(
            {
                "Neighborhood": name,
                "Latitude": float(latitude),
                "Longitude": float(longitude),
                "X_km": float(x_km),
                "Y_km": float(y_km),
                "Concentration": concentration,
                "Band": band,
                "FirstImpactedTime": timestamp,
                "RecommendedAction": action,
                "DraftNotice": build_geo_notice_message(
                    neighborhood=name,
                    band=band,
                    action=action,
                    incident_name=incident_name,
                    incident_type=incident_type,
                    source_latitude=source_latitude,
                    source_longitude=source_longitude,
                    notice_level=decision["NoticeLevel"],
                    cap_urgency=decision["CAPUrgency"],
                    timestamp=timestamp,
                ),
                **decision,
            }
        )
    return rank_broadcast_areas(
        pd.DataFrame(rows),
        concentration_column="Concentration",
        first_impacted_time_column="FirstImpactedTime",
    )


def summarize_forecast_impacts(
    incident_name: str,
    incident_type: str,
    source_latitude: float,
    source_longitude: float,
    frame_reports: list[pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    all_reports = pd.concat(frame_reports, ignore_index=True)
    reference_time = all_reports["Timestamp"].min() if not all_reports.empty else pd.NaT
    summary_rows: list[dict[str, Any]] = []
    for neighborhood, group in all_reports.groupby("Neighborhood", sort=False):
        peak_idx = group["Concentration"].idxmax()
        peak_row = group.loc[peak_idx]
        impacted_group = group[group["BroadcastRecommended"]].sort_values("Timestamp")
        first_impacted_time = (
            impacted_group.iloc[0]["Timestamp"] if not impacted_group.empty else pd.NaT
        )
        decision = broadcast_decision(
            peak_row["Band"],
            first_impacted_time=first_impacted_time,
            reference_time=reference_time,
            peak_concentration=peak_row["Concentration"],
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
                "DraftNotice": build_geo_notice_message(
                    neighborhood=neighborhood,
                    band=peak_row["Band"],
                    action=peak_row["RecommendedAction"],
                    incident_name=incident_name,
                    incident_type=incident_type,
                    source_latitude=source_latitude,
                    source_longitude=source_longitude,
                    notice_level=decision["NoticeLevel"],
                    cap_urgency=decision["CAPUrgency"],
                    timestamp=first_impacted_time if pd.notna(first_impacted_time) else None,
                ),
                **decision,
            }
        )

    summary = pd.DataFrame(summary_rows)
    if summary.empty:
        return summary, summary, []

    summary = rank_broadcast_areas(
        summary,
        concentration_column="PeakConcentration",
        first_impacted_time_column="FirstImpactedTime",
    )
    impacted = summary[summary["BroadcastRecommended"]].reset_index(drop=True)
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
            road=None,
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
    progress_callback: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    if not frames:
        return empty_top_location_report()

    concentration_stack = np.stack([frame.concentration for frame in frames], axis=0)
    reference_time = frames[0].timestamp
    peak_frame_indices = np.argmax(concentration_stack, axis=0)
    peak_concentration = np.max(concentration_stack, axis=0)
    hotspot_indices = select_top_hotspot_indices(
        terrain=terrain,
        peak_concentration=peak_concentration,
        max_locations=max_locations,
        min_spacing_km=min_spacing_km,
    )
    _emit_progress(
        progress_callback,
        f"5/6 [geocoding] Resolving labels for {len(hotspot_indices)} peak hotspot locations.",
    )

    rows: list[dict[str, Any]] = []
    hotspot_count = len(hotspot_indices)
    for hotspot_index, (row_idx, col_idx) in enumerate(hotspot_indices, start=1):
        latitude = float(terrain.latitude_grid[row_idx, col_idx])
        longitude = float(terrain.longitude_grid[row_idx, col_idx])
        _emit_progress(
            progress_callback,
            f"5/6 [geocoding {hotspot_index}/{hotspot_count}] "
            f"Looking up labels near ({latitude:.4f}, {longitude:.4f}).",
        )
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
        decision = broadcast_decision(
            band,
            first_impacted_time=first_impacted_time,
            reference_time=reference_time,
            peak_concentration=peak_value,
        )
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
                "DraftNotice": build_geo_notice_message(
                    neighborhood=label,
                    band=band,
                    action=action,
                    incident_name=incident_name,
                    incident_type=incident_type,
                    source_latitude=source_latitude,
                    source_longitude=source_longitude,
                    notice_level=decision["NoticeLevel"],
                    cap_urgency=decision["CAPUrgency"],
                    timestamp=first_impacted_time if pd.notna(first_impacted_time) else peak_time,
                ),
                **decision,
            }
        )

    if not rows:
        return empty_top_location_report()

    report = rank_broadcast_areas(
        pd.DataFrame(rows),
        concentration_column="PeakConcentration",
        first_impacted_time_column="FirstImpactedTime",
    )
    report = report.drop_duplicates(subset=["LocationLabel"], keep="first")
    report = report.reset_index(drop=True)
    report["BroadcastPriorityRank"] = np.arange(1, len(report) + 1)
    _emit_progress(progress_callback, "5/6 [geocoding] Hotspot reverse-geocoding complete.")
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
                "NoticeLevel",
                "CAPSeverity",
                "CAPUrgency",
                "MinutesToImpact",
                "RecommendedAction",
                "BroadcastRecommended",
                "BroadcastBasis",
                "BroadcastPriorityRank",
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
                "NoticeLevel",
                "CAPSeverity",
                "CAPUrgency",
                "MinutesToImpact",
                "RecommendedAction",
                "BroadcastRecommended",
                "BroadcastBasis",
                "BroadcastPriorityRank",
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
        }
        minutes_values = group["MinutesToImpact"] if "MinutesToImpact" in group.columns else pd.Series(dtype=float)
        minutes_series = pd.to_numeric(minutes_values, errors="coerce")
        decision = broadcast_decision(
            peak_row["PeakBand"],
            first_impacted_time=row["FirstImpactedTime"],
            peak_concentration=peak_row["PeakConcentration"],
            minutes_until_impact=(
                float(minutes_series.dropna().min())
                if not minutes_series.dropna().empty
                else None
            ),
        )
        row.update(decision)
        for extra_column in extra_columns:
            row[extra_column] = peak_row.get(extra_column)
        rows.append(row)

    summary = pd.DataFrame(rows)
    return rank_broadcast_areas(
        summary,
        concentration_column="PeakConcentration",
        first_impacted_time_column="FirstImpactedTime",
        hotspot_count_column="HotspotCount",
    )


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
                "notice_level": row.get("NoticeLevel"),
                "cap_severity": row.get("CAPSeverity"),
                "cap_urgency": row.get("CAPUrgency"),
                "broadcast_priority_rank": row.get("BroadcastPriorityRank"),
                "recommended_action": row["RecommendedAction"],
                "peak_time": row["PeakTime"],
                "first_impacted_time": row["FirstImpactedTime"],
                "broadcast_basis": row.get("BroadcastBasis"),
                "message": row["DraftNotice"],
            }
        )
    return payloads


def _emit_progress(progress_callback: Callable[[str], None] | None, message: str) -> None:
    if progress_callback is not None:
        progress_callback(message)


def _resolve_effective_grid_resolution(
    simulation_resolution: int,
    terrain_resolution: int | None,
) -> int:
    if terrain_resolution is None:
        return int(simulation_resolution)
    return max(int(simulation_resolution), int(terrain_resolution))


def _trajectory_positions_from_weather(
    weather_window: pd.DataFrame,
    *,
    wind_direction_offset_deg: float = 0.0,
    wind_speed_scale: float = 1.0,
) -> np.ndarray:
    if weather_window.empty:
        return np.array([[0.0, 0.0]], dtype=float)

    positions: list[tuple[float, float]] = [(0.0, 0.0)]
    current_x_km = 0.0
    current_y_km = 0.0
    previous_timestamp: pd.Timestamp | None = None
    previous_velocity: tuple[float, float] | None = None
    for weather_row in weather_window.to_dict(orient="records"):
        frame_timestamp = pd.Timestamp(weather_row["timestamp"])
        adjusted_wind_from_deg = (float(weather_row["wind_from_deg"]) + float(wind_direction_offset_deg)) % 360.0
        adjusted_wind_speed_mps = max(
            float(weather_row["wind_speed_mps"]) * float(wind_speed_scale),
            MIN_EFFECTIVE_TRANSPORT_SPEED_MPS,
        )
        current_velocity = _wind_velocity_components(
            adjusted_wind_from_deg,
            adjusted_wind_speed_mps,
        )
        if previous_timestamp is not None:
            dt_s = max((frame_timestamp - previous_timestamp).total_seconds(), 0.0)
            average_vx_mps = 0.5 * ((previous_velocity or current_velocity)[0] + current_velocity[0])
            average_vy_mps = 0.5 * ((previous_velocity or current_velocity)[1] + current_velocity[1])
            current_x_km += (average_vx_mps * dt_s) / 1000.0
            current_y_km += (average_vy_mps * dt_s) / 1000.0
            positions.append((current_x_km, current_y_km))
        previous_timestamp = frame_timestamp
        previous_velocity = current_velocity
    return np.asarray(positions, dtype=float)


def _stability_domain_spread_factor(stability_class: str | None) -> float:
    normalized = str(stability_class or DEFAULT_CONFIG.stability_class).strip().upper()
    return {
        "A": 0.28,
        "B": 0.26,
        "C": 0.24,
        "D": 0.22,
        "E": 0.19,
        "F": 0.16,
    }.get(normalized, 0.22)


def _estimate_forecast_domain_cfg(
    *,
    base_cfg: PlumeConfig,
    effective_resolution: int,
    weather_window: pd.DataFrame,
    geo_neighborhoods: dict[str, tuple[float, float]] | None,
    source_latitude: float,
    source_longitude: float,
    source_term: ResolvedSourceTerm,
    stability_class: str | None,
    progress_callback: Callable[[str], None] | None = None,
) -> PlumeConfig:
    if weather_window.empty:
        return copy_config(
            base_cfg,
            source_x_km=0.0,
            source_y_km=0.0,
            resolution=effective_resolution,
        )

    start_timestamp = pd.Timestamp(weather_window.iloc[0]["timestamp"])
    total_duration_hours = max(
        (pd.Timestamp(weather_window.iloc[-1]["timestamp"]) - start_timestamp).total_seconds() / 3600.0,
        0.0,
    )
    release_duration_hours = max(float(source_term.release_duration_minutes) / 60.0, 0.0)
    impact_horizon_hours = min(
        total_duration_hours,
        max(release_duration_hours + 6.0, 8.0),
    )
    domain_weather_window = weather_window[
        pd.to_datetime(weather_window["timestamp"], utc=True)
        <= start_timestamp + pd.Timedelta(hours=impact_horizon_hours)
    ].reset_index(drop=True)
    if domain_weather_window.empty:
        domain_weather_window = weather_window

    max_direction_offset_deg = max(
        abs(float(offset))
        for preset in UNCERTAINTY_SCENARIOS
        for offset in preset.direction_offsets_deg
    )
    max_wind_speed_scale = max(float(preset.wind_speed_scale) for preset in UNCERTAINTY_SCENARIOS)
    trajectory_offsets_deg = (-max_direction_offset_deg, 0.0, max_direction_offset_deg)
    trajectory_positions = [
        _trajectory_positions_from_weather(
            domain_weather_window,
            wind_direction_offset_deg=offset_deg,
            wind_speed_scale=max_wind_speed_scale,
        )
        for offset_deg in trajectory_offsets_deg
    ]
    combined_positions = np.vstack(trajectory_positions) if trajectory_positions else np.array([[0.0, 0.0]], dtype=float)
    max_displacement_km = float(np.max(np.hypot(combined_positions[:, 0], combined_positions[:, 1])))
    spread_factor = _stability_domain_spread_factor(stability_class)
    release_duration_hours = max(float(source_term.release_duration_minutes) / 60.0, 0.0)
    domain_margin_km = max(
        10.0,
        max_displacement_km * (spread_factor + 0.18),
        release_duration_hours * 3.0,
    )

    min_x_km = min(float(np.min(combined_positions[:, 0])) - domain_margin_km, float(base_cfg.x_min_km), 0.0)
    max_x_km = max(float(np.max(combined_positions[:, 0])) + domain_margin_km, float(base_cfg.x_max_km), 0.0)
    min_y_km = min(float(np.min(combined_positions[:, 1])) - domain_margin_km, float(base_cfg.y_min_km), 0.0)
    max_y_km = max(float(np.max(combined_positions[:, 1])) + domain_margin_km, float(base_cfg.y_max_km), 0.0)

    if geo_neighborhoods:
        x_values: list[float] = []
        y_values: list[float] = []
        for latitude, longitude in geo_neighborhoods.values():
            x_km, y_km = latlon_to_local_km(
                latitude,
                longitude,
                origin_latitude=source_latitude,
                origin_longitude=source_longitude,
            )
            x_values.append(float(x_km))
            y_values.append(float(y_km))
        if x_values and y_values:
            min_x_km = min(min_x_km, min(x_values) - 2.0)
            max_x_km = max(max_x_km, max(x_values) + 2.0)
            min_y_km = min(min_y_km, min(y_values) - 2.0)
            max_y_km = max(max_y_km, max(y_values) + 2.0)

    rounded_min_x_km = float(np.floor(min_x_km / 5.0) * 5.0)
    rounded_max_x_km = float(np.ceil(max_x_km / 5.0) * 5.0)
    rounded_min_y_km = float(np.floor(min_y_km / 5.0) * 5.0)
    rounded_max_y_km = float(np.ceil(max_y_km / 5.0) * 5.0)
    x_span_km = max(rounded_max_x_km - rounded_min_x_km, 1.0)
    y_span_km = max(rounded_max_y_km - rounded_min_y_km, 1.0)
    base_x_span_km = max(float(base_cfg.x_max_km - base_cfg.x_min_km), 1.0)
    base_y_span_km = max(float(base_cfg.y_max_km - base_cfg.y_min_km), 1.0)
    target_cell_size_km = max(
        base_x_span_km / max(int(effective_resolution) - 1, 1),
        base_y_span_km / max(int(effective_resolution) - 1, 1),
    )
    adaptive_resolution = max(
        int(effective_resolution),
        int(np.ceil(max(x_span_km, y_span_km) / max(target_cell_size_km, 0.5))) + 1,
    )
    adaptive_resolution = min(adaptive_resolution, MAX_DYNAMIC_FORECAST_GRID_RESOLUTION)

    expanded_cfg = copy_config(
        base_cfg,
        source_x_km=0.0,
        source_y_km=0.0,
        resolution=adaptive_resolution,
        x_min_km=rounded_min_x_km,
        x_max_km=rounded_max_x_km,
        y_min_km=rounded_min_y_km,
        y_max_km=rounded_max_y_km,
    )
    if (
        rounded_min_x_km < float(base_cfg.x_min_km)
        or rounded_max_x_km > float(base_cfg.x_max_km)
        or rounded_min_y_km < float(base_cfg.y_min_km)
        or rounded_max_y_km > float(base_cfg.y_max_km)
    ):
        _emit_progress(
            progress_callback,
            "2/6 [terrain] Expanded the map domain to "
            f"x={rounded_min_x_km:.0f}..{rounded_max_x_km:.0f} km, "
            f"y={rounded_min_y_km:.0f}..{rounded_max_y_km:.0f} km "
            "to keep the likely impacted area on the map.",
        )
    if adaptive_resolution > int(effective_resolution):
        _emit_progress(
            progress_callback,
            "2/6 [terrain] Increased the simulation grid to "
            f"{adaptive_resolution}x{adaptive_resolution} so neighborhood sampling stays accurate "
            "after the map extent expansion.",
        )
    return expanded_cfg


def simulate_forecast_alarm(
    source_latitude: float,
    source_longitude: float,
    incident_time: str | None,
    severity: str | float,
    incident_type: str,
    name: str,
    neighborhoods: dict[str, tuple[float, float]] | None = None,
    base_cfg: PlumeConfig = DEFAULT_CONFIG,
    duration_hours: int | float | str | None = "auto",
    forecast_hours: int = 48,
    simulation_resolution: int = DEFAULT_INTERNAL_FORECAST_RESOLUTION,
    terrain_resolution: int | None = DEFAULT_TERRAIN_SURFACE_RESOLUTION,
    frame_interval_minutes: int = DEFAULT_FORECAST_FRAME_INTERVAL_MINUTES,
    stability_class: str | None = None,
    release_height_m: float | None = None,
    source_term_profile: str | None = None,
    hazard_material: str | None = None,
    emission_rate_override: float | None = None,
    release_duration_minutes: float | None = None,
    initial_pulse_minutes: float | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> ForecastAlarmSimulation:
    effective_resolution = _resolve_effective_grid_resolution(
        simulation_resolution=simulation_resolution,
        terrain_resolution=terrain_resolution,
    )
    incident = HazardIncident(
        name=name,
        source_x_km=0.0,
        source_y_km=0.0,
        severity=severity,
        incident_type=incident_type,
        stability_class=stability_class,
        release_height_m=release_height_m,
        source_term_profile=source_term_profile,
        hazard_material=hazard_material,
        emission_rate_override=emission_rate_override,
        release_duration_minutes=release_duration_minutes,
        initial_pulse_minutes=initial_pulse_minutes,
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
    source_term = resolve_incident_source_term(incident, base_cfg=base_cfg)
    material_fate = resolve_material_fate(incident.hazard_material)
    auto_duration = _duration_hours_is_auto(duration_hours)
    if auto_duration:
        planning_duration_hours = max(
            AUTO_DURATION_MIN_HOURS,
            min(
                AUTO_DURATION_MAX_HOURS,
                max(float(source_term.release_duration_minutes) / 60.0, 0.0)
                + _material_airborne_cutoff_hours(material_fate)[0]
                + AUTO_DURATION_BUFFER_HOURS,
            ),
        )
    else:
        planning_duration_hours = _manual_duration_hours(duration_hours)

    _emit_progress(
        progress_callback,
        f"1/6 [weather] Fetching hourly weather forecast near ({source_latitude:.4f}, {source_longitude:.4f}).",
    )
    required_forecast_hours = max(int(forecast_hours), int(np.ceil(planning_duration_hours)) + 2)
    weather_df = fetch_hourly_weather_forecast(
        latitude=source_latitude,
        longitude=source_longitude,
        forecast_hours=required_forecast_hours,
        progress_callback=progress_callback,
    )
    if auto_duration:
        resolved_duration_hours, duration_basis = _resolve_auto_duration_hours(
            weather_df=weather_df,
            incident_time=incident_time,
            source_term=source_term,
            material_fate=material_fate,
            stability_class=incident.stability_class or base_cfg.stability_class,
            frame_interval_minutes=frame_interval_minutes,
        )
        _emit_progress(
            progress_callback,
            "1/6 [duration] Auto-selected simulation horizon: "
            f"{resolved_duration_hours:.1f} h ({duration_basis}).",
        )
    else:
        resolved_duration_hours = planning_duration_hours
        duration_basis = "manual duration override"
    weather_window = select_weather_window(
        weather_df=weather_df,
        incident_time=incident_time,
        duration_hours=resolved_duration_hours,
        frame_interval_minutes=frame_interval_minutes,
    )
    _emit_progress(
        progress_callback,
        "1/6 [weather] Forecast window selected: "
        f"{len(weather_window)} steps at {frame_interval_minutes}-minute intervals from "
        f"{weather_window.iloc[0]['timestamp'].strftime('%Y-%m-%d %H:%M UTC')} to "
        f"{weather_window.iloc[-1]['timestamp'].strftime('%Y-%m-%d %H:%M UTC')}.",
    )
    _emit_progress(
        progress_callback,
        "2/6 [terrain] Sampling public elevation data and building the terrain grid "
        f"at resolution {effective_resolution}x{effective_resolution}.",
    )

    terrain_cfg = _estimate_forecast_domain_cfg(
        base_cfg=base_cfg,
        effective_resolution=effective_resolution,
        weather_window=weather_window,
        geo_neighborhoods=geo_neighborhoods,
        source_latitude=source_latitude,
        source_longitude=source_longitude,
        source_term=source_term,
        stability_class=incident.stability_class or base_cfg.stability_class,
        progress_callback=progress_callback,
    )
    terrain = build_terrain_surface(
        source_latitude=source_latitude,
        source_longitude=source_longitude,
        cfg=terrain_cfg,
        progress_callback=progress_callback,
    )
    terrain_flow_field = build_terrain_flow_field(terrain)
    _emit_progress(
        progress_callback,
        "2/6 [terrain] Terrain steering, channeling, and orographic transport fields are ready.",
    )

    _emit_progress(
        progress_callback,
        "3/6 [source] Resolved source term: "
        f"{source_term.profile_label} | "
        f"{source_term.emission_rate:.0f} mass/s steady release for "
        f"{source_term.release_duration_minutes:.0f} min with an initial pulse of "
        f"{source_term.initial_pulse_mass:.0f} mass units.",
    )
    _emit_progress(
        progress_callback,
        "3/6 [fate] Airborne persistence: "
        f"{material_fate.profile_label} | "
        f"reactive half-life "
        f"{material_fate.reactive_airborne_half_life_minutes if material_fate.reactive_airborne_half_life_minutes is not None else 'n/a'} min, "
        f"deposition half-life "
        f"{material_fate.deposition_half_life_minutes if material_fate.deposition_half_life_minutes is not None else 'n/a'} min, "
        f"persistent airborne fraction {material_fate.residual_airborne_fraction:.2f}.",
    )
    _emit_progress(
        progress_callback,
        f"3/6 [plume] Simulating {len(weather_window)} transport snapshots with wind-turning and terrain-aware puff advection.",
    )
    frames, frame_reports, likely_peak_concentration, likely_first_impact_minutes = _simulate_transport_member(
        incident=incident,
        source_latitude=source_latitude,
        source_longitude=source_longitude,
        terrain=terrain,
        terrain_cfg=terrain_cfg,
        terrain_flow_field=terrain_flow_field,
        weather_window=weather_window,
        source_term=source_term,
        material_fate=material_fate,
        geo_neighborhoods=geo_neighborhoods,
        frame_interval_minutes=frame_interval_minutes,
        progress_callback=progress_callback,
        progress_prefix="3/6 [plume",
        progress_enabled=True,
        store_frames=True,
        store_reports=True,
    )
    _emit_progress(
        progress_callback,
        f"3/6 [plume] Completed {len(frames)} plume snapshots.",
    )

    likely_stability_class = _shift_stability_class(
        incident.stability_class or terrain_cfg.stability_class,
        UNCERTAINTY_SCENARIOS[0].stability_shift,
    )
    aggregates = [
        ScenarioAggregate(
            scenario_name=UNCERTAINTY_SCENARIOS[0].label,
            scenario_description=UNCERTAINTY_SCENARIOS[0].description,
            aggregation=UNCERTAINTY_SCENARIOS[0].aggregation,
            member_count=1,
            direction_offsets_deg=tuple(float(value) for value in UNCERTAINTY_SCENARIOS[0].direction_offsets_deg),
            wind_speed_scale=float(UNCERTAINTY_SCENARIOS[0].wind_speed_scale),
            source_emission_scale=float(UNCERTAINTY_SCENARIOS[0].source_emission_scale),
            release_duration_scale=float(UNCERTAINTY_SCENARIOS[0].release_duration_scale),
            stability_class=likely_stability_class,
            peak_concentration=likely_peak_concentration,
            first_impact_minutes_by_band=likely_first_impact_minutes,
        )
    ]
    for preset in UNCERTAINTY_SCENARIOS[1:]:
        aggregates.append(
            _aggregate_uncertainty_scenario(
                preset=preset,
                incident=incident,
                source_latitude=source_latitude,
                source_longitude=source_longitude,
                terrain=terrain,
                terrain_cfg=terrain_cfg,
                terrain_flow_field=terrain_flow_field,
                weather_window=weather_window,
                base_source_term=source_term,
                material_fate=material_fate,
                frame_interval_minutes=frame_interval_minutes,
                progress_callback=progress_callback,
            )
        )
    uncertainty_summary = build_uncertainty_summary(aggregates, terrain)
    action_polygons = build_action_polygon_report(
        source_latitude=source_latitude,
        source_longitude=source_longitude,
        terrain=terrain,
        weather_window=weather_window,
        aggregates=aggregates,
        progress_callback=progress_callback,
    )
    action_geojson = build_action_polygon_geojson(action_polygons)

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

    _emit_progress(
        progress_callback,
        "5/6 [geocoding] Reverse-geocoding the strongest simulated hotspots into neighborhood, city, and ZIP labels.",
    )
    top_locations = build_top_location_report(
        incident_name=name,
        incident_type=incident_type,
        source_latitude=source_latitude,
        source_longitude=source_longitude,
        terrain=terrain,
        frames=frames,
        progress_callback=progress_callback,
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
    notice_payloads = build_action_polygon_payloads(action_polygons)
    if not notice_payloads:
        notice_payloads = build_geo_notice_payloads(top_locations)

    if top_neighborhoods.empty:
        top_neighborhoods = neighborhood_summary.copy()
    if impacted_neighborhoods.empty and not top_neighborhoods.empty:
        impacted_neighborhoods = top_neighborhoods[top_neighborhoods["BroadcastRecommended"]].reset_index(
            drop=True
        )

    _emit_progress(
        progress_callback,
        "6/6 [output] Packaging the timelapse, uncertainty summary, action polygons, ranked impact tables, and draft notice payloads.",
    )

    return ForecastAlarmSimulation(
        incident=incident,
        source_term=source_term,
        material_fate=material_fate,
        resolved_duration_hours=resolved_duration_hours,
        duration_basis=duration_basis,
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
        uncertainty_summary=uncertainty_summary,
        action_polygons=action_polygons,
        action_geojson=action_geojson,
        notice_payloads=notice_payloads,
    )
