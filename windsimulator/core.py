from __future__ import annotations

import numpy as np

from .config import PlumeConfig


def make_grid(cfg: PlumeConfig):
    xs = np.linspace(cfg.x_min_km, cfg.x_max_km, cfg.resolution)
    ys = np.linspace(cfg.y_min_km, cfg.y_max_km, cfg.resolution)
    return np.meshgrid(xs, ys)


def wind_to_math_angle_rad(wind_from_deg: float) -> float:
    wind_to_deg = (wind_from_deg + 180.0) % 360.0
    return np.deg2rad(wind_to_deg)


def rotate_to_wind_frame(
    x_km,
    y_km,
    source_x_km: float,
    source_y_km: float,
    wind_from_deg: float,
):
    theta = wind_to_math_angle_rad(wind_from_deg)
    dx_m = (x_km - source_x_km) * 1000.0
    dy_m = (y_km - source_y_km) * 1000.0
    x_prime = np.cos(theta) * dx_m + np.sin(theta) * dy_m
    y_prime = -np.sin(theta) * dx_m + np.cos(theta) * dy_m
    return x_prime, y_prime


def sigma_yz(x_m, stability_class: str = "D"):
    x = np.maximum(x_m, 1.0)
    cls = stability_class.upper()
    if cls == "A":
        sy = 0.22 * x * (1 + 0.0001 * x) ** (-0.5)
        sz = 0.20 * x
    elif cls == "B":
        sy = 0.16 * x * (1 + 0.0001 * x) ** (-0.5)
        sz = 0.12 * x
    elif cls == "C":
        sy = 0.11 * x * (1 + 0.0001 * x) ** (-0.5)
        sz = 0.08 * x * (1 + 0.0002 * x) ** (-0.5)
    elif cls == "D":
        sy = 0.08 * x * (1 + 0.0001 * x) ** (-0.5)
        sz = 0.06 * x * (1 + 0.0015 * x) ** (-0.5)
    elif cls == "E":
        sy = 0.06 * x * (1 + 0.0001 * x) ** (-0.5)
        sz = 0.03 * x * (1 + 0.0003 * x) ** (-1.0)
    elif cls == "F":
        sy = 0.04 * x * (1 + 0.0001 * x) ** (-0.5)
        sz = 0.016 * x * (1 + 0.0003 * x) ** (-1.0)
    else:
        raise ValueError("stability_class must be one of A, B, C, D, E, F")
    return np.maximum(sy, 1.0), np.maximum(sz, 1.0)


def gaussian_plume_ground_level(
    x_m,
    y_m,
    q: float,
    wind_speed_mps: float,
    release_height_m: float,
    stability_class: str = "D",
    min_downwind_m: float = 10.0,
):
    concentration = np.zeros_like(x_m, dtype=float)
    mask = x_m > min_downwind_m
    if not np.any(mask):
        return concentration

    x_pos = x_m[mask]
    y_pos = y_m[mask]
    sigma_y, sigma_z = sigma_yz(x_pos, stability_class)
    prefactor = q / (np.pi * wind_speed_mps * sigma_y * sigma_z)
    crosswind = np.exp(-(y_pos**2) / (2.0 * sigma_y**2))
    vertical = np.exp(-(release_height_m**2) / (2.0 * sigma_z**2))
    concentration[mask] = prefactor * crosswind * vertical
    return concentration


def compute_concentration(cfg: PlumeConfig):
    x_km, y_km = make_grid(cfg)
    x_m, y_m = rotate_to_wind_frame(
        x_km,
        y_km,
        cfg.source_x_km,
        cfg.source_y_km,
        cfg.wind_from_deg,
    )
    concentration = gaussian_plume_ground_level(
        x_m=x_m,
        y_m=y_m,
        q=cfg.emission_rate,
        wind_speed_mps=max(cfg.wind_speed_mps, 0.1),
        release_height_m=cfg.release_height_m,
        stability_class=cfg.stability_class,
        min_downwind_m=cfg.min_downwind_m,
    )
    return x_km, y_km, concentration


def concentration_band(concentration: float, cfg: PlumeConfig) -> str:
    if concentration >= cfg.threshold_high:
        return "HIGH"
    if concentration >= cfg.threshold_medium:
        return "MEDIUM"
    if concentration >= cfg.threshold_low:
        return "LOW"
    return "MINIMAL"


def gaussian_puff_ground_level(
    x_m,
    y_m,
    t_s: float,
    q_total: float,
    wind_speed_mps: float,
    release_height_m: float,
    stability_class: str = "D",
    t_release_s: float = 0.0,
):
    if t_s <= t_release_s:
        return np.zeros_like(x_m, dtype=float)

    dt = t_s - t_release_s
    x_center = wind_speed_mps * dt
    x_for_sigma = max(x_center, 1.0)
    sigma_y, sigma_z = sigma_yz(np.array([x_for_sigma]), stability_class)
    sigma_y = sigma_y[0]
    sigma_z = sigma_z[0]
    sigma_x = max(sigma_y, 1.0)
    prefactor = q_total / (((2 * np.pi) ** 1.5) * sigma_x * sigma_y * sigma_z)
    along = np.exp(-((x_m - x_center) ** 2) / (2.0 * sigma_x**2))
    cross = np.exp(-(y_m**2) / (2.0 * sigma_y**2))
    vertical = np.exp(-(release_height_m**2) / (2.0 * sigma_z**2))
    return prefactor * along * cross * vertical


def gaussian_puff_ground_level_centered(
    x_km,
    y_km,
    center_x_km: float,
    center_y_km: float,
    heading_rad: float,
    q_total: float,
    travel_distance_m: float,
    release_height_m: float,
    stability_class: str = "D",
    receptor_elevation_m=None,
    source_elevation_m: float | None = None,
    min_spread_distance_m: float = 30.0,
    sigma_x_scale: float = 1.0,
    sigma_y_scale: float = 1.0,
    sigma_z_scale: float = 1.0,
):
    dx_m = (x_km - center_x_km) * 1000.0
    dy_m = (y_km - center_y_km) * 1000.0
    along_m = np.cos(heading_rad) * dx_m + np.sin(heading_rad) * dy_m
    cross_m = -np.sin(heading_rad) * dx_m + np.cos(heading_rad) * dy_m

    spread_distance_m = max(float(travel_distance_m), float(min_spread_distance_m), 1.0)
    sigma_y, sigma_z = sigma_yz(np.array([spread_distance_m]), stability_class)
    base_sigma_y = float(sigma_y[0])
    base_sigma_z = float(sigma_z[0])
    sigma_y = max(base_sigma_y * max(float(sigma_y_scale), 0.25), 1.0)
    sigma_z = max(base_sigma_z * max(float(sigma_z_scale), 0.25), 1.0)
    sigma_x = max(
        base_sigma_y * 1.35 * max(float(sigma_x_scale), 0.25),
        min_spread_distance_m * 0.5,
        1.0,
    )

    effective_release_height_m = max(float(release_height_m), 2.0)
    if receptor_elevation_m is not None and source_elevation_m is not None:
        elevation_offset_m = np.asarray(receptor_elevation_m, dtype=float) - float(source_elevation_m)
        # Higher terrain effectively lifts the plume centerline relative to the local ground;
        # lower terrain can slightly reduce that offset without driving it unrealistically to zero.
        effective_release_height_m = np.maximum(
            effective_release_height_m
            + np.maximum(elevation_offset_m, 0.0) * 0.35
            - np.maximum(-elevation_offset_m, 0.0) * 0.12,
            2.0,
        )

    prefactor = q_total / (((2.0 * np.pi) ** 1.5) * sigma_x * sigma_y * sigma_z)
    along = np.exp(-(along_m**2) / (2.0 * sigma_x**2))
    cross = np.exp(-(cross_m**2) / (2.0 * sigma_y**2))
    vertical = np.exp(-(effective_release_height_m**2) / (2.0 * sigma_z**2))
    return prefactor * along * cross * vertical


def compute_puff_concentration(
    cfg: PlumeConfig,
    t_s: float,
    puff_mass: float | None = None,
):
    puff_mass = puff_mass if puff_mass is not None else cfg.emission_rate * 60.0
    x_km, y_km = make_grid(cfg)
    x_m, y_m = rotate_to_wind_frame(
        x_km,
        y_km,
        cfg.source_x_km,
        cfg.source_y_km,
        cfg.wind_from_deg,
    )
    concentration = gaussian_puff_ground_level(
        x_m=x_m,
        y_m=y_m,
        t_s=t_s,
        q_total=puff_mass,
        wind_speed_mps=max(cfg.wind_speed_mps, 0.1),
        release_height_m=cfg.release_height_m,
        stability_class=cfg.stability_class,
    )
    return x_km, y_km, concentration
