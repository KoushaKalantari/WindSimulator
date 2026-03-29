from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .config import PlumeConfig
from .core import (
    concentration_band,
    gaussian_plume_ground_level,
    rotate_to_wind_frame,
)


def copy_config(cfg: PlumeConfig, **overrides) -> PlumeConfig:
    return replace(cfg, **overrides)


def sample_neighborhoods(
    cfg: PlumeConfig,
    neighborhoods: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for name, (x_km, y_km) in neighborhoods.items():
        x_coords = np.array([[x_km]])
        y_coords = np.array([[y_km]])
        x_m, y_m = rotate_to_wind_frame(
            x_coords,
            y_coords,
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
        )[0, 0]
        rows.append(
            {
                "Neighborhood": name,
                "X_km": x_km,
                "Y_km": y_km,
                "Concentration": concentration,
                "Band": concentration_band(concentration, cfg),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("Concentration", ascending=False)
        .reset_index(drop=True)
    )


def scenario_configs(
    base_cfg: PlumeConfig,
    scenarios: list[dict[str, float | str]],
) -> list[tuple[str, PlumeConfig]]:
    resolved: list[tuple[str, PlumeConfig]] = []
    for scenario in scenarios:
        scenario_name = str(scenario["Scenario"])
        overrides = {key: value for key, value in scenario.items() if key != "Scenario"}
        resolved.append((scenario_name, copy_config(base_cfg, **overrides)))
    return resolved


def scenario_summary(
    base_cfg: PlumeConfig,
    scenarios: list[dict[str, float | str]],
    neighborhoods: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    tables = []
    for scenario_name, cfg in scenario_configs(base_cfg, scenarios):
        table = sample_neighborhoods(cfg, neighborhoods)
        table.insert(0, "Scenario", scenario_name)
        tables.append(table)
    if not tables:
        return pd.DataFrame(
            columns=["Scenario", "Neighborhood", "X_km", "Y_km", "Concentration", "Band"]
        )
    return pd.concat(tables, ignore_index=True)


def scenario_pivots(scenario_df: pd.DataFrame):
    concentration_pivot = scenario_df.pivot_table(
        index="Neighborhood",
        columns="Scenario",
        values="Concentration",
        aggfunc="max",
    )
    band_pivot = scenario_df.pivot_table(
        index="Neighborhood",
        columns="Scenario",
        values="Band",
        aggfunc="first",
    )
    return concentration_pivot, band_pivot


def sample_grid_value(
    x_grid,
    y_grid,
    values,
    x_km: float,
    y_km: float,
) -> float:
    distance = (x_grid - x_km) ** 2 + (y_grid - y_km) ** 2
    row_idx, col_idx = np.unravel_index(np.argmin(distance), distance.shape)
    return float(values[row_idx, col_idx])
