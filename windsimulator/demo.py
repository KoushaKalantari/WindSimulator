from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .analysis import sample_neighborhoods, scenario_configs, scenario_pivots, scenario_summary
from .config import DEFAULT_CONFIG, DEFAULT_SCENARIOS, FICTIONAL_NEIGHBORHOODS, MODEL_NOTES, PlumeConfig
from .emergency import AlertPlan, HazardIncident, analyze_incident_alert
from .forecast_alarm import ForecastAlarmSimulation, simulate_forecast_alarm
from .plotting import (
    animate_forecast_alarm,
    animate_puff,
    plot_alert_overlay,
    plot_forecast_alarm_overlay,
    plot_plume,
    plot_puff,
)
from .widgets import interactive_plume_demo


@dataclass
class DemoResults:
    cfg: PlumeConfig
    neighborhoods: dict[str, tuple[float, float]]
    scenarios: list[dict[str, float | str]]
    base_df: pd.DataFrame
    scenario_df: pd.DataFrame
    concentration_pivot: pd.DataFrame
    band_pivot: pd.DataFrame
    animation: Any | None = None


def _display(obj) -> None:
    try:
        from IPython.display import display

        display(obj)
    except Exception:
        print(obj)


def _display_table(title: str, df: pd.DataFrame) -> None:
    if df.empty:
        return
    print(title)
    _display(df)


def _progress_line(message: str) -> None:
    print(message)


def _print_forecast_run_summary(simulation: ForecastAlarmSimulation) -> None:
    if simulation.weather_window.empty:
        return
    start_time = simulation.weather_window.iloc[0]["timestamp"]
    end_time = simulation.weather_window.iloc[-1]["timestamp"]
    print(
        "Run summary: "
        f"{len(simulation.frames)} forecast steps from "
        f"{start_time.strftime('%Y-%m-%d %H:%M UTC')} to "
        f"{end_time.strftime('%Y-%m-%d %H:%M UTC')}."
    )
    print(
        "Under the hood: "
        "public weather forecast -> terrain elevation grid -> plume snapshots -> "
        "reverse-geocoded hotspots -> ranked alert tables."
    )
    if not simulation.top_locations.empty:
        top_band = simulation.top_locations.iloc[0]["PeakBand"]
        print(
            f"Top hotspot labels found: {len(simulation.top_locations)} "
            f"(highest simulated band: {top_band})."
        )


def run_demo_experiment(
    cfg: PlumeConfig = DEFAULT_CONFIG,
    neighborhoods: dict[str, tuple[float, float]] = FICTIONAL_NEIGHBORHOODS,
    scenarios: list[dict[str, float | str]] = DEFAULT_SCENARIOS,
    create_animation: bool = False,
) -> DemoResults:
    plot_plume(cfg, neighborhoods=neighborhoods, title_suffix="(Base Scenario)")
    base_df = sample_neighborhoods(cfg, neighborhoods)
    _display(base_df)

    for scenario_name, scenario_cfg in scenario_configs(cfg, scenarios):
        plot_plume(
            scenario_cfg,
            neighborhoods=neighborhoods,
            title_suffix=f"({scenario_name})",
        )

    scenario_df = scenario_summary(cfg, scenarios, neighborhoods)
    concentration_pivot, band_pivot = scenario_pivots(scenario_df)
    _display(scenario_df.head(20))
    _display(concentration_pivot)
    _display(band_pivot)

    plot_puff(cfg, t_s=15 * 60, neighborhoods=neighborhoods)
    animation = animate_puff(cfg, neighborhoods=neighborhoods) if create_animation else None
    if animation is not None:
        _display(animation)

    print(MODEL_NOTES)
    return DemoResults(
        cfg=cfg,
        neighborhoods=neighborhoods,
        scenarios=scenarios,
        base_df=base_df,
        scenario_df=scenario_df,
        concentration_pivot=concentration_pivot,
        band_pivot=band_pivot,
        animation=animation,
    )


def run_interactive_demo(
    cfg: PlumeConfig = DEFAULT_CONFIG,
    neighborhoods: dict[str, tuple[float, float]] = FICTIONAL_NEIGHBORHOODS,
):
    return interactive_plume_demo(cfg, neighborhoods)


def run_alarm_demo(
    source_x_km: float = 0.0,
    source_y_km: float = 0.0,
    severity: str | float = "severe",
    incident_type: str = "leakage",
    name: str = "Simulated Hazard Alarm Demo",
    base_cfg: PlumeConfig = DEFAULT_CONFIG,
    neighborhoods: dict[str, tuple[float, float]] = FICTIONAL_NEIGHBORHOODS,
    wind_from_deg: float | None = None,
    wind_speed_mps: float | None = None,
    stability_class: str | None = None,
    release_height_m: float | None = None,
) -> AlertPlan:
    incident = HazardIncident(
        name=name,
        incident_type=incident_type,
        source_x_km=source_x_km,
        source_y_km=source_y_km,
        severity=severity,
        wind_from_deg=wind_from_deg,
        wind_speed_mps=wind_speed_mps,
        stability_class=stability_class,
        release_height_m=release_height_m,
    )
    alert_plan = analyze_incident_alert(
        incident=incident,
        base_cfg=base_cfg,
        neighborhoods=neighborhoods,
    )
    plot_alert_overlay(
        cfg=alert_plan.cfg,
        neighborhood_report=alert_plan.neighborhood_report,
        incident_name=incident.name,
    )
    if not alert_plan.impacted_neighborhoods.empty:
        _display(
            alert_plan.impacted_neighborhoods[
                ["Neighborhood", "Band", "Concentration", "RecommendedAction"]
            ]
        )
    if alert_plan.notice_payloads:
        _display(pd.DataFrame(alert_plan.notice_payloads))
    else:
        print("No neighborhoods crossed the simulated broadcast threshold.")
    print(MODEL_NOTES)
    return alert_plan


def run_forecast_alarm_demo(
    source_latitude: float,
    source_longitude: float,
    incident_time: str | None = None,
    severity: str | float = "severe",
    incident_type: str = "leakage",
    name: str = "Forecast Hazard Alarm Demo",
    neighborhoods: dict[str, tuple[float, float]] | None = None,
    duration_hours: int = 12,
    forecast_hours: int = 48,
    simulation_resolution: int = 25,
    stability_class: str | None = None,
    release_height_m: float | None = None,
    show_animation: bool = True,
    verbose: bool = False,
) -> ForecastAlarmSimulation:
    if verbose:
        print(
            "Starting forecast-driven hazard demo.\n"
            f"Inputs: location=({source_latitude:.4f}, {source_longitude:.4f}), "
            f"incident_time={incident_time}, severity={severity}, incident_type={incident_type}."
        )
    simulation = simulate_forecast_alarm(
        source_latitude=source_latitude,
        source_longitude=source_longitude,
        incident_time=incident_time,
        severity=severity,
        incident_type=incident_type,
        name=name,
        neighborhoods=neighborhoods,
        base_cfg=DEFAULT_CONFIG,
        duration_hours=duration_hours,
        forecast_hours=forecast_hours,
        simulation_resolution=simulation_resolution,
        stability_class=stability_class,
        release_height_m=release_height_m,
        progress_callback=_progress_line if verbose else None,
    )
    if verbose:
        _print_forecast_run_summary(simulation)
    peak_index = 0
    if simulation.frames:
        peak_index = max(
            range(len(simulation.frames)),
            key=lambda idx: float(np.max(simulation.frames[idx].concentration)),
        )
    plot_forecast_alarm_overlay(simulation, frame_index=peak_index)
    animation = None
    if show_animation and simulation.frames:
        animation = animate_forecast_alarm(simulation)
        simulation.animation = animation
        _display(animation)

    _display_table(
        "Top impacted neighborhoods",
        simulation.top_neighborhoods.reindex(
            columns=[
                "Neighborhood",
                "City",
                "PostalCode",
                "PeakBand",
                "PeakConcentration",
                "FirstImpactedTime",
                "RecommendedAction",
            ]
        )
        if not simulation.top_neighborhoods.empty
        else pd.DataFrame(),
    )
    _display_table(
        "Top impacted cities",
        simulation.top_cities.reindex(
            columns=[
                "City",
                "State",
                "Country",
                "PeakBand",
                "PeakConcentration",
                "FirstImpactedTime",
                "HotspotCount",
                "RecommendedAction",
            ]
        )
        if not simulation.top_cities.empty
        else pd.DataFrame(),
    )
    _display_table(
        "Top impacted postal / ZIP codes",
        simulation.top_postal_codes.reindex(
            columns=[
                "PostalCode",
                "City",
                "State",
                "PeakBand",
                "PeakConcentration",
                "FirstImpactedTime",
                "HotspotCount",
                "RecommendedAction",
            ]
        )
        if not simulation.top_postal_codes.empty
        else pd.DataFrame(),
    )
    _display_table(
        "Top impacted locations",
        simulation.top_locations.reindex(
            columns=[
                "LocationLabel",
                "Latitude",
                "Longitude",
                "PeakBand",
                "PeakConcentration",
                "PeakTime",
                "FirstImpactedTime",
            ]
        )
        if not simulation.top_locations.empty
        else pd.DataFrame(),
    )

    if simulation.notice_payloads:
        _display_table("Draft notice payloads", pd.DataFrame(simulation.notice_payloads))
    else:
        print("No neighborhoods, cities, or postal codes crossed the forecast-driven alert threshold.")

    print(MODEL_NOTES)
    return simulation
