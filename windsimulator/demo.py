from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation

from .analysis import sample_neighborhoods, scenario_configs, scenario_pivots, scenario_summary
from .config import (
    DEFAULT_CONFIG,
    DEFAULT_FORECAST_FRAME_INTERVAL_MINUTES,
    DEFAULT_FORECAST_SIMULATION_RESOLUTION,
    DEFAULT_SCENARIOS,
    DEFAULT_TERRAIN_SURFACE_RESOLUTION,
    FICTIONAL_NEIGHBORHOODS,
    MODEL_NOTES,
    PlumeConfig,
)
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
        from IPython.display import HTML, display

        if isinstance(obj, FuncAnimation):
            try:
                display(HTML(obj.to_jshtml()))
                return
            except Exception:
                pass
        display(obj)
    except Exception:
        print(obj)


def _display_table(title: str, df: pd.DataFrame) -> None:
    if df.empty:
        return
    print(title)
    _display(df)


def _notice_table_for_display(simulation: ForecastAlarmSimulation) -> pd.DataFrame:
    neighborhood_table = getattr(simulation, "top_neighborhoods", pd.DataFrame()).copy()
    if neighborhood_table.empty:
        neighborhood_table = getattr(simulation, "impacted_neighborhoods", pd.DataFrame()).copy()
    if neighborhood_table.empty:
        return pd.DataFrame(
            columns=[
                "Neighborhood",
                "City",
                "NoticeType",
                "PeakBand",
                "FirstImpactedTime",
                "RecommendedAction",
                "Priority",
            ]
        )

    if "BroadcastRecommended" in neighborhood_table.columns:
        neighborhood_table = neighborhood_table[neighborhood_table["BroadcastRecommended"]].copy()
    if neighborhood_table.empty:
        return pd.DataFrame(
            columns=[
                "Neighborhood",
                "City",
                "NoticeType",
                "PeakBand",
                "FirstImpactedTime",
                "RecommendedAction",
                "Priority",
            ]
        )

    if "PeakBand" not in neighborhood_table.columns and "Band" in neighborhood_table.columns:
        neighborhood_table["PeakBand"] = neighborhood_table["Band"]
    if "City" not in neighborhood_table.columns:
        neighborhood_table["City"] = ""

    display_table = neighborhood_table.reindex(
        columns=[
            "Neighborhood",
            "City",
            "NoticeLevel",
            "PeakBand",
            "FirstImpactedTime",
            "RecommendedAction",
            "BroadcastPriorityRank",
        ]
    ).rename(
        columns={
            "NoticeLevel": "NoticeType",
            "BroadcastPriorityRank": "Priority",
        }
    )
    return display_table.reset_index(drop=True)


def _progress_line(message: str) -> None:
    print(message, flush=True)


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
        "Resolved duration: "
        f"{simulation.resolved_duration_hours:.1f} h "
        f"({simulation.duration_basis})."
    )
    print(
        "Under the hood: "
        "public weather forecast -> terrain elevation grid -> source-term-aware plume snapshots -> "
        "uncertainty envelopes -> action polygons -> ranked alert tables."
    )
    print(
        "Resolved source term: "
        f"{simulation.source_term.profile_label} | "
        f"{simulation.source_term.emission_rate:.0f} mass/s for "
        f"{simulation.source_term.release_duration_minutes:.0f} min with "
        f"{simulation.source_term.initial_pulse_mass:.0f} initial pulse mass."
    )
    print(
        "Airborne persistence: "
        f"{simulation.material_fate.profile_label} | "
        f"reactive half-life "
        f"{simulation.material_fate.reactive_airborne_half_life_minutes if simulation.material_fate.reactive_airborne_half_life_minutes is not None else 'n/a'} min, "
        f"deposition half-life "
        f"{simulation.material_fate.deposition_half_life_minutes if simulation.material_fate.deposition_half_life_minutes is not None else 'n/a'} min."
    )
    if not simulation.uncertainty_summary.empty:
        print(
            f"Uncertainty envelopes: {len(simulation.uncertainty_summary)} "
            "scenario layers (likely, conservative, worst reasonable)."
        )
    if not simulation.action_polygons.empty:
        print(
            f"Action polygons generated: {len(simulation.action_polygons)} "
            "layered areas with GeoJSON-ready geometry."
        )
    if not simulation.top_locations.empty:
        top_band = simulation.top_locations.iloc[0]["PeakBand"]
        print(
            f"Top hotspot labels found: {len(simulation.top_locations)} "
            f"(highest simulated band: {top_band})."
        )
        print(
            "Broadcast ranking: NoticeLevel (Immediate Action > Warning > Advisory > Monitor), "
            "then CAPUrgency (Immediate > Expected > Future), then earliest impact time, then peak concentration."
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
                [
                    "BroadcastPriorityRank",
                    "Neighborhood",
                    "NoticeLevel",
                    "CAPUrgency",
                    "Band",
                    "Concentration",
                    "RecommendedAction",
                ]
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
    duration_hours: int | float | str | None = "auto",
    forecast_hours: int = 48,
    simulation_resolution: int = DEFAULT_FORECAST_SIMULATION_RESOLUTION,
    terrain_resolution: int | None = DEFAULT_TERRAIN_SURFACE_RESOLUTION,
    frame_interval_minutes: int = DEFAULT_FORECAST_FRAME_INTERVAL_MINUTES,
    overlay_basemap_styles: tuple[str | None, ...] = ("roadmap",),
    animation_basemap_style: str | None = "roadmap",
    show_overlay: bool = True,
    stability_class: str | None = None,
    release_height_m: float | None = None,
    source_term_profile: str | None = None,
    hazard_material: str | None = None,
    emission_rate_override: float | None = None,
    release_duration_minutes: float | None = None,
    initial_pulse_minutes: float | None = None,
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
        terrain_resolution=terrain_resolution,
        frame_interval_minutes=frame_interval_minutes,
        stability_class=stability_class,
        release_height_m=release_height_m,
        source_term_profile=source_term_profile,
        hazard_material=hazard_material,
        emission_rate_override=emission_rate_override,
        release_duration_minutes=release_duration_minutes,
        initial_pulse_minutes=initial_pulse_minutes,
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
    if show_overlay:
        plot_forecast_alarm_overlay(
            simulation,
            frame_index=peak_index,
            basemap_styles=overlay_basemap_styles,
        )
    animation = None
    if show_animation and simulation.frames:
        animation = animate_forecast_alarm(
            simulation,
            basemap_style=animation_basemap_style,
        )
        simulation.animation = animation
        _display(animation)

    notice_table = _notice_table_for_display(simulation)
    if not notice_table.empty:
        _display_table("Neighborhoods Requiring Notice", notice_table)
    else:
        print("No neighborhoods crossed the forecast-driven notice threshold.")

    return simulation
