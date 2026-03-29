from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .config import ALERT_COLOR_BY_BAND, PlumeConfig
from .core import (
    compute_concentration,
    compute_puff_concentration,
    wind_to_math_angle_rad,
)

if TYPE_CHECKING:
    from .forecast_alarm import ForecastAlarmSimulation


def _annotate_source(ax, cfg: PlumeConfig, label: str = "Source") -> None:
    ax.scatter([cfg.source_x_km], [cfg.source_y_km], marker="*", s=240)
    ax.text(cfg.source_x_km + 0.3, cfg.source_y_km + 0.3, label, fontsize=10)


def _annotate_neighborhoods(
    ax,
    neighborhoods: dict[str, tuple[float, float]] | None,
) -> None:
    if neighborhoods is None:
        return
    for name, (x_km, y_km) in neighborhoods.items():
        ax.scatter([x_km], [y_km], s=60)
        ax.text(x_km + 0.2, y_km + 0.2, name, fontsize=9)


def plot_plume(
    cfg: PlumeConfig,
    neighborhoods: dict[str, tuple[float, float]] | None = None,
    log_scale: bool = True,
    title_suffix: str = "",
):
    x_km, y_km, concentration = compute_concentration(cfg)
    fig, ax = plt.subplots(figsize=(11, 8))
    z_values = concentration
    colorbar_label = "concentration"
    if log_scale:
        z_values = np.log10(concentration + 1e-6)
        colorbar_label = "log10(concentration + 1e-6)"

    contour_fill = ax.contourf(x_km, y_km, z_values, levels=30)
    colorbar = plt.colorbar(contour_fill, ax=ax)
    colorbar.set_label(colorbar_label)

    contour_levels = [cfg.threshold_low, cfg.threshold_medium, cfg.threshold_high]
    contour_set = ax.contour(
        x_km,
        y_km,
        concentration,
        levels=contour_levels,
        colors="black",
        linewidths=1.5,
        linestyles=["dashed", "solid", "solid"],
    )
    ax.clabel(
        contour_set,
        inline=True,
        fontsize=9,
        fmt={
            cfg.threshold_low: "Low",
            cfg.threshold_medium: "Medium",
            cfg.threshold_high: "High",
        },
    )

    _annotate_source(ax, cfg, label="Source")
    theta = wind_to_math_angle_rad(cfg.wind_from_deg)
    arrow_len = 5.0
    ax.arrow(
        cfg.source_x_km,
        cfg.source_y_km,
        arrow_len * np.cos(theta),
        arrow_len * np.sin(theta),
        width=0.08,
        length_includes_head=True,
    )
    ax.text(
        cfg.source_x_km + arrow_len * np.cos(theta) + 0.2,
        cfg.source_y_km + arrow_len * np.sin(theta) + 0.2,
        "Plume direction",
        fontsize=10,
    )
    _annotate_neighborhoods(ax, neighborhoods)

    ax.set_title(
        "Generic Toxic Plume Simulator "
        f"{title_suffix}\n"
        f"Wind from {cfg.wind_from_deg:.0f}°, "
        f"speed={cfg.wind_speed_mps:.1f} m/s, "
        f"stability={cfg.stability_class}, "
        f"emission={cfg.emission_rate:.0f}"
    )
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_xlim(cfg.x_min_km, cfg.x_max_km)
    ax.set_ylim(cfg.y_min_km, cfg.y_max_km)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    plt.show()
    return fig, ax, concentration


def plot_puff(
    cfg: PlumeConfig,
    t_s: float,
    neighborhoods: dict[str, tuple[float, float]] | None = None,
):
    x_km, y_km, concentration = compute_puff_concentration(cfg, t_s=t_s)
    fig, ax = plt.subplots(figsize=(11, 8))
    z_values = np.log10(concentration + 1e-6)
    contour_fill = ax.contourf(x_km, y_km, z_values, levels=30)
    colorbar = plt.colorbar(contour_fill, ax=ax)
    colorbar.set_label("log10(concentration + 1e-6)")
    _annotate_source(ax, cfg)
    _annotate_neighborhoods(ax, neighborhoods)
    ax.set_title(
        "Generic Puff Release (Educational) "
        f"at t={t_s / 60:.1f} min\n"
        f"Wind from {cfg.wind_from_deg:.0f}°, "
        f"speed={cfg.wind_speed_mps:.1f} m/s, "
        f"stability={cfg.stability_class}"
    )
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_xlim(cfg.x_min_km, cfg.x_max_km)
    ax.set_ylim(cfg.y_min_km, cfg.y_max_km)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    plt.show()
    return fig, ax, concentration


def animate_puff(
    cfg: PlumeConfig,
    neighborhoods: dict[str, tuple[float, float]] | None = None,
    t_end_s: float = 60 * 60,
    frames: int = 40,
):
    x_km, y_km, _ = compute_puff_concentration(cfg, t_s=60)
    times = np.linspace(60, t_end_s, frames)
    fig, ax = plt.subplots(figsize=(11, 8))
    concentration0 = compute_puff_concentration(cfg, t_s=times[0])[2]
    contour_holder = [ax.contourf(x_km, y_km, np.log10(concentration0 + 1e-6), levels=30)]
    plt.colorbar(contour_holder[0], ax=ax, label="log10(concentration + 1e-6)")
    _annotate_source(ax, cfg)
    _annotate_neighborhoods(ax, neighborhoods)
    ax.set_xlim(cfg.x_min_km, cfg.x_max_km)
    ax.set_ylim(cfg.y_min_km, cfg.y_max_km)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    def update(frame_idx: int):
        for collection in contour_holder[0].collections:
            collection.remove()
        t_s = times[frame_idx]
        concentration = compute_puff_concentration(cfg, t_s=t_s)[2]
        contour_holder[0] = ax.contourf(
            x_km,
            y_km,
            np.log10(concentration + 1e-6),
            levels=30,
        )
        ax.set_title(
            "Generic Puff Animation (Educational) | "
            f"t={t_s / 60:.1f} min\n"
            f"Wind from {cfg.wind_from_deg:.0f}°, "
            f"speed={cfg.wind_speed_mps:.1f} m/s, "
            f"stability={cfg.stability_class}"
        )
        return contour_holder[0].collections

    anim = FuncAnimation(fig, update, frames=len(times), interval=200, blit=False)
    plt.close(fig)
    return anim


def plot_alert_overlay(
    cfg: PlumeConfig,
    neighborhood_report,
    incident_name: str = "Simulated Hazard Incident",
):
    x_km, y_km, concentration = compute_concentration(cfg)
    fig, ax = plt.subplots(figsize=(11, 8))
    z_values = np.log10(concentration + 1e-6)
    contour_fill = ax.contourf(x_km, y_km, z_values, levels=30, cmap="viridis")
    colorbar = plt.colorbar(contour_fill, ax=ax)
    colorbar.set_label("log10(concentration + 1e-6)")

    _annotate_source(ax, cfg, label="Incident source")
    theta = wind_to_math_angle_rad(cfg.wind_from_deg)
    arrow_len = 5.0
    ax.arrow(
        cfg.source_x_km,
        cfg.source_y_km,
        arrow_len * np.cos(theta),
        arrow_len * np.sin(theta),
        width=0.08,
        length_includes_head=True,
        color="white",
    )
    ax.text(
        cfg.source_x_km + arrow_len * np.cos(theta) + 0.2,
        cfg.source_y_km + arrow_len * np.sin(theta) + 0.2,
        "Projected plume path",
        fontsize=10,
        color="white",
    )

    for row in neighborhood_report.to_dict(orient="records"):
        band = row["Band"]
        color = row.get("Color", ALERT_COLOR_BY_BAND.get(band, "#94a3b8"))
        marker_size = 150 if row.get("BroadcastRecommended", False) else 80
        ax.scatter(
            [row["X_km"]],
            [row["Y_km"]],
            s=marker_size,
            c=color,
            edgecolors="black",
            linewidths=1.0,
            zorder=5,
        )
        if band != "MINIMAL":
            ax.text(
                row["X_km"] + 0.25,
                row["Y_km"] + 0.25,
                f"{row['Neighborhood']} ({band})",
                fontsize=9,
                zorder=6,
            )

    legend_handles = [
        Patch(facecolor=color, edgecolor="black", label=band.title())
        for band, color in ALERT_COLOR_BY_BAND.items()
    ]
    legend_handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="none",
            markeredgecolor="black",
            markersize=10,
            label="Broadcast recommended",
        )
    )
    ax.legend(handles=legend_handles, loc="upper right", title="Neighborhood impact")

    ax.set_title(
        f"{incident_name}\n"
        f"Simulated neighborhood impact overlay | "
        f"Wind from {cfg.wind_from_deg:.0f}°, "
        f"speed={cfg.wind_speed_mps:.1f} m/s, "
        f"stability={cfg.stability_class}"
    )
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_xlim(cfg.x_min_km, cfg.x_max_km)
    ax.set_ylim(cfg.y_min_km, cfg.y_max_km)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    plt.show()
    return fig, ax


def _location_text(row) -> str:
    for key in ("LocationLabel", "Neighborhood", "City", "PostalCode"):
        value = row.get(key)
        if value:
            return str(value)
    return "Impacted area"


def _location_band(row) -> str:
    return str(row.get("PeakBand", row.get("Band", "MINIMAL")))


def _plot_geo_impact_points(ax, location_report) -> list:
    text_handles = []
    if location_report is None or getattr(location_report, "empty", False):
        return text_handles
    for row in location_report.to_dict(orient="records"):
        band = _location_band(row)
        color = ALERT_COLOR_BY_BAND.get(band, "#94a3b8")
        marker_size = 150 if row.get("BroadcastRecommended", False) else 70
        ax.scatter(
            [row["Longitude"]],
            [row["Latitude"]],
            s=marker_size,
            c=color,
            edgecolors="black",
            linewidths=1.0,
            zorder=6,
        )
        if row.get("BroadcastRecommended", False):
            text_handles.append(
                ax.text(
                    row["Longitude"] + 0.01,
                    row["Latitude"] + 0.01,
                    f"{_location_text(row)} ({band})",
                    fontsize=8,
                    zorder=7,
                )
            )
    return text_handles


def plot_forecast_alarm_overlay(
    simulation: "ForecastAlarmSimulation",
    frame_index: int | None = None,
):
    if not simulation.frames:
        raise ValueError("simulation does not contain any forecast frames")
    frame_index = frame_index if frame_index is not None else 0
    frame = simulation.frames[frame_index]

    fig, ax = plt.subplots(figsize=(11, 8))
    terrain = simulation.terrain
    terrain_fill = ax.contourf(
        terrain.longitude_grid,
        terrain.latitude_grid,
        terrain.elevation_m,
        levels=18,
        cmap="terrain",
        alpha=0.8,
    )
    plt.colorbar(terrain_fill, ax=ax, label="terrain elevation (m)")
    plume_fill = ax.contourf(
        terrain.longitude_grid,
        terrain.latitude_grid,
        np.log10(frame.concentration + 1e-9),
        levels=20,
        cmap="magma",
        alpha=0.45,
    )
    plt.colorbar(plume_fill, ax=ax, label="log10(simulated concentration + 1e-9)")

    ax.scatter(
        [simulation.source_longitude],
        [simulation.source_latitude],
        marker="*",
        s=240,
        c="cyan",
        edgecolors="black",
        linewidths=1.0,
        zorder=8,
    )
    ax.text(
        simulation.source_longitude + 0.01,
        simulation.source_latitude + 0.01,
        "Incident source",
        fontsize=9,
        zorder=9,
    )

    overlay_locations = (
        simulation.top_locations.head(6)
        if getattr(simulation, "top_locations", None) is not None and not simulation.top_locations.empty
        else frame.neighborhood_report
    )
    _plot_geo_impact_points(ax, overlay_locations)
    legend_handles = [
        Patch(facecolor=color, edgecolor="black", label=band.title())
        for band, color in ALERT_COLOR_BY_BAND.items()
    ]
    ax.legend(handles=legend_handles, loc="upper right", title="Alert band")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        f"{simulation.incident.name}\n"
        f"Forecast overlay at {frame.timestamp.strftime('%Y-%m-%d %H:%M UTC')} | "
        f"wind from {frame.weather['wind_from_deg']:.0f}°, "
        f"speed={frame.weather['wind_speed_mps']:.1f} m/s"
    )
    ax.grid(True, alpha=0.25)
    plt.show()
    return fig, ax


def animate_forecast_alarm(simulation: "ForecastAlarmSimulation"):
    if not simulation.frames:
        raise ValueError("simulation does not contain any forecast frames")

    terrain = simulation.terrain
    fig, ax = plt.subplots(figsize=(11, 8))
    terrain_fill = ax.contourf(
        terrain.longitude_grid,
        terrain.latitude_grid,
        terrain.elevation_m,
        levels=18,
        cmap="terrain",
        alpha=0.8,
    )
    plt.colorbar(terrain_fill, ax=ax, label="terrain elevation (m)")
    ax.scatter(
        [simulation.source_longitude],
        [simulation.source_latitude],
        marker="*",
        s=240,
        c="cyan",
        edgecolors="black",
        linewidths=1.0,
        zorder=8,
    )
    ax.text(
        simulation.source_longitude + 0.01,
        simulation.source_latitude + 0.01,
        "Incident source",
        fontsize=9,
        zorder=9,
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.25)

    plume_holder = [
        ax.contourf(
            terrain.longitude_grid,
            terrain.latitude_grid,
            np.log10(simulation.frames[0].concentration + 1e-9),
            levels=20,
            cmap="magma",
            alpha=0.45,
        )
    ]
    plt.colorbar(plume_holder[0], ax=ax, label="log10(simulated concentration + 1e-9)")
    overlay_locations = (
        simulation.top_locations.head(6)
        if getattr(simulation, "top_locations", None) is not None and not simulation.top_locations.empty
        else simulation.frames[0].neighborhood_report
    )
    text_handles = _plot_geo_impact_points(ax, overlay_locations)

    def update(frame_idx: int):
        for collection in plume_holder[0].collections:
            collection.remove()
        for text_handle in text_handles[:]:
            text_handle.remove()
            text_handles.remove(text_handle)
        frame = simulation.frames[frame_idx]
        plume_holder[0] = ax.contourf(
            terrain.longitude_grid,
            terrain.latitude_grid,
            np.log10(frame.concentration + 1e-9),
            levels=20,
            cmap="magma",
            alpha=0.45,
        )
        overlay_locations = (
            simulation.top_locations.head(6)
            if getattr(simulation, "top_locations", None) is not None and not simulation.top_locations.empty
            else frame.neighborhood_report
        )
        text_handles.extend(_plot_geo_impact_points(ax, overlay_locations))
        ax.set_title(
            f"{simulation.incident.name}\n"
            f"Forecast spread at {frame.timestamp.strftime('%Y-%m-%d %H:%M UTC')} | "
            f"wind from {frame.weather['wind_from_deg']:.0f}°, "
            f"speed={frame.weather['wind_speed_mps']:.1f} m/s"
        )
        return plume_holder[0].collections + text_handles

    anim = FuncAnimation(fig, update, frames=len(simulation.frames), interval=500, blit=False)
    plt.close(fig)
    return anim
