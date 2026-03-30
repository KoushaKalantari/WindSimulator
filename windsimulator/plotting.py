from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.path import Path

from .basemaps import BasemapImage, fetch_basemap_image
from .config import ALERT_COLOR_BY_BAND, EMERGENCY_ALERT_THRESHOLDS, PlumeConfig
from .core import (
    compute_concentration,
    compute_puff_concentration,
    wind_to_math_angle_rad,
)
from .geocoding import ReferenceLabel, reference_labels_for_bounds

if TYPE_CHECKING:
    from .forecast_alarm import ForecastAlarmSimulation

try:
    import arabic_reshaper
except Exception:
    arabic_reshaper = None

try:
    from bidi.algorithm import get_display as bidi_get_display
except Exception:
    bidi_get_display = None


TERRAIN_CONTOUR_LEVELS = 14
DEFAULT_FORECAST_BASEMAP_STYLES = ("roadmap",)
DEFAULT_ANIMATION_BASEMAP_STYLE = "roadmap"
RTL_FONT_FAMILIES = [".SF Arabic", "Geeza Pro", "Arial Unicode MS", "DejaVu Sans"]
ACTION_POLYGON_LINESTYLES = {
    "Likely": "solid",
    "Conservative": "--",
    "Worst Reasonable": ":",
}
ACTION_POLYGON_FILL_ALPHA = {
    "Likely": 0.10,
    "Conservative": 0.06,
    "Worst Reasonable": 0.03,
}
HAZARD_BOUNDARY_BANDS = ("LOW", "MEDIUM", "HIGH")
HAZARD_BOUNDARY_LINEWIDTHS = {
    "LOW": 1.6,
    "MEDIUM": 2.0,
    "HIGH": 2.6,
}
PLUME_FILL_ALPHA = 0.48
MAX_MAP_LEGEND_ITEMS = 5


@dataclass(frozen=True)
class ResolvedBasemapPanel:
    basemap: BasemapImage | None
    title: str
    attribution: str
    dedupe_key: str
    warning: str | None = None


@lru_cache(maxsize=1)
def _broadcast_pin_marker() -> Path:
    # Build a simple teardrop marker whose tip sits on the data coordinate.
    angles = np.linspace(np.deg2rad(210.0), np.deg2rad(-30.0), 28)
    head = np.column_stack([0.72 * np.cos(angles), 0.72 * np.sin(angles) + 1.02])
    vertices = np.vstack(
        [
            np.array([[0.0, 0.0]]),
            head,
            np.array([[0.0, 0.0], [0.0, 0.0]]),
        ]
    )
    codes = [Path.MOVETO] + [Path.LINETO] * len(head) + [Path.LINETO, Path.CLOSEPOLY]
    return Path(vertices, codes)


def _plot_impact_marker(
    ax,
    *,
    x: float,
    y: float,
    color: str,
    broadcast_recommended: bool,
    zorder: int,
) -> None:
    if broadcast_recommended:
        ax.scatter(
            [x],
            [y],
            s=320,
            marker=_broadcast_pin_marker(),
            c=color,
            edgecolors="black",
            linewidths=1.1,
            zorder=zorder,
        )
        return

    ax.scatter(
        [x],
        [y],
        s=70,
        c=color,
        edgecolors="black",
        linewidths=1.0,
        zorder=zorder,
    )


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
        _remove_contour_set(contour_holder[0])
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
        return _contour_artists(contour_holder[0])

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
        _plot_impact_marker(
            ax,
            x=float(row["X_km"]),
            y=float(row["Y_km"]),
            color=color,
            broadcast_recommended=bool(row.get("BroadcastRecommended", False)),
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
            marker=_broadcast_pin_marker(),
            color="w",
            markerfacecolor="#64748b",
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
    for key in ("Neighborhood", "LocationLabel", "City", "PostalCode"):
        value = row.get(key)
        if value:
            return str(value)
    return "Impacted area"


def _location_band(row) -> str:
    return str(row.get("PeakBand", row.get("Band", "MINIMAL")))


def _contains_rtl(text: str) -> bool:
    return any(
        ("\u0590" <= char <= "\u08ff") or ("\ufb1d" <= char <= "\ufefc")
        for char in text
    )


def _shape_label_text(text: str) -> str:
    if not text or not _contains_rtl(text):
        return text
    if arabic_reshaper is None or bidi_get_display is None:
        return text
    try:
        return bidi_get_display(arabic_reshaper.reshape(text))
    except Exception:
        return text


def _label_text_style(text: str) -> dict[str, object]:
    if not _contains_rtl(text):
        return {}
    return {
        "fontfamily": RTL_FONT_FAMILIES,
        "ha": "right",
        "multialignment": "right",
    }


def _plot_geo_impact_points(ax, location_report) -> list:
    text_handles = []
    if location_report is None or getattr(location_report, "empty", False):
        return text_handles
    for row in location_report.to_dict(orient="records"):
        band = _location_band(row)
        color = ALERT_COLOR_BY_BAND.get(band, "#94a3b8")
        _plot_impact_marker(
            ax,
            x=float(row["Longitude"]),
            y=float(row["Latitude"]),
            color=color,
            broadcast_recommended=bool(row.get("BroadcastRecommended", False)),
            zorder=6,
        )
        if row.get("BroadcastRecommended", False):
            location_text = _location_text(row)
            label_text = _shape_label_text(location_text)
            text_handles.append(
                ax.text(
                    row["Longitude"] + 0.01,
                    row["Latitude"] + 0.01,
                    label_text,
                    fontsize=8,
                    bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 1.5},
                    zorder=7,
                    **_label_text_style(location_text),
                )
            )
    return text_handles


def _frame_overlay_locations(frame, max_locations: int = 6):
    location_report = getattr(frame, "neighborhood_report", None)
    if location_report is None or getattr(location_report, "empty", False):
        return location_report

    filtered = location_report
    if "BroadcastRecommended" in filtered.columns:
        impacted = filtered[filtered["BroadcastRecommended"]]
        if not impacted.empty:
            filtered = impacted
    elif "Band" in filtered.columns:
        impacted = filtered[filtered["Band"] != "MINIMAL"]
        if not impacted.empty:
            filtered = impacted
    if "Concentration" in filtered.columns:
        filtered = filtered.sort_values("Concentration", ascending=False)
    return filtered.head(max_locations).reset_index(drop=True)


def _simulation_overlay_neighborhoods(simulation, frame=None, max_locations: int = 6):
    impacted = getattr(simulation, "impacted_neighborhoods", None)
    if impacted is not None and not getattr(impacted, "empty", False):
        filtered = impacted.copy()
        if "PeakConcentration" in filtered.columns:
            filtered = filtered.sort_values("PeakConcentration", ascending=False)
        return filtered.head(max_locations).reset_index(drop=True)

    summary = getattr(simulation, "neighborhood_summary", None)
    if summary is not None and not getattr(summary, "empty", False):
        filtered = summary.copy()
        if "BroadcastRecommended" in filtered.columns:
            broadcast = filtered[filtered["BroadcastRecommended"]]
            if not broadcast.empty:
                filtered = broadcast
        if "PeakConcentration" in filtered.columns:
            filtered = filtered.sort_values("PeakConcentration", ascending=False)
        return filtered.head(max_locations).reset_index(drop=True)

    if frame is not None:
        return _frame_overlay_locations(frame, max_locations=max_locations)
    return pd.DataFrame()


def _reference_focus_points(simulation, frame=None, max_points: int = 10) -> tuple[tuple[float, float], ...]:
    points: list[tuple[float, float]] = [
        (float(simulation.source_latitude), float(simulation.source_longitude)),
    ]
    overlay_locations = _simulation_overlay_neighborhoods(simulation, frame=frame, max_locations=max_points)
    if overlay_locations is not None and not getattr(overlay_locations, "empty", False):
        for row in overlay_locations.to_dict(orient="records"):
            latitude = row.get("Latitude")
            longitude = row.get("Longitude")
            if latitude is None or longitude is None:
                continue
            points.append((float(latitude), float(longitude)))
    top_locations = getattr(simulation, "top_locations", None)
    if top_locations is not None and not getattr(top_locations, "empty", False):
        sorted_top_locations = top_locations.copy()
        sort_columns: list[str] = []
        ascending: list[bool] = []
        if "PeakConcentration" in sorted_top_locations.columns:
            sort_columns.append("PeakConcentration")
            ascending.append(False)
        elif "Concentration" in sorted_top_locations.columns:
            sort_columns.append("Concentration")
            ascending.append(False)
        if sort_columns:
            sorted_top_locations = sorted_top_locations.sort_values(sort_columns, ascending=ascending)
        for row in sorted_top_locations.head(max_points).to_dict(orient="records"):
            latitude = row.get("Latitude")
            longitude = row.get("Longitude")
            if latitude is None or longitude is None:
                continue
            points.append((float(latitude), float(longitude)))
    deduped: list[tuple[float, float]] = []
    for latitude, longitude in points:
        if any(abs(latitude - existing_lat) < 0.003 and abs(longitude - existing_lon) < 0.003 for existing_lat, existing_lon in deduped):
            continue
        deduped.append((latitude, longitude))
    return tuple(deduped[:max_points])


def _normalized_place_label(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().lower().split())


def _label_within_bounds(latitude: object, longitude: object, bounds) -> bool:
    try:
        latitude_value = float(latitude)
        longitude_value = float(longitude)
    except (TypeError, ValueError):
        return False
    return (
        float(bounds[1]) <= latitude_value <= float(bounds[3])
        and float(bounds[0]) <= longitude_value <= float(bounds[2])
    )


def _sorted_reference_rows(rows: pd.DataFrame | None) -> pd.DataFrame:
    if rows is None or getattr(rows, "empty", False):
        return pd.DataFrame()
    sorted_rows = rows.copy()
    sort_columns: list[str] = []
    ascending: list[bool] = []
    if "BroadcastRecommended" in sorted_rows.columns:
        sort_columns.append("BroadcastRecommended")
        ascending.append(False)
    if "BroadcastPriorityRank" in sorted_rows.columns:
        sort_columns.append("BroadcastPriorityRank")
        ascending.append(True)
    if "HotspotCount" in sorted_rows.columns:
        sort_columns.append("HotspotCount")
        ascending.append(False)
    if "PeakConcentration" in sorted_rows.columns:
        sort_columns.append("PeakConcentration")
        ascending.append(False)
    elif "Concentration" in sorted_rows.columns:
        sort_columns.append("Concentration")
        ascending.append(False)
    if sort_columns:
        sorted_rows = sorted_rows.sort_values(sort_columns, ascending=ascending)
    return sorted_rows


def _reference_labels_from_rows(
    rows: pd.DataFrame | None,
    *,
    label_column: str,
    kind: str,
    bounds,
    max_labels: int,
    seen_labels: set[str],
    excluded_labels: set[str],
) -> list[ReferenceLabel]:
    labels: list[ReferenceLabel] = []
    if rows is None or getattr(rows, "empty", False) or label_column not in rows.columns:
        return labels
    sorted_rows = _sorted_reference_rows(rows)
    for row in sorted_rows.to_dict(orient="records"):
        label_value = row.get(label_column)
        label_key = _normalized_place_label(label_value)
        if not label_key or label_key in seen_labels or label_key in excluded_labels:
            continue
        latitude = row.get("Latitude")
        longitude = row.get("Longitude")
        if not _label_within_bounds(latitude, longitude, bounds):
            continue
        labels.append(
            ReferenceLabel(
                label=str(label_value),
                latitude=float(latitude),
                longitude=float(longitude),
                kind=kind,
            )
        )
        seen_labels.add(label_key)
        if len(labels) >= max_labels:
            break
    return labels


def _simulation_reference_labels(simulation, *, bounds, frame=None, excluded_labels: set[str]) -> list[ReferenceLabel]:
    seen_labels = set(excluded_labels)
    labels: list[ReferenceLabel] = []
    top_locations = getattr(simulation, "top_locations", None)
    neighborhood_summary = getattr(simulation, "neighborhood_summary", None)
    frame_locations = getattr(frame, "neighborhood_report", None) if frame is not None else None

    labels.extend(
        _reference_labels_from_rows(
            top_locations,
            label_column="City",
            kind="city",
            bounds=bounds,
            max_labels=8,
            seen_labels=seen_labels,
            excluded_labels=excluded_labels,
        )
    )
    labels.extend(
        _reference_labels_from_rows(
            neighborhood_summary,
            label_column="Neighborhood",
            kind="neighborhood",
            bounds=bounds,
            max_labels=10,
            seen_labels=seen_labels,
            excluded_labels=excluded_labels,
        )
    )
    labels.extend(
        _reference_labels_from_rows(
            top_locations,
            label_column="Neighborhood",
            kind="neighborhood",
            bounds=bounds,
            max_labels=8,
            seen_labels=seen_labels,
            excluded_labels=excluded_labels,
        )
    )
    labels.extend(
        _reference_labels_from_rows(
            frame_locations,
            label_column="Neighborhood",
            kind="neighborhood",
            bounds=bounds,
            max_labels=6,
            seen_labels=seen_labels,
            excluded_labels=excluded_labels,
        )
    )
    return labels


def _reference_labels_for_map(simulation, *, bounds, frame=None) -> list[ReferenceLabel]:
    latitude_min = float(bounds[1])
    latitude_max = float(bounds[3])
    longitude_min = float(bounds[0])
    longitude_max = float(bounds[2])
    focus_points = _reference_focus_points(simulation, frame=frame)
    overlay_locations = _simulation_overlay_neighborhoods(simulation, frame=frame, max_locations=6)
    excluded_labels = {
        _normalized_place_label(_location_text(row))
        for row in overlay_locations.to_dict(orient="records")
    } if overlay_locations is not None and not getattr(overlay_locations, "empty", False) else set()

    labels = _simulation_reference_labels(
        simulation,
        bounds=bounds,
        frame=frame,
        excluded_labels=excluded_labels,
    )
    seen_labels = {_normalized_place_label(label.label) for label in labels}
    seen_labels.update(excluded_labels)

    fallback_labels = reference_labels_for_bounds(
        latitude_min=latitude_min,
        longitude_min=longitude_min,
        latitude_max=latitude_max,
        longitude_max=longitude_max,
        focus_points=focus_points,
        max_city_labels=8,
        max_neighborhood_labels=10,
        max_road_labels=8,
    )
    for label in fallback_labels:
        label_key = _normalized_place_label(label.label)
        if not label_key or label_key in seen_labels or label_key in excluded_labels:
            continue
        labels.append(label)
        seen_labels.add(label_key)
    return labels


def _plot_reference_labels(ax, reference_labels: list[ReferenceLabel]) -> list[object]:
    artists: list[object] = []
    for label in reference_labels:
        label_text = _shape_label_text(label.label)
        label_style = _label_text_style(label.label)
        if label.kind == "city":
            artists.append(
                ax.text(
                    label.longitude,
                    label.latitude,
                    label_text,
                    fontsize=10.2,
                    fontweight="semibold",
                    color="#0f172a",
                    path_effects=[patheffects.withStroke(linewidth=3.5, foreground="white")],
                    zorder=6,
                    **label_style,
                )
            )
        elif label.kind == "neighborhood":
            artists.append(
                ax.text(
                    label.longitude,
                    label.latitude,
                    label_text,
                    fontsize=8.2,
                    color="#1f2937",
                    path_effects=[patheffects.withStroke(linewidth=3.0, foreground="white")],
                    zorder=5.8,
                    **label_style,
                )
            )
        elif label.kind == "road":
            artists.append(
                ax.text(
                    label.longitude,
                    label.latitude,
                    label_text,
                    fontsize=7.2,
                    fontstyle="italic",
                    color="#475569",
                    path_effects=[patheffects.withStroke(linewidth=2.6, foreground="white")],
                    zorder=5,
                    **label_style,
                )
            )
    return artists


def _append_point(longitudes: list[float], latitudes: list[float], longitude, latitude) -> None:
    try:
        longitude_value = float(longitude)
        latitude_value = float(latitude)
    except (TypeError, ValueError):
        return
    if not (np.isfinite(longitude_value) and np.isfinite(latitude_value)):
        return
    longitudes.append(longitude_value)
    latitudes.append(latitude_value)


def _append_location_report_points(longitudes: list[float], latitudes: list[float], location_report) -> None:
    if location_report is None or getattr(location_report, "empty", False):
        return
    for row in location_report.to_dict(orient="records"):
        _append_point(longitudes, latitudes, row.get("Longitude"), row.get("Latitude"))


def _append_action_polygon_points(longitudes: list[float], latitudes: list[float], action_polygons) -> None:
    if action_polygons is None or getattr(action_polygons, "empty", False):
        return

    polygon_rows = action_polygons
    if "BroadcastRecommended" in polygon_rows.columns:
        impacted_rows = polygon_rows[polygon_rows["BroadcastRecommended"]]
        if not impacted_rows.empty:
            polygon_rows = impacted_rows

    for row in polygon_rows.to_dict(orient="records"):
        _append_point(longitudes, latitudes, row.get("CentroidLongitude"), row.get("CentroidLatitude"))
        coordinates = row.get("GeoJSONCoordinates") or []
        for ring in coordinates:
            for point in ring or []:
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    _append_point(longitudes, latitudes, point[0], point[1])


def _terrain_bounds(
    terrain,
    *,
    location_report=None,
    action_polygons=None,
    source_longitude: float | None = None,
    source_latitude: float | None = None,
) -> tuple[float, float, float, float]:
    longitudes = [
        float(np.nanmin(terrain.longitude_grid)),
        float(np.nanmax(terrain.longitude_grid)),
    ]
    latitudes = [
        float(np.nanmin(terrain.latitude_grid)),
        float(np.nanmax(terrain.latitude_grid)),
    ]
    _append_point(longitudes, latitudes, source_longitude, source_latitude)
    _append_location_report_points(longitudes, latitudes, location_report)
    _append_action_polygon_points(longitudes, latitudes, action_polygons)

    longitude_min = min(longitudes)
    longitude_max = max(longitudes)
    latitude_min = min(latitudes)
    latitude_max = max(latitudes)
    longitude_pad = max(0.01, (longitude_max - longitude_min) * 0.08)
    latitude_pad = max(0.01, (latitude_max - latitude_min) * 0.08)
    return (
        longitude_min - longitude_pad,
        latitude_min - latitude_pad,
        longitude_max + longitude_pad,
        latitude_max + latitude_pad,
    )


def _add_fallback_terrain_background(ax, terrain):
    ax.set_facecolor("#f8fafc")
    levels = _terrain_elevation_levels(terrain.elevation_m)
    ax.contour(
        terrain.longitude_grid,
        terrain.latitude_grid,
        terrain.elevation_m,
        levels=levels,
        colors="#64748b",
        linewidths=0.55,
        alpha=0.45,
        zorder=0,
    )
    return None


def _terrain_elevation_levels(elevation_m: np.ndarray) -> np.ndarray:
    finite = np.asarray(elevation_m, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.linspace(0.0, 1.0, TERRAIN_CONTOUR_LEVELS)
    low = float(np.nanpercentile(finite, 2.0))
    high = float(np.nanpercentile(finite, 98.0))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.nanmin(finite))
        high = float(np.nanmax(finite))
    if high <= low:
        high = low + 1.0
    return np.linspace(low, high, TERRAIN_CONTOUR_LEVELS)


def _plume_log_levels(concentration: np.ndarray) -> np.ndarray:
    finite = np.asarray(concentration, dtype=float)
    finite = finite[np.isfinite(finite) & (finite > 0.0)]
    if finite.size == 0:
        return np.linspace(-9.0, -6.0, 13)

    percentile_floor = float(np.nanpercentile(finite, 5.0))
    percentile_ceiling = float(np.nanpercentile(finite, 99.5))
    reference_floor = min(percentile_floor, EMERGENCY_ALERT_THRESHOLDS["LOW"] * 0.5)
    reference_ceiling = max(percentile_ceiling, EMERGENCY_ALERT_THRESHOLDS["HIGH"])
    log_floor = float(np.floor(np.log10(max(reference_floor, 1e-12)) * 2.0) / 2.0)
    log_ceiling = float(np.ceil(np.log10(max(reference_ceiling, 1e-12)) * 2.0) / 2.0)
    if log_ceiling - log_floor < 2.0:
        log_ceiling = log_floor + 2.0
    return np.linspace(log_floor, log_ceiling, 15)


def _plume_colorbar_ticks(levels: np.ndarray) -> np.ndarray:
    if levels.size <= 1:
        return levels
    log_floor = float(levels[0])
    log_ceiling = float(levels[-1])
    tick_step = 0.5 if (log_ceiling - log_floor) <= 3.5 else 1.0
    start_tick = np.ceil(log_floor / tick_step) * tick_step
    ticks = np.arange(start_tick, log_ceiling + 0.25 * tick_step, tick_step)
    return np.round(ticks, 2)


def _resolve_basemap_panel(
    terrain,
    basemap_style: str | None,
    *,
    bounds: tuple[float, float, float, float] | None = None,
) -> ResolvedBasemapPanel:
    if basemap_style is None:
        return ResolvedBasemapPanel(
            basemap=None,
            title="Terrain Elevation",
            attribution="Terrain elevation surface",
            dedupe_key="terrain-fallback",
            warning="Road/city basemap unavailable; showing terrain contours only.",
        )

    longitude_min, latitude_min, longitude_max, latitude_max = bounds or _terrain_bounds(terrain)
    try:
        basemap = fetch_basemap_image(
            longitude_min=longitude_min,
            latitude_min=latitude_min,
            longitude_max=longitude_max,
            latitude_max=latitude_max,
            style=basemap_style,
        )
    except Exception:
        return ResolvedBasemapPanel(
            basemap=None,
            title=f"{basemap_style.title()} imagery unavailable",
            attribution="Requested basemap unavailable; showing terrain elevation instead",
            dedupe_key="terrain-fallback",
            warning=(
                "Road/city labels could not be loaded from the public basemap source; "
                "showing terrain contours only."
            ),
        )
    return ResolvedBasemapPanel(
        basemap=basemap,
        title=basemap.title,
        attribution=basemap.attribution,
        dedupe_key=basemap.style,
    )


def _resolved_basemap_panels(
    terrain,
    basemap_styles: tuple[str | None, ...],
    *,
    bounds: tuple[float, float, float, float] | None = None,
) -> list[ResolvedBasemapPanel]:
    panels: list[ResolvedBasemapPanel] = []
    for basemap_style in basemap_styles:
        panel = _resolve_basemap_panel(terrain, basemap_style, bounds=bounds)
        if any(existing.dedupe_key == panel.dedupe_key for existing in panels):
            continue
        panels.append(panel)
    return panels or [_resolve_basemap_panel(terrain, None, bounds=bounds)]


def _add_basemap_background(ax, terrain, panel: ResolvedBasemapPanel) -> object | None:
    if panel.basemap is None:
        return _add_fallback_terrain_background(ax, terrain)

    basemap_style = panel.basemap.style
    ax.imshow(
        panel.basemap.image_rgba,
        extent=panel.basemap.extent,
        origin="upper",
        interpolation="nearest" if basemap_style in {"roadmap", "terrain_map"} else "bilinear",
        zorder=0,
    )
    ax.contour(
        terrain.longitude_grid,
        terrain.latitude_grid,
        terrain.elevation_m,
        levels=min(8, TERRAIN_CONTOUR_LEVELS),
        colors="white" if basemap_style == "satellite" else "#334155",
        linewidths=0.55,
        alpha=0.12 if basemap_style in {"roadmap", "terrain_map"} else 0.28,
        zorder=1,
    )
    return None


def _overlay_source_marker(ax, simulation) -> None:
    ax.scatter(
        [simulation.source_longitude],
        [simulation.source_latitude],
        marker="*",
        s=240,
        c="#22d3ee",
        edgecolors="black",
        linewidths=1.0,
        zorder=8,
    )
    ax.text(
        simulation.source_longitude + 0.01,
        simulation.source_latitude + 0.01,
        "Incident source",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#94a3b8", "pad": 1.5},
        zorder=9,
    )


def _overlay_plume(ax, terrain, concentration):
    levels = _plume_log_levels(concentration)
    plume_fill = ax.contourf(
        terrain.longitude_grid,
        terrain.latitude_grid,
        np.log10(concentration + 1e-9),
        levels=levels,
        cmap="inferno",
        alpha=PLUME_FILL_ALPHA,
        zorder=3,
    )
    boundary_levels = []
    boundary_colors = []
    boundary_widths = []
    max_concentration = float(np.nanmax(concentration)) if np.size(concentration) else 0.0
    for band in HAZARD_BOUNDARY_BANDS:
        threshold = EMERGENCY_ALERT_THRESHOLDS[band]
        if max_concentration >= threshold:
            boundary_levels.append(threshold)
            boundary_colors.append(ALERT_COLOR_BY_BAND[band])
            boundary_widths.append(HAZARD_BOUNDARY_LINEWIDTHS[band])
    boundary_contours = None
    if boundary_levels:
        boundary_contours = ax.contour(
            terrain.longitude_grid,
            terrain.latitude_grid,
            concentration,
            levels=boundary_levels,
            colors=boundary_colors,
            linewidths=boundary_widths,
            alpha=0.98,
            zorder=4,
        )
    return plume_fill, boundary_contours


def _plot_action_polygons(ax, action_polygons, max_polygons: int = 8) -> list[object]:
    artists: list[object] = []
    if action_polygons is None or getattr(action_polygons, "empty", False):
        return artists

    for polygon_rank, row in enumerate(
        action_polygons[action_polygons["BroadcastRecommended"]].head(max_polygons).to_dict(orient="records"),
        start=1,
    ):
        coordinates = row.get("GeoJSONCoordinates") or []
        if not coordinates or not coordinates[0]:
            continue
        outer_ring = coordinates[0]
        longitudes = [point[0] for point in outer_ring]
        latitudes = [point[1] for point in outer_ring]
        band = str(row.get("Band", "LOW"))
        scenario_name = str(row.get("ScenarioName", "Likely"))
        color = ALERT_COLOR_BY_BAND.get(band, "#94a3b8")
        fill_alpha = ACTION_POLYGON_FILL_ALPHA.get(scenario_name, 0.04)
        linestyle = ACTION_POLYGON_LINESTYLES.get(scenario_name, "solid")
        artists.extend(
            ax.fill(
                longitudes,
                latitudes,
                color=color,
                alpha=fill_alpha,
                zorder=2,
            )
        )
        line_handle = ax.plot(
            longitudes,
            latitudes,
            color=color,
            linestyle=linestyle,
            linewidth=1.8,
            alpha=0.95,
            zorder=5,
        )
        artists.extend(line_handle)
    return artists


def _bounds_location_report_for_simulation(simulation, frame=None):
    top_locations = getattr(simulation, "top_locations", None)
    if top_locations is not None and not getattr(top_locations, "empty", False):
        return top_locations
    if frame is not None:
        return getattr(frame, "neighborhood_report", None)

    all_reports = []
    for simulation_frame in getattr(simulation, "frames", []):
        report = getattr(simulation_frame, "neighborhood_report", None)
        if report is None or getattr(report, "empty", False):
            continue
        if "Band" in report.columns:
            impacted = report[report["Band"] != "MINIMAL"]
            if not impacted.empty:
                report = impacted
        all_reports.append(report)

    if not all_reports:
        return None
    if len(all_reports) == 1:
        return all_reports[0]
    return pd.concat(all_reports, ignore_index=True)


def _remove_contour_set(contour_set) -> None:
    if contour_set is None:
        return
    if hasattr(contour_set, "remove"):
        contour_set.remove()
        return
    for collection in getattr(contour_set, "collections", []):
        collection.remove()


def _contour_artists(contour_set) -> list:
    if contour_set is None:
        return []
    return list(getattr(contour_set, "collections", []))


def _alert_band_legend_handles(
    include_broadcast: bool = True,
    include_source: bool = True,
) -> list[object]:
    handles: list[object] = []
    for band in HAZARD_BOUNDARY_BANDS:
        handles.append(
            Line2D(
                [0],
                [0],
                color=ALERT_COLOR_BY_BAND[band],
                linewidth=HAZARD_BOUNDARY_LINEWIDTHS[band],
                label=f"{band.title()} boundary",
            )
        )
    if include_source:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="*",
                color="w",
                markerfacecolor="#22d3ee",
                markeredgecolor="black",
                markersize=14,
                linewidth=0.0,
                label="Incident source",
            )
        )
    if include_broadcast:
        handles.append(
            Line2D(
                [0],
                [0],
                marker=_broadcast_pin_marker(),
                color="w",
                markerfacecolor="#64748b",
                markeredgecolor="black",
                markersize=10,
                label="Broadcast recommended",
            )
        )
    return handles[:MAX_MAP_LEGEND_ITEMS]


def _place_alert_band_legend(
    fig,
    *,
    title: str = "Map key",
    include_broadcast: bool = True,
    include_source: bool = True,
) -> None:
    handles = _alert_band_legend_handles(
        include_broadcast=include_broadcast,
        include_source=include_source,
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=min(len(handles), 5),
        title=title,
        frameon=True,
        facecolor="white",
        edgecolor="#cbd5e1",
    )


def _timelapse_interval_ms(frame_count: int, target_duration_s: float = 12.0) -> int:
    if frame_count <= 0:
        return 200
    return int(np.clip((target_duration_s * 1000.0) / frame_count, 80.0, 400.0))


def _format_basemap_axis(
    ax,
    terrain,
    title: str,
    attribution: str | None = None,
    warning: str | None = None,
    *,
    bounds: tuple[float, float, float, float] | None = None,
) -> None:
    longitude_min, latitude_min, longitude_max, latitude_max = bounds or _terrain_bounds(terrain)
    ax.set_xlim(longitude_min, longitude_max)
    ax.set_ylim(latitude_min, latitude_max)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.grid(True, alpha=0.10, color="#e2e8f0", linewidth=0.45)
    if attribution:
        ax.text(
            0.01,
            0.01,
            attribution,
            transform=ax.transAxes,
            fontsize=6.5,
            va="bottom",
            ha="left",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1.6},
            zorder=10,
        )
    if warning:
        ax.text(
            0.5,
            0.98,
            warning,
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            ha="center",
            color="#991b1b",
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#ef4444", "pad": 2.0},
            zorder=10,
        )


def plot_forecast_alarm_overlay(
    simulation: "ForecastAlarmSimulation",
    frame_index: int | None = None,
    basemap_styles: tuple[str | None, ...] = DEFAULT_FORECAST_BASEMAP_STYLES,
):
    if not simulation.frames:
        raise ValueError("simulation does not contain any forecast frames")
    frame_index = frame_index if frame_index is not None else 0
    frame = simulation.frames[frame_index]
    terrain = simulation.terrain
    overlay_locations = _simulation_overlay_neighborhoods(simulation, frame=frame, max_locations=6)
    plot_bounds = _terrain_bounds(
        terrain,
        location_report=_bounds_location_report_for_simulation(simulation, frame=frame),
        action_polygons=getattr(simulation, "action_polygons", None),
        source_longitude=getattr(simulation, "source_longitude", None),
        source_latitude=getattr(simulation, "source_latitude", None),
    )
    panels = _resolved_basemap_panels(
        terrain,
        tuple(basemap_styles) if basemap_styles else (None,),
        bounds=plot_bounds,
    )
    reference_labels = _reference_labels_for_map(simulation, bounds=plot_bounds, frame=frame)
    fig, axes = plt.subplots(1, len(panels), figsize=(11 * len(panels), 8), squeeze=False)
    axes = axes.ravel()

    plume_fill = None
    hazard_boundaries = None
    for ax, panel in zip(axes, panels):
        _add_basemap_background(ax, terrain, panel)
        plume_fill, hazard_boundaries = _overlay_plume(ax, terrain, frame.concentration)
        _plot_action_polygons(ax, getattr(simulation, "action_polygons", None))
        _plot_reference_labels(ax, reference_labels)
        _overlay_source_marker(ax, simulation)
        _plot_geo_impact_points(ax, overlay_locations)
        _format_basemap_axis(
            ax,
            terrain,
            title=panel.title,
            attribution=panel.attribution,
            warning=panel.warning,
            bounds=plot_bounds,
        )

    if plume_fill is not None:
        plume_colorbar = fig.colorbar(
            plume_fill,
            ax=axes.tolist(),
            label="log10(simulated concentration + 1e-9)",
            shrink=0.9,
            pad=0.01,
        )
        plume_colorbar.set_ticks(_plume_colorbar_ticks(plume_fill.levels))
        plume_colorbar.ax.tick_params(labelsize=8)

    _place_alert_band_legend(
        fig,
    )
    fig.suptitle(
        f"{simulation.incident.name}\n"
        f"Forecast overlay at {frame.timestamp.strftime('%Y-%m-%d %H:%M UTC')} | "
        f"wind from {frame.weather['wind_from_deg']:.0f}°, "
        f"speed={frame.weather['wind_speed_mps']:.1f} m/s"
    )
    fig.subplots_adjust(top=0.86, bottom=0.16, wspace=0.18)
    plt.show()
    return fig, axes[0] if len(axes) == 1 else axes


def animate_forecast_alarm(
    simulation: "ForecastAlarmSimulation",
    basemap_style: str | None = DEFAULT_ANIMATION_BASEMAP_STYLE,
):
    if not simulation.frames:
        raise ValueError("simulation does not contain any forecast frames")

    terrain = simulation.terrain
    fig, ax = plt.subplots(figsize=(11, 8))
    plot_bounds = _terrain_bounds(
        terrain,
        location_report=_bounds_location_report_for_simulation(simulation),
        action_polygons=getattr(simulation, "action_polygons", None),
        source_longitude=getattr(simulation, "source_longitude", None),
        source_latitude=getattr(simulation, "source_latitude", None),
    )
    panel = _resolve_basemap_panel(terrain, basemap_style, bounds=plot_bounds)
    reference_labels = _reference_labels_for_map(simulation, bounds=plot_bounds)
    _add_basemap_background(ax, terrain, panel)
    _plot_reference_labels(ax, reference_labels)
    _overlay_source_marker(ax, simulation)
    _format_basemap_axis(
        ax,
        terrain,
        title=panel.title,
        attribution=panel.attribution,
        warning=panel.warning,
        bounds=plot_bounds,
    )

    plume_holder = [
        _overlay_plume(ax, terrain, simulation.frames[0].concentration)
    ]
    plume_colorbar = plt.colorbar(plume_holder[0][0], ax=ax, label="log10(simulated concentration + 1e-9)")
    plume_colorbar.set_ticks(_plume_colorbar_ticks(plume_holder[0][0].levels))
    plume_colorbar.ax.tick_params(labelsize=8)
    _plot_action_polygons(ax, getattr(simulation, "action_polygons", None))
    _place_alert_band_legend(
        fig,
    )
    overlay_locations = _frame_overlay_locations(simulation.frames[0])
    text_handles = _plot_geo_impact_points(ax, overlay_locations)
    initial_frame = simulation.frames[0]
    ax.set_title(
        f"{simulation.incident.name}\n"
        f"Forecast spread at {initial_frame.timestamp.strftime('%Y-%m-%d %H:%M UTC')} | "
        f"wind from {initial_frame.weather['wind_from_deg']:.0f}°, "
        f"speed={initial_frame.weather['wind_speed_mps']:.1f} m/s"
    )
    fig.subplots_adjust(bottom=0.16)

    def update(frame_idx: int):
        _remove_contour_set(plume_holder[0][0])
        _remove_contour_set(plume_holder[0][1])
        for text_handle in text_handles[:]:
            text_handle.remove()
            text_handles.remove(text_handle)
        frame = simulation.frames[frame_idx]
        plume_holder[0] = _overlay_plume(ax, terrain, frame.concentration)
        overlay_locations = _frame_overlay_locations(frame)
        text_handles.extend(_plot_geo_impact_points(ax, overlay_locations))
        ax.set_title(
            f"{simulation.incident.name}\n"
            f"Forecast spread at {frame.timestamp.strftime('%Y-%m-%d %H:%M UTC')} | "
            f"wind from {frame.weather['wind_from_deg']:.0f}°, "
            f"speed={frame.weather['wind_speed_mps']:.1f} m/s"
        )
        return _contour_artists(plume_holder[0][0]) + _contour_artists(plume_holder[0][1]) + text_handles

    anim = FuncAnimation(
        fig,
        update,
        frames=len(simulation.frames),
        interval=_timelapse_interval_ms(len(simulation.frames)),
        blit=False,
    )
    plt.close(fig)
    return anim
