from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .analysis import copy_config, sample_neighborhoods
from .config import (
    ALERT_COLOR_BY_BAND,
    DEFAULT_CONFIG,
    EMERGENCY_ALERT_THRESHOLDS,
    FICTIONAL_NEIGHBORHOODS,
    INCIDENT_TYPE_MULTIPLIER,
    INCIDENT_TYPE_RELEASE_HEIGHT_M,
    RECOMMENDED_ACTION_BY_BAND,
    SEVERITY_MULTIPLIER_BY_LEVEL,
    PlumeConfig,
)


@dataclass(frozen=True)
class HazardIncident:
    source_x_km: float
    source_y_km: float
    severity: str | float = "moderate"
    incident_type: str = "leakage"
    name: str = "Simulated Hazard Incident"
    wind_from_deg: float | None = None
    wind_speed_mps: float | None = None
    stability_class: str | None = None
    release_height_m: float | None = None


@dataclass
class AlertPlan:
    incident: HazardIncident
    cfg: PlumeConfig
    neighborhood_report: pd.DataFrame
    impacted_neighborhoods: pd.DataFrame
    notice_payloads: list[dict[str, Any]]


def severity_multiplier(severity: str | float) -> float:
    if isinstance(severity, (int, float)):
        return max(float(severity), 0.1)

    normalized = str(severity).strip().lower()
    if normalized not in SEVERITY_MULTIPLIER_BY_LEVEL:
        valid_levels = ", ".join(sorted(SEVERITY_MULTIPLIER_BY_LEVEL))
        raise ValueError(
            f"severity must be numeric or one of: {valid_levels}"
        )
    return SEVERITY_MULTIPLIER_BY_LEVEL[normalized]


def incident_type_multiplier(incident_type: str) -> float:
    normalized = incident_type.strip().lower()
    if normalized not in INCIDENT_TYPE_MULTIPLIER:
        valid_types = ", ".join(sorted(INCIDENT_TYPE_MULTIPLIER))
        raise ValueError(f"incident_type must be one of: {valid_types}")
    return INCIDENT_TYPE_MULTIPLIER[normalized]


def incident_release_height(
    incident_type: str,
    base_cfg: PlumeConfig,
    explicit_release_height_m: float | None = None,
) -> float:
    if explicit_release_height_m is not None:
        return explicit_release_height_m
    normalized = incident_type.strip().lower()
    return max(
        base_cfg.release_height_m,
        INCIDENT_TYPE_RELEASE_HEIGHT_M[normalized],
    )


def incident_to_config(
    incident: HazardIncident,
    base_cfg: PlumeConfig = DEFAULT_CONFIG,
) -> PlumeConfig:
    emission_rate = (
        base_cfg.emission_rate
        * severity_multiplier(incident.severity)
        * incident_type_multiplier(incident.incident_type)
    )
    return copy_config(
        base_cfg,
        source_x_km=incident.source_x_km,
        source_y_km=incident.source_y_km,
        emission_rate=emission_rate,
        wind_from_deg=incident.wind_from_deg
        if incident.wind_from_deg is not None
        else base_cfg.wind_from_deg,
        wind_speed_mps=incident.wind_speed_mps
        if incident.wind_speed_mps is not None
        else base_cfg.wind_speed_mps,
        stability_class=incident.stability_class
        if incident.stability_class is not None
        else base_cfg.stability_class,
        release_height_m=incident_release_height(
            incident.incident_type,
            base_cfg,
            incident.release_height_m,
        ),
    )


def recommended_action_for_band(band: str) -> str:
    return RECOMMENDED_ACTION_BY_BAND.get(band, "Manual review required")


def broadcast_recommended_for_band(band: str) -> bool:
    return band in {"LOW", "MEDIUM", "HIGH"}


def emergency_band_for_concentration(concentration: float) -> str:
    if concentration >= EMERGENCY_ALERT_THRESHOLDS["HIGH"]:
        return "HIGH"
    if concentration >= EMERGENCY_ALERT_THRESHOLDS["MEDIUM"]:
        return "MEDIUM"
    if concentration >= EMERGENCY_ALERT_THRESHOLDS["LOW"]:
        return "LOW"
    return "MINIMAL"


def build_notice_message(
    neighborhood: str,
    band: str,
    action: str,
    incident: HazardIncident,
) -> str:
    incident_type = incident.incident_type.strip().lower()
    return (
        f"Emergency notice for {neighborhood}: "
        f"A simulated {incident_type} event near "
        f"({incident.source_x_km:.1f} km, {incident.source_y_km:.1f} km) "
        f"may impact your area. Predicted impact level: {band}. "
        f"Recommended action: {action}. Follow official emergency instructions."
    )


def build_notice_payloads(impacted_neighborhoods: pd.DataFrame) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for row in impacted_neighborhoods.to_dict(orient="records"):
        payloads.append(
            {
                "neighborhood": row["Neighborhood"],
                "band": row["Band"],
                "recommended_action": row["RecommendedAction"],
                "message": row["DraftNotice"],
            }
        )
    return payloads


def analyze_incident_alert(
    incident: HazardIncident,
    base_cfg: PlumeConfig = DEFAULT_CONFIG,
    neighborhoods: dict[str, tuple[float, float]] = FICTIONAL_NEIGHBORHOODS,
) -> AlertPlan:
    cfg = incident_to_config(incident, base_cfg=base_cfg)
    report = sample_neighborhoods(cfg, neighborhoods).copy()
    report["ModelBand"] = report["Band"]
    report["Band"] = report["Concentration"].map(emergency_band_for_concentration)
    report["IncidentName"] = incident.name
    report["IncidentType"] = incident.incident_type
    report["SeverityInput"] = incident.severity
    report["SeverityMultiplier"] = severity_multiplier(incident.severity)
    report["Color"] = report["Band"].map(ALERT_COLOR_BY_BAND)
    report["RecommendedAction"] = report["Band"].map(recommended_action_for_band)
    report["BroadcastRecommended"] = report["Band"].map(broadcast_recommended_for_band)
    report["DraftNotice"] = [
        build_notice_message(
            neighborhood=row["Neighborhood"],
            band=row["Band"],
            action=row["RecommendedAction"],
            incident=incident,
        )
        for row in report.to_dict(orient="records")
    ]

    impacted = report[report["BroadcastRecommended"]].reset_index(drop=True)
    payloads = build_notice_payloads(impacted)
    return AlertPlan(
        incident=incident,
        cfg=cfg,
        neighborhood_report=report,
        impacted_neighborhoods=impacted,
        notice_payloads=payloads,
    )
