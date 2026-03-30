from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .analysis import copy_config, sample_neighborhoods
from .config import (
    ALERT_COLOR_BY_BAND,
    DEFAULT_CONFIG,
    EMERGENCY_ALERT_THRESHOLDS,
    FICTIONAL_NEIGHBORHOODS,
    INCIDENT_SOURCE_TERMS,
    INCIDENT_TYPE_MULTIPLIER,
    INCIDENT_TYPE_RELEASE_HEIGHT_M,
    MATERIAL_FATE_PROFILES,
    MATERIAL_FATE_PROFILE_ALIASES,
    RECOMMENDED_ACTION_BY_BAND,
    SEVERITY_MULTIPLIER_BY_LEVEL,
    HazardMaterialFatePreset,
    IncidentSourceTermPreset,
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
    source_term_profile: str | None = None
    hazard_material: str | None = None
    emission_rate_override: float | None = None
    release_duration_minutes: float | None = None
    initial_pulse_minutes: float | None = None


@dataclass
class AlertPlan:
    incident: HazardIncident
    source_term: "ResolvedSourceTerm"
    cfg: PlumeConfig
    neighborhood_report: pd.DataFrame
    impacted_neighborhoods: pd.DataFrame
    notice_payloads: list[dict[str, Any]]


@dataclass(frozen=True)
class ResolvedSourceTerm:
    profile_key: str
    profile_label: str
    emission_rate: float
    release_duration_minutes: float
    initial_pulse_minutes: float
    initial_pulse_mass: float
    release_height_m: float
    description: str


@dataclass(frozen=True)
class ResolvedMaterialFate:
    profile_key: str
    profile_label: str
    reactive_airborne_half_life_minutes: float | None
    deposition_half_life_minutes: float | None
    residual_airborne_fraction: float
    description: str
    source_basis: str


NOTICE_LEVEL_BY_BAND: dict[str, str] = {
    "MINIMAL": "Monitor",
    "LOW": "Advisory",
    "MEDIUM": "Warning",
    "HIGH": "Immediate Action",
}

NOTICE_LEVEL_PRIORITY: dict[str, int] = {
    "Monitor": 0,
    "Advisory": 1,
    "Warning": 2,
    "Immediate Action": 3,
}

CAP_SEVERITY_BY_BAND: dict[str, str] = {
    "MINIMAL": "Minor",
    "LOW": "Minor",
    "MEDIUM": "Moderate",
    "HIGH": "Severe",
}

CAP_URGENCY_PRIORITY: dict[str, int] = {
    "Unknown": 0,
    "Future": 1,
    "Expected": 2,
    "Immediate": 3,
}


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


def _normalized_source_term_key(incident_type: str) -> str:
    normalized = incident_type.strip().lower()
    if normalized not in INCIDENT_SOURCE_TERMS:
        valid_types = ", ".join(sorted(INCIDENT_SOURCE_TERMS))
        raise ValueError(f"incident_type must be one of: {valid_types}")
    return normalized


def _normalized_material_fate_key(hazard_material: str | None) -> str:
    raw_key = (hazard_material or "generic_toxic_gas").strip().lower()
    raw_key = "_".join(part for part in raw_key.replace("-", " ").split() if part)
    normalized = MATERIAL_FATE_PROFILE_ALIASES.get(raw_key, raw_key)
    if normalized not in MATERIAL_FATE_PROFILES:
        valid_materials = ", ".join(sorted(MATERIAL_FATE_PROFILES))
        raise ValueError(f"hazard_material must be one of: {valid_materials}")
    return normalized


def incident_source_term_preset(
    incident_type: str,
    source_term_profile: str | None = None,
) -> tuple[str, IncidentSourceTermPreset]:
    profile_key = source_term_profile.strip().lower() if source_term_profile else incident_type
    normalized = _normalized_source_term_key(profile_key)
    return normalized, INCIDENT_SOURCE_TERMS[normalized]


def material_fate_preset(
    hazard_material: str | None,
) -> tuple[str, HazardMaterialFatePreset]:
    normalized = _normalized_material_fate_key(hazard_material)
    return normalized, MATERIAL_FATE_PROFILES[normalized]


def incident_type_multiplier(incident_type: str) -> float:
    normalized, preset = incident_source_term_preset(incident_type)
    return INCIDENT_TYPE_MULTIPLIER.get(normalized, preset.emission_multiplier)


def incident_release_height(
    incident_type: str,
    base_cfg: PlumeConfig,
    explicit_release_height_m: float | None = None,
    source_term_profile: str | None = None,
) -> float:
    if explicit_release_height_m is not None:
        return explicit_release_height_m
    normalized, preset = incident_source_term_preset(incident_type, source_term_profile)
    return max(
        base_cfg.release_height_m,
        INCIDENT_TYPE_RELEASE_HEIGHT_M.get(normalized, preset.release_height_m),
    )


def resolve_incident_source_term(
    incident: HazardIncident,
    base_cfg: PlumeConfig = DEFAULT_CONFIG,
    *,
    emission_scale: float = 1.0,
    duration_scale: float = 1.0,
) -> ResolvedSourceTerm:
    profile_key, preset = incident_source_term_preset(
        incident.incident_type,
        incident.source_term_profile,
    )
    emission_rate = (
        float(incident.emission_rate_override)
        if incident.emission_rate_override is not None
        else (
            base_cfg.emission_rate
            * severity_multiplier(incident.severity)
            * preset.emission_multiplier
        )
    )
    emission_rate = max(float(emission_rate) * float(emission_scale), 0.0)
    release_duration_minutes = max(
        float(incident.release_duration_minutes)
        if incident.release_duration_minutes is not None
        else float(preset.release_duration_minutes),
        0.0,
    ) * max(float(duration_scale), 0.1)
    initial_pulse_minutes = max(
        float(incident.initial_pulse_minutes)
        if incident.initial_pulse_minutes is not None
        else float(preset.initial_pulse_minutes),
        0.0,
    ) * max(float(duration_scale), 0.1)
    release_height_m = incident_release_height(
        incident.incident_type,
        base_cfg,
        incident.release_height_m,
        incident.source_term_profile,
    )
    return ResolvedSourceTerm(
        profile_key=profile_key,
        profile_label=preset.label,
        emission_rate=emission_rate,
        release_duration_minutes=release_duration_minutes,
        initial_pulse_minutes=initial_pulse_minutes,
        initial_pulse_mass=emission_rate * initial_pulse_minutes * 60.0,
        release_height_m=release_height_m,
        description=preset.description,
    )


def resolve_material_fate(hazard_material: str | None) -> ResolvedMaterialFate:
    profile_key, preset = material_fate_preset(hazard_material)
    return ResolvedMaterialFate(
        profile_key=profile_key,
        profile_label=preset.label,
        reactive_airborne_half_life_minutes=preset.reactive_airborne_half_life_minutes,
        deposition_half_life_minutes=preset.deposition_half_life_minutes,
        residual_airborne_fraction=float(preset.residual_airborne_fraction),
        description=preset.description,
        source_basis=preset.source_basis,
    )


def incident_to_config(
    incident: HazardIncident,
    base_cfg: PlumeConfig = DEFAULT_CONFIG,
) -> PlumeConfig:
    source_term = resolve_incident_source_term(incident, base_cfg=base_cfg)
    return copy_config(
        base_cfg,
        source_x_km=incident.source_x_km,
        source_y_km=incident.source_y_km,
        emission_rate=source_term.emission_rate,
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
            incident.source_term_profile,
        ),
    )


def recommended_action_for_band(band: str) -> str:
    return RECOMMENDED_ACTION_BY_BAND.get(band, "Manual review required")


def notice_level_for_band(band: str) -> str:
    return NOTICE_LEVEL_BY_BAND.get(str(band).upper(), "Monitor")


def cap_severity_for_band(band: str) -> str:
    return CAP_SEVERITY_BY_BAND.get(str(band).upper(), "Unknown")


def minutes_to_impact(
    first_impacted_time: pd.Timestamp | None,
    reference_time: pd.Timestamp | None,
) -> float | None:
    if first_impacted_time is None or reference_time is None:
        return None
    if pd.isna(first_impacted_time) or pd.isna(reference_time):
        return None
    delta_minutes = (
        pd.Timestamp(first_impacted_time) - pd.Timestamp(reference_time)
    ).total_seconds() / 60.0
    return max(float(delta_minutes), 0.0)


def cap_urgency_for_minutes_to_impact(minutes_until_impact: float | None) -> str:
    if minutes_until_impact is None:
        return "Unknown"
    if minutes_until_impact <= 15.0:
        return "Immediate"
    if minutes_until_impact <= 60.0:
        return "Expected"
    return "Future"


def _broadcast_basis_text(
    band: str,
    notice_level: str,
    cap_urgency: str,
    peak_concentration: float | None = None,
    minutes_until_impact: float | None = None,
) -> str:
    band_text = f"PeakBand={band}"
    if peak_concentration is not None:
        band_text += f", PeakConcentration={float(peak_concentration):.2e}"
    if notice_level == "Monitor":
        return (
            f"{band_text}; NoticeLevel=Monitor; below the demo's public-notice threshold."
        )
    impact_text = ""
    if minutes_until_impact is not None:
        impact_text = f", MinutesToImpact={int(round(minutes_until_impact))}"
    return (
        f"{band_text}; NoticeLevel={notice_level}; CAPUrgency={cap_urgency}"
        f"{impact_text}."
    )


def broadcast_decision(
    band: str,
    *,
    first_impacted_time: pd.Timestamp | None = None,
    reference_time: pd.Timestamp | None = None,
    peak_concentration: float | None = None,
    minutes_until_impact: float | None = None,
) -> dict[str, Any]:
    normalized_band = str(band).upper()
    notice_level = notice_level_for_band(normalized_band)
    if minutes_until_impact is None:
        minutes_until_impact = minutes_to_impact(first_impacted_time, reference_time)
    cap_urgency = cap_urgency_for_minutes_to_impact(minutes_until_impact)
    return {
        "NoticeLevel": notice_level,
        "CAPSeverity": cap_severity_for_band(normalized_band),
        "CAPUrgency": cap_urgency,
        "MinutesToImpact": minutes_until_impact,
        "BroadcastRecommended": notice_level != "Monitor",
        "BroadcastBasis": _broadcast_basis_text(
            normalized_band,
            notice_level,
            cap_urgency,
            peak_concentration=peak_concentration,
            minutes_until_impact=minutes_until_impact,
        ),
    }


def rank_broadcast_areas(
    report: pd.DataFrame,
    *,
    concentration_column: str,
    first_impacted_time_column: str | None = "FirstImpactedTime",
    hotspot_count_column: str | None = None,
) -> pd.DataFrame:
    if report.empty:
        return report

    ranked = report.copy()
    ranked["_NoticePriority"] = ranked["NoticeLevel"].map(NOTICE_LEVEL_PRIORITY).fillna(0)
    ranked["_UrgencyPriority"] = ranked["CAPUrgency"].map(CAP_URGENCY_PRIORITY).fillna(0)

    sort_columns = ["_NoticePriority", "_UrgencyPriority"]
    ascending = [False, False]

    if first_impacted_time_column and first_impacted_time_column in ranked.columns:
        ranked["_FirstImpactSort"] = pd.to_datetime(
            ranked[first_impacted_time_column],
            utc=True,
            errors="coerce",
        ).fillna(pd.Timestamp("2262-01-01T00:00:00Z"))
        sort_columns.append("_FirstImpactSort")
        ascending.append(True)

    if hotspot_count_column and hotspot_count_column in ranked.columns:
        sort_columns.append(hotspot_count_column)
        ascending.append(False)

    sort_columns.append(concentration_column)
    ascending.append(False)

    ranked = ranked.sort_values(sort_columns, ascending=ascending).reset_index(drop=True)
    ranked["BroadcastPriorityRank"] = np.arange(1, len(ranked) + 1)
    drop_columns = ["_NoticePriority", "_UrgencyPriority"]
    if "_FirstImpactSort" in ranked.columns:
        drop_columns.append("_FirstImpactSort")
    return ranked.drop(columns=drop_columns)


def broadcast_recommended_for_band(band: str) -> bool:
    return notice_level_for_band(band) != "Monitor"


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
    notice_level: str | None = None,
    cap_urgency: str | None = None,
) -> str:
    incident_type = incident.incident_type.strip().lower()
    notice_level = notice_level or notice_level_for_band(band)
    urgency_text = ""
    if cap_urgency and cap_urgency != "Unknown":
        urgency_text = f" Urgency: {cap_urgency}."
    return (
        f"Emergency notice for {neighborhood}: "
        f"A simulated {incident_type} event near "
        f"({incident.source_x_km:.1f} km, {incident.source_y_km:.1f} km) "
        f"may impact your area. Predicted impact level: {band}. "
        f"Recommended notice level: {notice_level}.{urgency_text} "
        f"Recommended action: {action}. Follow official emergency instructions."
    )


def build_notice_payloads(impacted_neighborhoods: pd.DataFrame) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for row in impacted_neighborhoods.to_dict(orient="records"):
        payloads.append(
            {
                "neighborhood": row["Neighborhood"],
                "band": row["Band"],
                "notice_level": row.get("NoticeLevel"),
                "cap_severity": row.get("CAPSeverity"),
                "cap_urgency": row.get("CAPUrgency"),
                "broadcast_priority_rank": row.get("BroadcastPriorityRank"),
                "recommended_action": row["RecommendedAction"],
                "broadcast_basis": row.get("BroadcastBasis"),
                "message": row["DraftNotice"],
            }
        )
    return payloads


def analyze_incident_alert(
    incident: HazardIncident,
    base_cfg: PlumeConfig = DEFAULT_CONFIG,
    neighborhoods: dict[str, tuple[float, float]] = FICTIONAL_NEIGHBORHOODS,
) -> AlertPlan:
    source_term = resolve_incident_source_term(incident, base_cfg=base_cfg)
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
    report["FirstImpactedTime"] = pd.NaT
    broadcast_details = report.apply(
        lambda row: broadcast_decision(
            row["Band"],
            peak_concentration=row["Concentration"],
        ),
        axis=1,
        result_type="expand",
    )
    report = pd.concat([report, broadcast_details], axis=1)
    report["DraftNotice"] = [
        build_notice_message(
            neighborhood=row["Neighborhood"],
            band=row["Band"],
            action=row["RecommendedAction"],
            incident=incident,
            notice_level=row["NoticeLevel"],
            cap_urgency=row["CAPUrgency"],
        )
        for row in report.to_dict(orient="records")
    ]
    report = rank_broadcast_areas(
        report,
        concentration_column="Concentration",
        first_impacted_time_column=None,
    )

    impacted = report[report["BroadcastRecommended"]].reset_index(drop=True)
    payloads = build_notice_payloads(impacted)
    return AlertPlan(
        incident=incident,
        source_term=source_term,
        cfg=cfg,
        neighborhood_report=report,
        impacted_neighborhoods=impacted,
        notice_payloads=payloads,
    )
