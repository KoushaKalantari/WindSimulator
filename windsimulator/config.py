from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlumeConfig:
    x_min_km: float = -25.0
    x_max_km: float = 25.0
    y_min_km: float = -20.0
    y_max_km: float = 20.0
    resolution: int = 320
    source_x_km: float = 0.0
    source_y_km: float = 0.0
    wind_from_deg: float = 270.0
    wind_speed_mps: float = 5.0
    emission_rate: float = 1200.0
    release_height_m: float = 10.0
    stability_class: str = "D"
    min_downwind_m: float = 10.0
    threshold_low: float = 0.5
    threshold_medium: float = 2.0
    threshold_high: float = 5.0


@dataclass(frozen=True)
class IncidentSourceTermPreset:
    label: str
    emission_multiplier: float
    release_duration_minutes: float
    initial_pulse_minutes: float
    release_height_m: float
    description: str


@dataclass(frozen=True)
class UncertaintyScenarioPreset:
    label: str
    direction_offsets_deg: tuple[float, ...]
    wind_speed_scale: float
    source_emission_scale: float
    release_duration_scale: float
    stability_shift: int
    aggregation: str
    description: str


@dataclass(frozen=True)
class HazardMaterialFatePreset:
    label: str
    reactive_airborne_half_life_minutes: float | None
    deposition_half_life_minutes: float | None
    residual_airborne_fraction: float
    description: str
    source_basis: str


DEFAULT_CONFIG = PlumeConfig()
DEFAULT_FORECAST_SIMULATION_RESOLUTION = 25
DEFAULT_INTERNAL_FORECAST_RESOLUTION = 40
DEFAULT_FORECAST_FRAME_INTERVAL_MINUTES = 60
DEFAULT_NOTEBOOK_INCIDENT_RESOLUTION = 20
DEFAULT_NOTEBOOK_TERRAIN_RESOLUTION = 20
DEFAULT_NOTEBOOK_TIMELAPSE_DURATION_HOURS = 24
DEFAULT_NOTEBOOK_TIMELAPSE_STEP_MINUTES = 10
DEFAULT_TERRAIN_SURFACE_RESOLUTION = 25
MAX_TERRAIN_PROVIDER_FETCH_RESOLUTION = 32
MIN_TERRAIN_PROVIDER_FETCH_RESOLUTION = 16

FICTIONAL_NEIGHBORHOODS: dict[str, tuple[float, float]] = {
    "Northgate": (2.0, 8.0),
    "Eastfield": (11.0, 1.0),
    "Westpark": (-9.5, -2.5),
    "Southbank": (1.0, -10.0),
    "Old Town": (5.0, 4.0),
    "Hillview": (-4.0, 9.0),
    "Riverside": (8.0, -6.5),
    "Market Row": (-1.0, 2.0),
}

DEFAULT_SCENARIOS: list[dict[str, float | str]] = [
    {
        "Scenario": "Cool-season westerly",
        "wind_from_deg": 270.0,
        "wind_speed_mps": 5.0,
        "stability_class": "D",
        "emission_rate": 1200.0,
    },
    {
        "Scenario": "Warm-season easterly",
        "wind_from_deg": 90.0,
        "wind_speed_mps": 5.0,
        "stability_class": "D",
        "emission_rate": 1200.0,
    },
    {
        "Scenario": "Stable night",
        "wind_from_deg": 270.0,
        "wind_speed_mps": 3.0,
        "stability_class": "F",
        "emission_rate": 1200.0,
    },
    {
        "Scenario": "Unstable day",
        "wind_from_deg": 270.0,
        "wind_speed_mps": 7.0,
        "stability_class": "B",
        "emission_rate": 1200.0,
    },
]

SEVERITY_MULTIPLIER_BY_LEVEL: dict[str, float] = {
    "minor": 0.5,
    "moderate": 1.0,
    "severe": 2.0,
    "critical": 4.0,
}

INCIDENT_TYPE_MULTIPLIER: dict[str, float] = {
    "threat": 0.75,
    "leakage": 1.0,
    "explosion": 2.5,
}

INCIDENT_TYPE_RELEASE_HEIGHT_M: dict[str, float] = {
    "threat": 10.0,
    "leakage": 10.0,
    "explosion": 25.0,
}

INCIDENT_SOURCE_TERMS: dict[str, IncidentSourceTermPreset] = {
    "threat": IncidentSourceTermPreset(
        label="Threat / suspicious release",
        emission_multiplier=0.75,
        release_duration_minutes=90.0,
        initial_pulse_minutes=0.0,
        release_height_m=10.0,
        description="Lower-confidence release with a shorter sustained source term.",
    ),
    "leakage": IncidentSourceTermPreset(
        label="Leakage / venting release",
        emission_multiplier=1.0,
        release_duration_minutes=240.0,
        initial_pulse_minutes=8.0,
        release_height_m=10.0,
        description="Sustained near-ground release with a small initial surge.",
    ),
    "explosion": IncidentSourceTermPreset(
        label="Explosion / energetic release",
        emission_multiplier=2.5,
        release_duration_minutes=45.0,
        initial_pulse_minutes=35.0,
        release_height_m=25.0,
        description="Front-loaded release with higher lofting and a shorter tail.",
    ),
}

UNCERTAINTY_SCENARIOS: tuple[UncertaintyScenarioPreset, ...] = (
    UncertaintyScenarioPreset(
        label="Likely",
        direction_offsets_deg=(0.0,),
        wind_speed_scale=1.0,
        source_emission_scale=1.0,
        release_duration_scale=1.0,
        stability_shift=0,
        aggregation="baseline",
        description="Best-estimate run using the forecast winds and the resolved source term.",
    ),
    UncertaintyScenarioPreset(
        label="Conservative",
        direction_offsets_deg=(-8.0, 0.0, 8.0),
        wind_speed_scale=0.92,
        source_emission_scale=1.2,
        release_duration_scale=1.25,
        stability_shift=1,
        aggregation="max",
        description="Planning envelope that widens the plume with modest wind and source uncertainty.",
    ),
    UncertaintyScenarioPreset(
        label="Worst Reasonable",
        direction_offsets_deg=(-15.0, 0.0, 15.0),
        wind_speed_scale=0.82,
        source_emission_scale=1.45,
        release_duration_scale=1.5,
        stability_shift=2,
        aggregation="max",
        description="Stress-case envelope within a still-plausible range of source and wind variation.",
    ),
)

MATERIAL_FATE_PROFILES: dict[str, HazardMaterialFatePreset] = {
    "generic_toxic_gas": HazardMaterialFatePreset(
        label="Generic volatile toxic gas",
        reactive_airborne_half_life_minutes=2880.0,
        deposition_half_life_minutes=None,
        residual_airborne_fraction=0.0,
        description=(
            "Heuristic gas-phase fade for generic VOC-like releases; use a substance-specific "
            "model when the chemical identity is known."
        ),
        source_basis=(
            "ATSDR notes benzene evaporates into air very quickly and breaks down in air within a few days."
        ),
    ),
    "oil_smoke": HazardMaterialFatePreset(
        label="Oil smoke / combustion aerosol",
        reactive_airborne_half_life_minutes=None,
        deposition_half_life_minutes=4320.0,
        residual_airborne_fraction=1.0,
        description=(
            "Fine smoke particulates can remain airborne for days but gradually settle and are "
            "removed from the air over time."
        ),
        source_basis=(
            "EPA particle pollution guidance says fine particles can stay suspended for long periods, "
            "and CDC notes smoke can remain in the air for days after a fire."
        ),
    ),
    "radioactive_particulate": HazardMaterialFatePreset(
        label="Radioactive particulate / fallout-like aerosol",
        reactive_airborne_half_life_minutes=None,
        deposition_half_life_minutes=7920.0,
        residual_airborne_fraction=1.0,
        description=(
            "Without isotope-specific data, do not assume meaningful radiological decay over a "
            "24-hour run; model primarily slow airborne removal of particulate contamination."
        ),
        source_basis=(
            "EPA notes radioactive decay can range from hours to millions of years, so airborne "
            "removal is modeled separately from isotope decay on this timescale."
        ),
    ),
    "uf6": HazardMaterialFatePreset(
        label="UF6 hydrolysis plume",
        reactive_airborne_half_life_minutes=35.0,
        deposition_half_life_minutes=480.0,
        residual_airborne_fraction=0.35,
        description=(
            "Models rapid UF6 hydrolysis plus a smaller lingering airborne hazard from HF/uranyl "
            "fluoride that deposits faster than generic smoke."
        ),
        source_basis=(
            "ATSDR reports airborne UF6 has an about 35-minute half-life and forms uranyl fluoride "
            "and HF; NRC notes UF6 reacts rapidly with moisture."
        ),
    ),
}

MATERIAL_FATE_PROFILE_ALIASES: dict[str, str] = {
    "gas": "generic_toxic_gas",
    "generic_gas": "generic_toxic_gas",
    "toxic_gas": "generic_toxic_gas",
    "fumes": "generic_toxic_gas",
    "volatile_gas": "generic_toxic_gas",
    "voc": "generic_toxic_gas",
    "smoke": "oil_smoke",
    "combustion_smoke": "oil_smoke",
    "toxic_oil_smoke": "oil_smoke",
    "radioactive": "radioactive_particulate",
    "radiological": "radioactive_particulate",
    "nuclear": "radioactive_particulate",
    "nulcear": "radioactive_particulate",
    "nuclear_waste": "radioactive_particulate",
    "nulcear_waste": "radioactive_particulate",
    "fallout": "radioactive_particulate",
    "uf6_residue": "uf6",
    "airborne_uf6_residue": "uf6",
}

RECOMMENDED_ACTION_BY_BAND: dict[str, str] = {
    "MINIMAL": "No neighborhood broadcast",
    "LOW": "Precautionary evacuation notice",
    "MEDIUM": "Urgent evacuation notice",
    "HIGH": "Immediate evacuation order",
}

ALERT_COLOR_BY_BAND: dict[str, str] = {
    "MINIMAL": "#94a3b8",
    "LOW": "#facc15",
    "MEDIUM": "#fb923c",
    "HIGH": "#ef4444",
}

EMERGENCY_ALERT_THRESHOLDS: dict[str, float] = {
    "LOW": 1e-8,
    "MEDIUM": 1e-6,
    "HIGH": 1e-4,
}

MODEL_NOTES = """
Model caveats:
1. This is a simplified educational Gaussian plume / puff model.
2. It uses public forecast winds, incident-specific source-term presets, turn-aware puff transport heuristics, terrain channeling/blocking, elevation-aware ground impacts, and simplified material-persistence heuristics, not a validated dispersion solver.
3. It does not model buildings, street canyons, rainout, full atmospheric chemistry, isotope-specific radiological decay, or full 3D fluid dynamics.
4. Its uncertainty envelopes and action polygons are derived from internal heuristic scenarios, not authoritative evacuation polygons or validated threat zones.
5. Thresholds here are arbitrary educational bands, not official health thresholds.
6. Real emergency-response modeling requires validated source terms, local sensor data, specialist tools, and human review.
7. This repository is a simulation prototype and should not be used as the sole basis for real evacuation orders.
8. Notice levels here are internal demo tiers aligned loosely to CAP-style severity/urgency terms, not official IPAWS issuance criteria.

Recommended safe uses:
- Education
- Emergency-planning concepts
- Comparing generic weather scenarios
- Testing visualization ideas on public map and terrain data
""".strip()
