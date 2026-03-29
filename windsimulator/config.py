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


DEFAULT_CONFIG = PlumeConfig()

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
2. It uses coarse public forecast winds and a simple terrain/elevation modifier, not a validated dispersion solver.
3. It does not model buildings, street canyons, rainout, chemistry, or full 3D fluid dynamics.
4. Place names and postal codes are estimated from reverse geocoding around simulated hotspots, not authoritative evacuation polygons.
5. Thresholds here are arbitrary educational bands, not official health thresholds.
6. Real emergency-response modeling requires validated source terms, local sensor data, specialist tools, and human review.
7. This repository is a simulation prototype and should not be used as the sole basis for real evacuation orders.

Recommended safe uses:
- Education
- Emergency-planning concepts
- Comparing generic weather scenarios
- Testing visualization ideas on public map and terrain data
""".strip()
