from __future__ import annotations

import importlib.util
import subprocess
import sys


REQUIRED_DEMO_PACKAGES: dict[str, str] = {
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "ipywidgets": "ipywidgets",
    "certifi": "certifi",
    "PIL": "Pillow",
    "arabic_reshaper": "arabic-reshaper",
    "bidi": "python-bidi",
}


def ensure_demo_dependencies(
    packages: dict[str, str] | None = None,
) -> list[str]:
    packages = packages or REQUIRED_DEMO_PACKAGES
    missing = [
        package_name
        for module_name, package_name in packages.items()
        if importlib.util.find_spec(module_name) is None
    ]
    if missing:
        if importlib.util.find_spec("pip") is None:
            import ensurepip

            ensurepip.bootstrap()
        print(f"Installing missing packages: {', '.join(missing)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
    else:
        print("All required packages are already installed.")
    return missing


def enable_notebook_inline_plots() -> None:
    try:
        get_ipython().run_line_magic("matplotlib", "inline")
    except Exception:
        pass


def run_notebook_alarm_demo(**kwargs):
    ensure_demo_dependencies()
    enable_notebook_inline_plots()
    from .demo import run_alarm_demo

    return run_alarm_demo(**kwargs)


def run_notebook_forecast_alarm_demo(**kwargs):
    ensure_demo_dependencies()
    enable_notebook_inline_plots()
    from .demo import run_forecast_alarm_demo

    return run_forecast_alarm_demo(**kwargs)


def run_notebook_incident_demo(
    location: tuple[float, float],
    incident_time="auto",
    severity: str | float = "severe",
    **kwargs,
):
    ensure_demo_dependencies()
    enable_notebook_inline_plots()
    from .config import (
        DEFAULT_NOTEBOOK_INCIDENT_RESOLUTION,
        DEFAULT_NOTEBOOK_TERRAIN_RESOLUTION,
        DEFAULT_NOTEBOOK_TIMELAPSE_STEP_MINUTES,
    )
    from .demo import run_forecast_alarm_demo

    kwargs.setdefault("incident_type", "leakage")
    kwargs.setdefault("name", "Forecast Hazard Demo")
    kwargs.setdefault("duration_hours", "auto")
    kwargs.setdefault("forecast_hours", 48)
    kwargs.setdefault("frame_interval_minutes", DEFAULT_NOTEBOOK_TIMELAPSE_STEP_MINUTES)
    kwargs.setdefault("simulation_resolution", DEFAULT_NOTEBOOK_INCIDENT_RESOLUTION)
    kwargs.setdefault("terrain_resolution", DEFAULT_NOTEBOOK_TERRAIN_RESOLUTION)
    kwargs.setdefault("hazard_material", "generic_toxic_gas")
    kwargs.setdefault("show_overlay", False)
    kwargs.setdefault("show_animation", True)
    kwargs.setdefault("verbose", True)
    return run_forecast_alarm_demo(
        source_latitude=float(location[0]),
        source_longitude=float(location[1]),
        incident_time=incident_time,
        severity=severity,
        **kwargs,
    )
