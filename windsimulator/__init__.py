from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "AlertPlan": (".emergency", "AlertPlan"),
    "HAS_WIDGETS": (".widgets", "HAS_WIDGETS"),
    "DEFAULT_CONFIG": (".config", "DEFAULT_CONFIG"),
    "DEFAULT_SCENARIOS": (".config", "DEFAULT_SCENARIOS"),
    "FICTIONAL_NEIGHBORHOODS": (".config", "FICTIONAL_NEIGHBORHOODS"),
    "HazardIncident": (".emergency", "HazardIncident"),
    "MODEL_NOTES": (".config", "MODEL_NOTES"),
    "PlumeConfig": (".config", "PlumeConfig"),
    "analyze_incident_alert": (".emergency", "analyze_incident_alert"),
    "animate_puff": (".plotting", "animate_puff"),
    "copy_config": (".analysis", "copy_config"),
    "enable_notebook_inline_plots": (".bootstrap", "enable_notebook_inline_plots"),
    "ensure_demo_dependencies": (".bootstrap", "ensure_demo_dependencies"),
    "interactive_plume_demo": (".widgets", "interactive_plume_demo"),
    "plot_alert_overlay": (".plotting", "plot_alert_overlay"),
    "plot_forecast_alarm_overlay": (".plotting", "plot_forecast_alarm_overlay"),
    "plot_plume": (".plotting", "plot_plume"),
    "plot_puff": (".plotting", "plot_puff"),
    "run_alarm_demo": (".demo", "run_alarm_demo"),
    "run_demo_experiment": (".demo", "run_demo_experiment"),
    "run_forecast_alarm_demo": (".demo", "run_forecast_alarm_demo"),
    "run_interactive_demo": (".demo", "run_interactive_demo"),
    "run_notebook_incident_demo": (".bootstrap", "run_notebook_incident_demo"),
    "run_notebook_alarm_demo": (".bootstrap", "run_notebook_alarm_demo"),
    "run_notebook_forecast_alarm_demo": (".bootstrap", "run_notebook_forecast_alarm_demo"),
    "sample_neighborhoods": (".analysis", "sample_neighborhoods"),
    "scenario_configs": (".analysis", "scenario_configs"),
    "scenario_pivots": (".analysis", "scenario_pivots"),
    "scenario_summary": (".analysis", "scenario_summary"),
    "simulate_forecast_alarm": (".forecast_alarm", "simulate_forecast_alarm"),
}

__all__ = [
    "AlertPlan",
    "HAS_WIDGETS",
    "DEFAULT_CONFIG",
    "DEFAULT_SCENARIOS",
    "FICTIONAL_NEIGHBORHOODS",
    "HazardIncident",
    "MODEL_NOTES",
    "PlumeConfig",
    "analyze_incident_alert",
    "animate_puff",
    "copy_config",
    "enable_notebook_inline_plots",
    "ensure_demo_dependencies",
    "interactive_plume_demo",
    "plot_alert_overlay",
    "plot_forecast_alarm_overlay",
    "plot_plume",
    "plot_puff",
    "run_alarm_demo",
    "run_demo_experiment",
    "run_forecast_alarm_demo",
    "run_interactive_demo",
    "run_notebook_incident_demo",
    "run_notebook_alarm_demo",
    "run_notebook_forecast_alarm_demo",
    "sample_neighborhoods",
    "scenario_configs",
    "scenario_pivots",
    "scenario_summary",
    "simulate_forecast_alarm",
]


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _EXPORTS[name]
    module = import_module(module_name, __name__)
    return getattr(module, attribute_name)


def __dir__():
    return sorted(set(globals()) | set(__all__))
