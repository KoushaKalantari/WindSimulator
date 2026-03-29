from __future__ import annotations

from dataclasses import replace

from .analysis import sample_neighborhoods
from .config import PlumeConfig
from .plotting import plot_plume

try:
    import ipywidgets as widgets
    from ipywidgets import interact

    HAS_WIDGETS = True
except Exception:
    widgets = None
    interact = None
    HAS_WIDGETS = False


def interactive_plume_demo(
    base_cfg: PlumeConfig,
    neighborhoods: dict[str, tuple[float, float]],
):
    if not HAS_WIDGETS:
        print("ipywidgets not installed. Static notebook cells still work.")
        return None

    def render_demo(
        wind_from_deg: float = 270.0,
        wind_speed_mps: float = 5.0,
        stability_class: str = "D",
        emission_rate: float = 1200.0,
        release_height_m: float = 10.0,
    ):
        cfg = replace(
            base_cfg,
            wind_from_deg=wind_from_deg,
            wind_speed_mps=wind_speed_mps,
            stability_class=stability_class,
            emission_rate=emission_rate,
            release_height_m=release_height_m,
        )
        plot_plume(cfg, neighborhoods=neighborhoods, title_suffix="(Interactive)")
        table = sample_neighborhoods(cfg, neighborhoods)
        try:
            from IPython.display import display

            display(table)
        except Exception:
            print(table)

    return interact(
        render_demo,
        wind_from_deg=widgets.FloatSlider(
            min=0,
            max=360,
            step=5,
            value=base_cfg.wind_from_deg,
            description="Wind from",
        ),
        wind_speed_mps=widgets.FloatSlider(
            min=0.5,
            max=20.0,
            step=0.5,
            value=base_cfg.wind_speed_mps,
            description="Wind m/s",
        ),
        stability_class=widgets.Dropdown(
            options=["A", "B", "C", "D", "E", "F"],
            value=base_cfg.stability_class,
            description="Stability",
        ),
        emission_rate=widgets.FloatLogSlider(
            base=10,
            min=1,
            max=5,
            step=0.1,
            value=base_cfg.emission_rate,
            description="Emission",
        ),
        release_height_m=widgets.FloatSlider(
            min=0,
            max=100,
            step=1,
            value=base_cfg.release_height_m,
            description="Height m",
        ),
    )
