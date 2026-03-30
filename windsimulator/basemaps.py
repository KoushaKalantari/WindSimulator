from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import math
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image


WEB_MERCATOR_MAX_LATITUDE = 85.05112878
BASEMAP_CACHE_TTL_S = 30 * 86400
BASEMAP_HTTP_RETRIES = 5
BASEMAP_HTTP_TIMEOUT_S = 30.0
BASEMAP_RETRY_BACKOFF_S = 1.5
BASEMAP_USER_AGENT = "WindSimulationNotebook/1.0"
BASEMAP_MIN_ZOOM = 3
BASEMAP_MAX_ZOOM = 16
BASEMAP_MAX_TILES = 128
_BASEMAP_CACHE_DIR = Path(__file__).resolve().parent.parent / ".windsimulator_cache" / "basemaps"


@dataclass(frozen=True)
class BasemapProvider:
    style: str
    title: str
    url_template: str
    attribution: str


@dataclass(frozen=True)
class BasemapImage:
    style: str
    title: str
    attribution: str
    zoom: int
    image_rgba: np.ndarray
    extent: tuple[float, float, float, float]


BASEMAP_PROVIDERS = {
    "roadmap": BasemapProvider(
        style="roadmap",
        title="Street Reference Map",
        url_template="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        attribution="Map data: OpenStreetMap contributors",
    ),
    "terrain_map": BasemapProvider(
        style="terrain_map",
        title="Topographic Reference Map",
        url_template=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Topo_Map/MapServer/tile/{z}/{y}/{x}"
        ),
        attribution="Map: Esri World Topographic Map and contributors",
    ),
    "terrain": BasemapProvider(
        style="terrain",
        title="Terrain Basemap",
        url_template="https://a.tile.opentopomap.org/{z}/{x}/{y}.png",
        attribution=(
            "Map data: OpenStreetMap contributors, SRTM | "
            "Cartography: OpenTopoMap (CC-BY-SA)"
        ),
    ),
    "satellite": BasemapProvider(
        style="satellite",
        title="Satellite Basemap",
        url_template=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        attribution="Imagery: Esri World Imagery and contributors",
    ),
    "carto_voyager": BasemapProvider(
        style="carto_voyager",
        title="Modern Street Map",
        url_template="https://a.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png",
        attribution="© OpenStreetMap contributors © CARTO",
    ),
}


def _clip_latitude(latitude: float) -> float:
    return max(-WEB_MERCATOR_MAX_LATITUDE, min(WEB_MERCATOR_MAX_LATITUDE, float(latitude)))


def _tile_x_for_longitude(longitude: float, zoom: int) -> float:
    return ((float(longitude) + 180.0) / 360.0) * (2**zoom)


def _tile_y_for_latitude(latitude: float, zoom: int) -> float:
    latitude = _clip_latitude(latitude)
    latitude_rad = math.radians(latitude)
    return (
        (1.0 - math.asinh(math.tan(latitude_rad)) / math.pi)
        / 2.0
        * (2**zoom)
    )


def _longitude_for_tile_x(tile_x: float, zoom: int) -> float:
    return (float(tile_x) / (2**zoom)) * 360.0 - 180.0


def _latitude_for_tile_y(tile_y: float, zoom: int) -> float:
    n = math.pi - (2.0 * math.pi * float(tile_y) / (2**zoom))
    return math.degrees(math.atan(math.sinh(n)))


def _tile_bounds(
    longitude_min: float,
    latitude_min: float,
    longitude_max: float,
    latitude_max: float,
    zoom: int,
) -> tuple[int, int, int, int]:
    max_index = (2**zoom) - 1
    x_min = int(math.floor(_tile_x_for_longitude(longitude_min, zoom)))
    x_max = int(math.floor(_tile_x_for_longitude(longitude_max, zoom)))
    y_min = int(math.floor(_tile_y_for_latitude(latitude_max, zoom)))
    y_max = int(math.floor(_tile_y_for_latitude(latitude_min, zoom)))
    return (
        max(0, min(max_index, x_min)),
        max(0, min(max_index, x_max)),
        max(0, min(max_index, y_min)),
        max(0, min(max_index, y_max)),
    )


def _select_zoom(
    longitude_min: float,
    latitude_min: float,
    longitude_max: float,
    latitude_max: float,
    max_tiles: int = BASEMAP_MAX_TILES,
) -> int:
    selected_zoom = BASEMAP_MIN_ZOOM
    for zoom in range(BASEMAP_MAX_ZOOM, BASEMAP_MIN_ZOOM - 1, -1):
        x_min, x_max, y_min, y_max = _tile_bounds(
            longitude_min,
            latitude_min,
            longitude_max,
            latitude_max,
            zoom,
        )
        tile_count = (x_max - x_min + 1) * (y_max - y_min + 1)
        if tile_count <= max_tiles:
            selected_zoom = zoom
            break
    return selected_zoom


def _tile_cache_path(provider_style: str, zoom: int, tile_x: int, tile_y: int) -> Path:
    return _BASEMAP_CACHE_DIR / provider_style / str(zoom) / str(tile_x) / f"{tile_y}.png"


def _load_cached_tile(provider_style: str, zoom: int, tile_x: int, tile_y: int) -> Image.Image | None:
    cache_path = _tile_cache_path(provider_style, zoom, tile_x, tile_y)
    if not cache_path.exists():
        return None
    if (time.time() - cache_path.stat().st_mtime) > BASEMAP_CACHE_TTL_S:
        try:
            cache_path.unlink()
        except OSError:
            pass
        return None
    try:
        return Image.open(cache_path).convert("RGBA")
    except Exception:
        try:
            cache_path.unlink()
        except OSError:
            pass
        return None


def _store_cached_tile(
    provider_style: str,
    zoom: int,
    tile_x: int,
    tile_y: int,
    image: Image.Image,
) -> None:
    cache_path = _tile_cache_path(provider_style, zoom, tile_x, tile_y)
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = cache_path.with_suffix(".tmp")
        image.save(temp_path, format="PNG")
        temp_path.replace(cache_path)
    except OSError:
        pass


def _fetch_tile_image(provider: BasemapProvider, zoom: int, tile_x: int, tile_y: int) -> Image.Image:
    cached = _load_cached_tile(provider.style, zoom, tile_x, tile_y)
    if cached is not None:
        return cached

    url = provider.url_template.format(z=zoom, x=tile_x, y=tile_y)
    request = Request(url, headers={"User-Agent": BASEMAP_USER_AGENT})
    last_error: Exception | None = None
    for attempt in range(BASEMAP_HTTP_RETRIES):
        try:
            with urlopen(request, timeout=BASEMAP_HTTP_TIMEOUT_S) as response:
                image = Image.open(BytesIO(response.read())).convert("RGBA")
                _store_cached_tile(provider.style, zoom, tile_x, tile_y, image)
                return image
        except (HTTPError, URLError, OSError) as exc:
            last_error = exc
            if attempt == BASEMAP_HTTP_RETRIES - 1:
                break
            time.sleep(BASEMAP_RETRY_BACKOFF_S * (2**attempt))
    raise RuntimeError(
        f"Unable to retrieve {provider.style} basemap tiles after {BASEMAP_HTTP_RETRIES} attempts."
    ) from last_error


def fetch_basemap_image(
    longitude_min: float,
    latitude_min: float,
    longitude_max: float,
    latitude_max: float,
    style: str,
) -> BasemapImage:
    if style not in BASEMAP_PROVIDERS:
        supported = ", ".join(sorted(BASEMAP_PROVIDERS))
        raise ValueError(f"Unsupported basemap style {style!r}. Choose from: {supported}.")

    provider = BASEMAP_PROVIDERS[style]
    zoom = _select_zoom(
        longitude_min=longitude_min,
        latitude_min=latitude_min,
        longitude_max=longitude_max,
        latitude_max=latitude_max,
    )
    x_min, x_max, y_min, y_max = _tile_bounds(
        longitude_min,
        latitude_min,
        longitude_max,
        latitude_max,
        zoom,
    )
    tiles: list[list[Image.Image]] = []
    tile_width = tile_height = 256
    for tile_y in range(y_min, y_max + 1):
        row_tiles: list[Image.Image] = []
        for tile_x in range(x_min, x_max + 1):
            tile_image = _fetch_tile_image(provider, zoom, tile_x, tile_y)
            tile_width, tile_height = tile_image.size
            row_tiles.append(tile_image)
        tiles.append(row_tiles)

    mosaic_width = (x_max - x_min + 1) * tile_width
    mosaic_height = (y_max - y_min + 1) * tile_height
    mosaic = Image.new("RGBA", (mosaic_width, mosaic_height))
    for row_index, row_tiles in enumerate(tiles):
        for column_index, tile_image in enumerate(row_tiles):
            mosaic.paste(tile_image, (column_index * tile_width, row_index * tile_height))

    extent = (
        _longitude_for_tile_x(x_min, zoom),
        _longitude_for_tile_x(x_max + 1, zoom),
        _latitude_for_tile_y(y_max + 1, zoom),
        _latitude_for_tile_y(y_min, zoom),
    )
    return BasemapImage(
        style=provider.style,
        title=provider.title,
        attribution=provider.attribution,
        zoom=zoom,
        image_rgba=np.asarray(mosaic),
        extent=extent,
    )
