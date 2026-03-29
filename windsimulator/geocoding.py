from __future__ import annotations

import json
import os
import ssl
import time
from dataclasses import dataclass
from functools import lru_cache
from urllib.parse import urlencode
from urllib.request import Request, urlopen

try:
    import certifi
except Exception:
    certifi = None


NOMINATIM_REVERSE_URL = "https://nominatim.openstreetmap.org/reverse"
DEFAULT_USER_AGENT = "WindSimulator/0.1 (prototype hazard demo)"
_LAST_NOMINATIM_REQUEST_TS = 0.0


@dataclass(frozen=True)
class ReverseGeocodeResult:
    latitude: float
    longitude: float
    display_name: str
    neighborhood: str | None
    city: str | None
    postcode: str | None
    state: str | None
    country: str | None


def _ssl_context():
    return ssl.create_default_context(cafile=certifi.where() if certifi is not None else None)


def _nominatim_headers() -> dict[str, str]:
    return {
        "User-Agent": os.environ.get("NOMINATIM_USER_AGENT", DEFAULT_USER_AGENT),
        "Accept-Language": "en",
    }


def _respect_nominatim_rate_limit() -> None:
    global _LAST_NOMINATIM_REQUEST_TS
    now = time.time()
    wait_s = 1.0 - (now - _LAST_NOMINATIM_REQUEST_TS)
    if wait_s > 0:
        time.sleep(wait_s)
    _LAST_NOMINATIM_REQUEST_TS = time.time()


def _pick_first(address: dict, *keys: str) -> str | None:
    for key in keys:
        value = address.get(key)
        if value:
            return str(value)
    return None


@lru_cache(maxsize=256)
def reverse_geocode(latitude: float, longitude: float) -> ReverseGeocodeResult:
    params = {
        "lat": f"{latitude:.6f}",
        "lon": f"{longitude:.6f}",
        "format": "jsonv2",
        "addressdetails": 1,
        "zoom": 18,
    }
    email = os.environ.get("NOMINATIM_EMAIL")
    if email:
        params["email"] = email
    url = f"{NOMINATIM_REVERSE_URL}?{urlencode(params)}"
    request = Request(url, headers=_nominatim_headers())
    _respect_nominatim_rate_limit()
    with urlopen(request, context=_ssl_context()) as response:
        payload = json.loads(response.read().decode("utf-8"))

    address = payload.get("address", {})
    return ReverseGeocodeResult(
        latitude=latitude,
        longitude=longitude,
        display_name=str(payload.get("display_name", "")),
        neighborhood=_pick_first(address, "neighbourhood", "suburb", "quarter", "hamlet"),
        city=_pick_first(address, "city", "town", "municipality", "village", "county"),
        postcode=_pick_first(address, "postcode"),
        state=_pick_first(address, "state"),
        country=_pick_first(address, "country"),
    )
