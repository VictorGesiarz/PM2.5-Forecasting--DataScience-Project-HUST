"""Hourly weather from the Open-Meteo Historical Weather API (no key needed).

Endpoint: https://archive-api.open-meteo.com/v1/archive
Timestamps are requested in GMT/UTC (Open-Meteo default) and converted later.
Wind speed is requested in m/s (the API default is km/h), which removes the
unit ambiguity found in the Kaggle file.
Open-Meteo hourly values are instantaneous at the hour, except precipitation
which is the sum over the PRECEDING hour.
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "surface_pressure",
    "pressure_msl",
    "wind_speed_10m",
    "wind_direction_10m",
    "cloud_cover",
    "boundary_layer_height",
]


def fetch_weather(
    lat: float,
    lon: float,
    start: str,
    end: str,
    raw_path: Path | None = None,
    hourly=HOURLY_VARS,
) -> pd.DataFrame:
    """Return hourly weather indexed by UTC timestamp; keeps the raw JSON."""
    if raw_path is not None and Path(raw_path).exists():
        payload = json.loads(Path(raw_path).read_text())
    else:
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start,
            "end_date": end,
            "hourly": ",".join(hourly),
            "wind_speed_unit": "ms",
            "timezone": "GMT",
        }
        url = f"{ARCHIVE_URL}?{urllib.parse.urlencode(params)}"
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout=120) as r:
                    payload = json.load(r)
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(5 * (attempt + 1))
        if raw_path is not None:
            Path(raw_path).parent.mkdir(parents=True, exist_ok=True)
            Path(raw_path).write_text(json.dumps(payload))
    df = pd.DataFrame(payload["hourly"])
    df["datetime_utc"] = pd.to_datetime(df.pop("time"), utc=True)
    return df.set_index("datetime_utc")
