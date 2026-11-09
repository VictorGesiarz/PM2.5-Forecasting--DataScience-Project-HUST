"""Access to the OpenAQ public AWS archive (no API key required).

OpenAQ publishes every measurement it ingests in a public S3 bucket,
`openaq-data-archive`, one gzipped CSV per location and per day:

    records/csv.gz/locationid={id}/year={yyyy}/month={mm}/location-{id}-{yyyymmdd}.csv.gz

Each file has the columns
    location_id, sensors_id, location, datetime, lat, lon, parameter, units, value
where `datetime` is ISO 8601 with the local UTC offset (+07:00 for Hanoi)
and marks the END of the averaging hour (OpenAQ convention).

This is the same data served by the API v3 endpoint
/v3/sensors/{id}/measurements, but it can be bulk-downloaded without an
X-API-Key header and without the 60 req/min limit. If an API key is
available (env var OPENAQ_API_KEY), `api_locations_near` can be used to
cross-check the station metadata.
"""
from __future__ import annotations

import gzip
import io
import os
import re
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

BUCKET_URL = "https://openaq-data-archive.s3.amazonaws.com"
PREFIX = "records/csv.gz"

# Hanoi locations found in the archive (see notebooks/01_station_inventory.ipynb).
# 2161290-2161323: Hanoi government network (Sở TN&MT / CEM), data from Jan 2024.
# 4946811-4946813: Hanoi Air Quality Monitoring Network, data from Jul 2025.
# 7441: "Hanoi" (US Embassy monitor, AirNow feed). 2539: old US Diplomatic Post id.
HANOI_LOCATION_IDS = (
    list(range(2161290, 2161324)) + [4946811, 4946812, 4946813] + [7441, 2539]
)


def _get(url: str, retries: int = 4, timeout: int = 60) -> bytes:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except Exception:  # network hiccup -> exponential backoff
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def list_keys(location_id: int) -> list[str]:
    """List every daily file key for one location (handles S3 pagination)."""
    keys: list[str] = []
    token = None
    while True:
        params = {"list-type": "2", "prefix": f"{PREFIX}/locationid={location_id}/"}
        if token:
            params["continuation-token"] = token
        xml = _get(f"{BUCKET_URL}/?{urllib.parse.urlencode(params)}").decode()
        keys += re.findall(r"<Key>(.*?)</Key>", xml)
        m = re.search(r"<NextContinuationToken>(.*?)</NextContinuationToken>", xml)
        if not m:
            return keys
        token = m.group(1)


def key_date(key: str) -> pd.Timestamp:
    return pd.to_datetime(re.search(r"-(\d{8})\.csv\.gz$", key).group(1), format="%Y%m%d")


def file_inventory(location_ids=HANOI_LOCATION_IDS, workers: int = 16) -> pd.DataFrame:
    """One row per (location, daily file): cheap inventory from S3 listings only."""
    with ThreadPoolExecutor(workers) as ex:
        results = list(ex.map(list_keys, location_ids))
    rows = [
        {"location_id": loc, "key": k, "date": key_date(k)}
        for loc, keys in zip(location_ids, results)
        for k in keys
    ]
    return pd.DataFrame(rows)


def read_day(key: str, cache_dir: Path | None = None) -> pd.DataFrame:
    """Download (or read from cache) one daily file, untouched (raw)."""
    if cache_dir is not None:
        path = Path(cache_dir) / key
        if path.exists():
            raw = path.read_bytes()
        else:
            raw = _get(f"{BUCKET_URL}/{key}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
    else:
        raw = _get(f"{BUCKET_URL}/{key}")
    return pd.read_csv(io.BytesIO(gzip.decompress(raw)), dtype={"value": "float64"})


def download_location(
    location_id: int,
    cache_dir: Path,
    start: str | None = None,
    end: str | None = None,
    parameters: tuple[str, ...] | None = ("pm25",),
    workers: int = 16,
) -> pd.DataFrame:
    """Download all daily files of a location (raw files cached on disk).

    Returns the concatenated measurements, optionally filtered on parameter.
    `start`/`end` (inclusive, local dates) restrict the files fetched.
    """
    keys = list_keys(location_id)
    if start:
        keys = [k for k in keys if key_date(k) >= pd.Timestamp(start)]
    if end:
        keys = [k for k in keys if key_date(k) <= pd.Timestamp(end)]
    with ThreadPoolExecutor(workers) as ex:
        frames = list(ex.map(lambda k: read_day(k, cache_dir), keys))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if parameters:
        df = df[df["parameter"].isin(parameters)]
    return df.reset_index(drop=True)


def to_hourly_utc(raw: pd.DataFrame) -> pd.DataFrame:
    """Parse timestamps to UTC and drop exact duplicate records."""
    df = raw.copy()
    df["datetime_utc"] = pd.to_datetime(df["datetime"], utc=True)
    return df.drop_duplicates(["sensors_id", "parameter", "datetime_utc"]).sort_values(
        "datetime_utc"
    )


def api_locations_near(lat: float, lon: float, radius_m: int = 25000) -> list[dict]:
    """Optional cross-check through OpenAQ API v3 (needs OPENAQ_API_KEY)."""
    key = os.environ.get("OPENAQ_API_KEY")
    if not key:
        raise RuntimeError("Set OPENAQ_API_KEY to use the OpenAQ API v3.")
    import json

    url = (
        "https://api.openaq.org/v3/locations?"
        + urllib.parse.urlencode(
            {"coordinates": f"{lat},{lon}", "radius": radius_m, "limit": 1000}
        )
    )
    req = urllib.request.Request(url, headers={"X-API-Key": key})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)["results"]
