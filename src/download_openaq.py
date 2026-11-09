"""Download raw PM2.5 records of every Hanoi OpenAQ location.

Usage:  uv run python src/download_openaq.py [location_id ...]

Raw daily .csv.gz files are cached untouched in data/raw/openaq/ (re-runs only
fetch new days). The PM2.5 records of all locations are concatenated into
data/interim/openaq_pm25_raw.parquet, and the station metadata into
data/interim/openaq_locations.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from openaq_archive import HANOI_LOCATION_IDS, download_location  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "openaq"
INTERIM = ROOT / "data" / "interim"


def main(ids: list[int]) -> None:
    INTERIM.mkdir(parents=True, exist_ok=True)
    frames, meta = [], []
    for loc in ids:
        df = download_location(loc, RAW, parameters=None)
        if df.empty:
            print(f"{loc}: no files")
            continue
        first = df.iloc[0]
        params = sorted(df["parameter"].unique())
        meta.append(
            {
                "location_id": loc,
                "name": str(first["location"]).rsplit("-", 1)[0],
                "lat": float(first["lat"]),
                "lon": float(first["lon"]),
                "parameters": ",".join(params),
                "pm25_sensor_id": (
                    int(df.loc[df["parameter"] == "pm25", "sensors_id"].mode()[0])
                    if "pm25" in params else None
                ),
            }
        )
        pm = df[df["parameter"] == "pm25"]
        print(f"{loc} {meta[-1]['name']}: {len(df)} records, {len(pm)} pm25")
        frames.append(pm)
    out = pd.concat(frames, ignore_index=True)
    out["value"] = out["value"].astype(float)
    out["datetime_utc"] = pd.to_datetime(out["datetime"], utc=True)
    out.to_parquet(INTERIM / "openaq_pm25_raw.parquet", index=False)
    pd.DataFrame(meta).to_csv(INTERIM / "openaq_locations.csv", index=False)
    print("saved", len(out), "pm25 records")


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or HANOI_LOCATION_IDS)
