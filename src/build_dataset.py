"""Build the hourly, daily and feature datasets of one or more stations.

Usage:  uv run python src/build_dataset.py [location_id ...]   (default: 7441 2161292)

Requires data/interim/openaq_pm25_raw.parquet (src/download_openaq.py).
Outputs, per station, in data/processed/:
    station_{id}_hourly.parquet   regular hourly grid (UTC), PM2.5 + weather, NaN = missing
    station_{id}_daily.csv        local-day PM2.5 mean, hour count, validity flag, weather
    station_{id}_excluded_days.csv  days with < 18 valid hours
    station_{id}_features.csv     model table: issue day t, features, target = PM2.5(t+1)
    station_{id}_report.json      cleaning counts
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from openmeteo import fetch_weather  # noqa: E402
from preprocessing import (  # noqa: E402
    assert_no_future_leak,
    build_features,
    clean_hourly_pm25,
    daily_mean,
    daily_weather,
)

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
RAW_WEATHER = ROOT / "data" / "raw" / "openmeteo"
DEFAULT_STATIONS = [7441, 2161292]


def build_station(loc: int, raw: pd.DataFrame, meta: pd.DataFrame) -> dict:
    m = meta.loc[loc]
    hourly, report = clean_hourly_pm25(raw[raw["location_id"] == loc])
    report.update(location_id=loc, name=m["name"], lat=m["lat"], lon=m["lon"])

    # weather on the station's coordinates, over the station's span (+1 day margin)
    start = (hourly.index.min() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    end = (hourly.index.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    weather = fetch_weather(
        m["lat"], m["lon"], start, end,
        raw_path=RAW_WEATHER / f"weather_{loc}_{start}_{end}.json",
    )

    hourly_df = hourly.to_frame().join(weather, how="left")
    hourly_df.to_parquet(PROCESSED / f"station_{loc}_hourly.parquet")

    pm_daily = daily_mean(hourly)
    w_daily = daily_weather(weather).reindex(pm_daily.index)
    daily = pm_daily.rename(columns={"mean": "pm25"}).join(w_daily)
    daily.to_csv(PROCESSED / f"station_{loc}_daily.csv")
    excluded = daily.loc[~daily["valid"], ["n_hours"]]
    excluded.to_csv(PROCESSED / f"station_{loc}_excluded_days.csv")

    feats = build_features(daily["pm25"], w_daily)
    assert_no_future_leak(feats, daily["pm25"])
    feats.to_csv(PROCESSED / f"station_{loc}_features.csv")

    usable = feats.dropna()
    report.update(
        days_total=len(daily),
        days_valid=int(daily["valid"].sum()),
        days_excluded=len(excluded),
        feature_rows_complete=len(usable),
        first_day=str(daily.index.min().date()),
        last_day=str(daily.index.max().date()),
    )
    (PROCESSED / f"station_{loc}_report.json").write_text(json.dumps(report, indent=2, default=str))
    return report


def main(ids: list[int]) -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    raw = pd.read_parquet(INTERIM / "openaq_pm25_raw.parquet")
    meta = pd.read_csv(INTERIM / "openaq_locations.csv").set_index("location_id")
    for loc in ids:
        rep = build_station(loc, raw, meta)
        print(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or DEFAULT_STATIONS)
