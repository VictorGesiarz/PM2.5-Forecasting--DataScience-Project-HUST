"""Cleaning, regular time grid, daily aggregation and feature construction.

Conventions (decided in notebooks/02_build_dataset.ipynb):
* All raw timestamps are converted to UTC, then to local time Asia/Bangkok
  (UTC+7, no DST) for the day boundary.
* OpenAQ timestamps mark the END of the averaging hour. We shift them by -1 h
  so that every hourly value is labelled by the START of its hour; the local
  day D then contains the hours [D 00:00, D 24:00).
* Missing hours are never interpolated: they become NaN on a regular hourly
  grid. A day is valid if at least MIN_HOURS (18) of its 24 hours are present.
* Lag / rolling features are computed on the regular DAILY grid (shift by
  calendar days, not by rows) so they never jump over gaps and never use the
  future.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LOCAL_TZ = "Asia/Bangkok"
MIN_HOURS = 18
PM25_MAX_PLAUSIBLE = 1000.0  # µg/m³, above any physically plausible hourly mean

# Lunar new year (Tết) first day, used for the Tết-period calendar feature.
TET_DATES = pd.to_datetime(
    [
        "2016-02-08", "2017-01-28", "2018-02-16", "2019-02-05", "2020-01-25",
        "2021-02-12", "2022-02-01", "2023-01-22", "2024-02-10", "2025-01-29",
        "2026-02-17", "2027-02-06",
    ]
)

# US EPA PM2.5 AQI breakpoints (24-h mean, 2024 revision), upper bounds in µg/m³.
AQI_BINS = [-np.inf, 9.0, 35.4, 55.4, 125.4, 225.4, np.inf]
AQI_LABELS = [
    "Good",
    "Moderate",
    "Unhealthy for sensitive groups",
    "Unhealthy",
    "Very unhealthy",
    "Hazardous",
]


# --------------------------------------------------------------------------- #
# Hourly cleaning
# --------------------------------------------------------------------------- #
def clean_hourly_pm25(
    raw: pd.DataFrame, value_col: str = "value", time_col: str = "datetime_utc",
    period_end: bool = True, stuck_hours: int = 12,
) -> tuple[pd.Series, dict]:
    """From raw records to a regular hourly UTC series (hour-start labels).

    Returns the series and a dict counting what each rule removed.
    Rules: negative / sentinel values (-999), values above PM25_MAX_PLAUSIBLE,
    duplicates (mean of duplicates), and runs of >= `stuck_hours` identical
    consecutive values (frozen sensor).
    """
    df = raw[[time_col, value_col]].rename(columns={time_col: "t", value_col: "v"})
    report = {"raw_records": len(df)}
    neg = df["v"] < 0
    high = df["v"] > PM25_MAX_PLAUSIBLE
    report["negative_or_sentinel"] = int(neg.sum())
    report["above_max"] = int(high.sum())
    df = df[~neg & ~high & df["v"].notna()]
    if period_end:
        # timestamps mark the END of the averaging period, whose length is the
        # typical spacing between records (1 h for most stations, a few
        # minutes for the 2025 network) -> shift to period start before flooring
        step = df["t"].sort_values().diff().median()
        step = min(step, pd.Timedelta(hours=1)) if pd.notna(step) else pd.Timedelta(hours=1)
        report["record_interval"] = str(step)
        df = df.assign(t=df["t"] - step)
    df = df.assign(t=df["t"].dt.floor("h"))
    n_before = len(df)
    s = df.groupby("t")["v"].mean()
    report["duplicates_merged"] = n_before - len(s)

    grid = pd.date_range(s.index.min(), s.index.max(), freq="h", tz="UTC")
    s = s.reindex(grid)
    s.index.name = "datetime_utc"

    # frozen sensor: same value repeated for many consecutive present hours
    run_id = (s != s.shift()).cumsum()
    run_len = s.groupby(run_id).transform("size")
    stuck = s.notna() & (run_len >= stuck_hours)
    report["stuck_values"] = int(stuck.sum())
    s[stuck] = np.nan

    report["grid_hours"] = len(s)
    report["valid_hours"] = int(s.notna().sum())
    report["coverage"] = report["valid_hours"] / report["grid_hours"]
    return s.rename("pm25"), report


# --------------------------------------------------------------------------- #
# Daily aggregation
# --------------------------------------------------------------------------- #
def to_local(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return index.tz_convert(LOCAL_TZ)


def daily_mean(hourly: pd.Series, min_hours: int = MIN_HOURS) -> pd.DataFrame:
    """Local-day mean of an hourly UTC series; invalid days -> NaN."""
    s = hourly.copy()
    s.index = to_local(s.index)
    g = s.groupby(s.index.tz_localize(None).normalize())
    out = pd.DataFrame({"mean": g.mean(), "n_hours": g.count()})
    full = pd.date_range(out.index.min(), out.index.max(), freq="D")
    out = out.reindex(full)
    out["n_hours"] = out["n_hours"].fillna(0).astype(int)
    out["valid"] = out["n_hours"] >= min_hours
    out.loc[~out["valid"], "mean"] = np.nan
    out.index.name = "date"
    return out


def daily_weather(weather: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly Open-Meteo weather (UTC index) to local days."""
    w = weather.copy()
    w.index = to_local(w.index).tz_localize(None)
    # wind as u/v components so that the daily mean direction is meaningful
    rad = np.deg2rad(w["wind_direction_10m"])
    w["wind_u"] = -w["wind_speed_10m"] * np.sin(rad)
    w["wind_v"] = -w["wind_speed_10m"] * np.cos(rad)
    day = w.index.normalize()
    agg = {
        "temperature_2m": ["mean", "min", "max"],
        "relative_humidity_2m": "mean",
        "dew_point_2m": "mean",
        "precipitation": "sum",
        "surface_pressure": "mean",
        "pressure_msl": "mean",
        "wind_speed_10m": ["mean", "max"],
        "wind_u": "mean",
        "wind_v": "mean",
        "cloud_cover": "mean",
        "boundary_layer_height": ["mean", "min"],
    }
    agg = {k: v for k, v in agg.items() if k in w.columns}
    d = w.groupby(day).agg(agg)
    d.columns = ["_".join(c) if c[1] != "mean" else c[0] for c in d.columns]
    d = d.rename(
        columns={
            "temperature_2m": "temp", "temperature_2m_min": "temp_min",
            "temperature_2m_max": "temp_max", "relative_humidity_2m": "rh",
            "dew_point_2m": "dew_point", "precipitation_sum": "precip",
            "surface_pressure": "pressure", "pressure_msl": "pressure_msl",
            "wind_speed_10m": "wind_speed", "wind_speed_10m_max": "wind_speed_max",
            "cloud_cover": "cloud_cover", "boundary_layer_height": "blh",
            "boundary_layer_height_min": "blh_min",
        }
    )
    d["temp_range"] = d["temp_max"] - d["temp_min"]
    d["pressure_change"] = d["pressure_msl"].diff()  # day t vs day t-1: past only
    d.index.name = "date"
    return d


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def calendar_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    f = pd.DataFrame(index=dates)
    f["month"] = dates.month
    f["dow"] = dates.dayofweek
    f["is_weekend"] = (dates.dayofweek >= 5).astype(int)
    doy = dates.dayofyear
    f["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    # signed distance (days) to the nearest Tết; Tết period = [-7, +7] days
    diffs = (dates.values[:, None] - TET_DATES.values[None, :]).astype("timedelta64[D]").astype(int)
    nearest = diffs[np.arange(len(dates)), np.abs(diffs).argmin(axis=1)]
    f["tet_period"] = (np.abs(nearest) <= 7).astype(int)
    f["is_dry_season"] = dates.month.isin([11, 12, 1, 2, 3, 4]).astype(int)
    return f


def build_features(daily_pm25: pd.Series, weather_daily: pd.DataFrame | None,
                   n_lags: int = 7) -> pd.DataFrame:
    """Feature table indexed by issue day t; target = PM2.5 mean of day t+1.

    All shifts are on the regular daily grid, so a missing day stays missing
    in every lag that refers to it (no misalignment after gaps).
    """
    pm = daily_pm25.asfreq("D")
    X = pd.DataFrame(index=pm.index)
    for k in range(n_lags):  # lag0 = today's mean (known at end of day t)
        X[f"pm25_lag{k}"] = pm.shift(k)
    X["pm25_roll3"] = pm.rolling(3, min_periods=3).mean()
    X["pm25_roll7"] = pm.rolling(7, min_periods=5).mean()
    X["pm25_std7"] = pm.rolling(7, min_periods=5).std()
    X["pm25_diff1"] = X["pm25_lag0"] - X["pm25_lag1"]
    if weather_daily is not None:
        X = X.join(weather_daily, how="left")
    X = X.join(calendar_features(X.index))
    X["target"] = pm.shift(-1)
    X["target_date"] = X.index + pd.Timedelta(days=1)
    return X


def assert_no_future_leak(X: pd.DataFrame, daily_pm25: pd.Series, n_lags: int = 7) -> None:
    """Check that every lag feature equals the PM2.5 of a day <= t."""
    pm = daily_pm25.asfreq("D")
    for k in range(n_lags):
        expected = pm.reindex(X.index - pd.Timedelta(days=k)).values
        got = X[f"pm25_lag{k}"].values
        both = ~np.isnan(expected) | ~np.isnan(got)
        assert np.allclose(np.nan_to_num(got[both]), np.nan_to_num(expected[both])), k
    tgt = pm.reindex(X.index + pd.Timedelta(days=1)).values
    m = ~np.isnan(tgt)
    assert np.allclose(X["target"].values[m], tgt[m])
    # rolling windows end at day t (inclusive): recompute roll3 explicitly
    r3 = np.array([pm.reindex(pd.date_range(d - pd.Timedelta(days=2), d)).mean()
                   if pm.reindex(pd.date_range(d - pd.Timedelta(days=2), d)).notna().all()
                   else np.nan for d in X.index])
    m = ~np.isnan(r3)
    assert np.allclose(X["pm25_roll3"].values[m], r3[m])


def aqi_category(pm25) -> pd.Series:
    return pd.cut(pd.Series(pm25), AQI_BINS, labels=AQI_LABELS)
