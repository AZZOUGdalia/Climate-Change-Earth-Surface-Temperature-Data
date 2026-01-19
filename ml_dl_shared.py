from __future__ import annotations

"""
Shared preprocessing for the Climate Change temperature notebooks.

This module provides a single, consistent data preparation path that can be
imported from multiple notebooks. It loads the raw CSV files from the `archive`
folder, builds a cleaned monthly city table with numeric coordinates, aggregates
monthly observations into seasonal averages, and constructs a modelling dataset
that is ready for sequence forecasting.

The primary intent is reproducibility and consistency across notebooks. When
`Deep_Learning.ipynb` and `ML DL.ipynb` import from this file, they use the same
definitions of `city_clean`, `city_seasonal_clean`, and `dl_df`, which reduces
the risk of subtle drift caused by duplicated preprocessing code.
"""

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "archive"

FILES: Dict[str, str] = {
    "global": "GlobalTemperatures.csv",
    "country": "GlobalLandTemperaturesByCountry.csv",
    "state": "GlobalLandTemperaturesByState.csv",
    "major_city": "GlobalLandTemperaturesByMajorCity.csv",
    "city": "GlobalLandTemperaturesByCity.csv",
}


def _safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep=",", encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, sep=",", encoding="latin1")


def _add_time_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dt"] = pd.to_datetime(df["dt"], errors="coerce")
    df = df.dropna(subset=["dt"])
    df["year"] = df["dt"].dt.year.astype("Int64")
    df["month"] = df["dt"].dt.month.astype("Int64")
    return df


def _latlon_to_float(x: str | float) -> float:
    if pd.isna(x):
        return np.nan
    text = str(x).strip()
    sign = -1 if text and text[-1] in {"S", "W"} else 1
    try:
        return sign * float(text[:-1])
    except ValueError:
        return np.nan


def _load_raw_dfs() -> Dict[str, pd.DataFrame]:
    dfs: Dict[str, pd.DataFrame] = {}
    for name, filename in FILES.items():
        path = DATA_DIR / filename
        dfs[name] = _safe_read_csv(path)
    return dfs


def _build_city_clean(raw_city: pd.DataFrame) -> pd.DataFrame:
    city_clean = _add_time_cols(raw_city)
    city_clean = city_clean[
        [
            "dt",
            "year",
            "month",
            "City",
            "Country",
            "Latitude",
            "Longitude",
            "AverageTemperature",
            "AverageTemperatureUncertainty",
        ]
    ]
    city_clean = city_clean.dropna(subset=["AverageTemperature"])
    city_clean["lat"] = city_clean["Latitude"].apply(_latlon_to_float)
    city_clean["lon"] = city_clean["Longitude"].apply(_latlon_to_float)
    city_clean = city_clean.dropna(subset=["lat", "lon"])
    return city_clean


def _build_city_seasonal(city_clean: pd.DataFrame) -> pd.DataFrame:
    def month_to_season(m: int) -> str:  # noqa: D103
        if m in (12, 1, 2):
            return "Winter"
        if m in (3, 4, 5):
            return "Spring"
        if m in (6, 7, 8):
            return "Summer"
        return "Autumn"

    seasonal = city_clean.copy()
    seasonal["season"] = seasonal["month"].astype(int).apply(month_to_season)
    seasonal = (
        seasonal.groupby(["City", "Country", "year", "season"], as_index=False)
        .agg(
            {
                "AverageTemperature": "mean",
                "AverageTemperatureUncertainty": "mean",
                "lat": "first",
                "lon": "first",
            }
        )
    )

    base_mask = seasonal["year"].between(1951, 1980)
    climatology = (
        seasonal.loc[base_mask]
        .groupby(["City", "Country", "season"], as_index=False)["AverageTemperature"]
        .mean()
        .rename(columns={"AverageTemperature": "baseline_temp"})
    )
    seasonal = seasonal.merge(climatology, on=["City", "Country", "season"], how="left")
    seasonal["TempAnomaly"] = seasonal["AverageTemperature"] - seasonal["baseline_temp"]
    eps = 1e-6
    seasonal["sigma_dev"] = (
        seasonal["TempAnomaly"].abs()
        / (seasonal["AverageTemperatureUncertainty"] + eps)
    )
    return seasonal


def _build_dl_df(city_seasonal: pd.DataFrame) -> pd.DataFrame:
    required_cols = [
        "City",
        "Country",
        "year",
        "season",
        "AverageTemperature",
        "AverageTemperatureUncertainty",
        "lat",
        "lon",
    ]
    dl_df = city_seasonal[required_cols].copy()
    season_to_id = {"Winter": 0, "Spring": 1, "Summer": 2, "Autumn": 3}
    dl_df["season_id"] = dl_df["season"].map(season_to_id)
    non_missing = [
        "AverageTemperature",
        "AverageTemperatureUncertainty",
        "lat",
        "lon",
        "season_id",
        "year",
    ]
    dl_df = dl_df.dropna(subset=non_missing)
    dl_df["year"] = dl_df["year"].astype(int)
    dl_df["season_id"] = dl_df["season_id"].astype(int)
    dl_df["t_index"] = dl_df["year"] * 4 + dl_df["season_id"]
    dl_df["season_sin"] = np.sin(2 * np.pi * dl_df["season_id"] / 4.0)
    dl_df["season_cos"] = np.cos(2 * np.pi * dl_df["season_id"] / 4.0)

    base_mask = dl_df["year"].between(1951, 1980)
    clim_ref = (
        dl_df.loc[base_mask]
        .groupby(["City", "Country", "season_id"], as_index=False)["AverageTemperature"]
        .mean()
        .rename(columns={"AverageTemperature": "clim_mean_ref"})
    )
    dl_df = dl_df.merge(clim_ref, on=["City", "Country", "season_id"], how="left")
    dl_df["clim_mean_all"] = dl_df.groupby(
        ["City", "Country", "season_id"]
    )["AverageTemperature"].transform("mean")
    dl_df["clim_mean"] = dl_df["clim_mean_ref"].fillna(dl_df["clim_mean_all"])
    dl_df["temp_anomaly"] = dl_df["AverageTemperature"] - dl_df["clim_mean"]

    _year_mean = float(dl_df["year"].mean())
    _year_std = float(dl_df["year"].std())
    dl_df["year_norm"] = (dl_df["year"] - _year_mean) / (_year_std if _year_std else 1.0)

    per_country_cap = 6
    pair_counts = (
        dl_df.groupby(["Country", "City"])
        .size()
        .reset_index(name="n_obs")
        .sort_values(["Country", "n_obs"], ascending=[True, False])
    )
    top_pairs = (
        pair_counts.groupby("Country", as_index=False)
        .head(per_country_cap)[["Country", "City"]]
    )
    dl_df = dl_df.merge(top_pairs, on=["Country", "City"], how="inner")
    dl_df = dl_df.sort_values(["City", "Country", "t_index"]).reset_index(drop=True)
    return dl_df


_RAW = _load_raw_dfs()
city_clean = _build_city_clean(_RAW["city"])
city_seasonal_clean = _build_city_seasonal(city_clean)
dl_df = _build_dl_df(city_seasonal_clean)

TARGET_COL = "temp_anomaly"
PER_COUNTRY_CAP = 6

__all__ = [
    "DATA_DIR",
    "FILES",
    "city_clean",
    "city_seasonal_clean",
    "dl_df",
    "TARGET_COL",
    "PER_COUNTRY_CAP",
]
