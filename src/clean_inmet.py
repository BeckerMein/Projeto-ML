"""Selection and cleaning routines for standardized INMET hourly data."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import PipelineConfig

LOGGER = logging.getLogger(__name__)


def build_station_catalog(hourly: pd.DataFrame) -> pd.DataFrame:
    """Create one row per station with availability metadata."""

    catalog = (
        hourly.groupby("station_code", dropna=False)
        .agg(
            station_name=("station_name", "first"),
            city=("city", "first"),
            state=("state", "first"),
            latitude=("latitude", "first"),
            longitude=("longitude", "first"),
            altitude=("altitude", "first"),
            first_year_available=("year", "min"),
            last_year_available=("year", "max"),
        )
        .reset_index()
    )
    return catalog


def select_stations(catalog: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Select stations by UF or by configurable bounding box."""

    mode = config.mode.lower()
    if mode == "uf":
        selected = catalog[catalog["state"].astype(str).str.upper() == config.region.upper()].copy()
    elif mode == "buffer":
        bbox = config.bbox
        selected = catalog[
            catalog["latitude"].between(bbox["min_lat"], bbox["max_lat"], inclusive="both")
            & catalog["longitude"].between(bbox["min_lon"], bbox["max_lon"], inclusive="both")
        ].copy()
    else:
        raise ValueError("Modo geografico invalido. Use 'uf' ou 'buffer'.")

    selected = selected.sort_values(["state", "city", "station_code"])
    output = config.processed_dir / "stations_selected.csv"
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output, index=False)
    LOGGER.info("Estacoes selecionadas salvas em %s (%s estacoes)", output, len(selected))
    return selected


def apply_geographic_selection(hourly: pd.DataFrame, config: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter hourly data to selected stations and write stations_selected.csv."""

    catalog = build_station_catalog(hourly)
    selected = select_stations(catalog, config)
    selected_codes = set(selected["station_code"].dropna().astype(str))
    filtered = hourly[hourly["station_code"].astype(str).isin(selected_codes)].copy()
    if filtered.empty:
        raise RuntimeError("A selecao geografica nao retornou dados horarios.")
    return filtered, selected


def flag_outliers(group: pd.DataFrame, column: str, config: PipelineConfig) -> pd.Series:
    """Return a boolean outlier mask for one station-variable group."""

    values = group[column]
    if config.outlier_method == "none":
        return pd.Series(False, index=group.index)
    if values.notna().sum() < 8:
        return pd.Series(False, index=group.index)
    if config.outlier_method == "iqr":
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        if not np.isfinite(iqr) or iqr == 0:
            return pd.Series(False, index=group.index)
        lower = q1 - config.iqr_multiplier * iqr
        upper = q3 + config.iqr_multiplier * iqr
        return (values < lower) | (values > upper)
    if config.outlier_method == "zscore":
        mean = values.mean()
        std = values.std()
        if not np.isfinite(std) or std == 0:
            return pd.Series(False, index=group.index)
        return ((values - mean).abs() / std) > config.zscore_threshold
    raise ValueError("Metodo de outlier invalido. Use iqr, zscore ou none.")


def apply_outlier_flags(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Flag outliers per station for wind and solar variables."""

    df["flag_outlier_wind"] = False
    df["flag_outlier_solar"] = False
    for _, group in df.groupby("station_code", dropna=False):
        wind_mask = flag_outliers(group, "wind_speed_10m_ms", config)
        solar_mask = flag_outliers(group, "solar_radiation_kj_m2", config)
        df.loc[group.index, "flag_outlier_wind"] = wind_mask.fillna(False)
        df.loc[group.index, "flag_outlier_solar"] = solar_mask.fillna(False)
    return df


def clean_hourly_data(hourly: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Clean hourly INMET data, add quality flags and write parquet outputs."""

    df = hourly.copy()
    for column in ["wind_speed_10m_ms", "solar_radiation_kj_m2", "latitude", "longitude", "altitude"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce")
    df["timestamp_local"] = pd.to_datetime(df["timestamp_local"], errors="coerce")
    df = df.dropna(subset=["timestamp_utc", "timestamp_local"])
    df["date"] = df["timestamp_local"].dt.date
    df["year"] = df["timestamp_local"].dt.year
    df["month"] = df["timestamp_local"].dt.month
    df["day"] = df["timestamp_local"].dt.day
    df["hour"] = df["timestamp_local"].dt.hour

    df["flag_missing_wind"] = df["wind_speed_10m_ms"].isna()
    df["flag_missing_solar"] = df["solar_radiation_kj_m2"].isna()
    df["flag_invalid_wind"] = (df["wind_speed_10m_ms"] < 0) | (df["wind_speed_10m_ms"] > config.wind_max_ms)
    df["flag_invalid_solar"] = (df["solar_radiation_kj_m2"] < 0) | (
        df["solar_radiation_kj_m2"] > config.solar_max_kj_m2
    )

    df.loc[df["flag_invalid_wind"], "wind_speed_10m_ms"] = np.nan
    df.loc[df["flag_invalid_solar"], "solar_radiation_kj_m2"] = np.nan

    night_mask = (df["hour"] < config.solar_day_start_hour) | (df["hour"] > config.solar_day_end_hour)
    if config.nighttime_solar_policy == "zero":
        df.loc[night_mask & df["solar_radiation_kj_m2"].notna(), "solar_radiation_kj_m2"] = 0.0
    elif config.nighttime_solar_policy == "nan":
        df.loc[night_mask, "solar_radiation_kj_m2"] = np.nan
    else:
        raise ValueError("nighttime_solar_policy deve ser 'nan' ou 'zero'.")

    df["flag_missing_wind"] = df["wind_speed_10m_ms"].isna()
    df["flag_missing_solar"] = df["solar_radiation_kj_m2"].isna()
    df = apply_outlier_flags(df, config)

    clean = df.copy()
    if config.drop_outliers_in_clean:
        clean.loc[clean["flag_outlier_wind"], "wind_speed_10m_ms"] = np.nan
        clean.loc[clean["flag_outlier_solar"], "solar_radiation_kj_m2"] = np.nan

    config.processed_dir.mkdir(parents=True, exist_ok=True)
    flagged_path = config.processed_dir / "inmet_pe_hourly_flagged.parquet"
    clean_path = config.processed_dir / "inmet_pe_hourly_clean.parquet"
    df.to_parquet(flagged_path, index=False)
    clean.to_parquet(clean_path, index=False)
    LOGGER.info("Base horaria com flags salva em %s", flagged_path)
    LOGGER.info("Base horaria limpa salva em %s", clean_path)
    return clean

