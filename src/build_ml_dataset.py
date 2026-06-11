"""Build a station-day dataset ready for machine learning."""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

from .config import PipelineConfig

LOGGER = logging.getLogger(__name__)


NUMERIC_FEATURES_TO_NORMALIZE = [
    "latitude",
    "longitude",
    "altitude",
    "wind_daily_mean_ms",
    "wind_daily_std_ms",
    "solar_daily_kwh_m2_day",
    "missing_rate_wind",
    "missing_rate_solar",
]


def build_ml_dataset(daily: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Create a compact ML-friendly station-day table."""

    df = daily.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["day_of_year"] = df["date"].dt.dayofyear
    df["month_sin"] = np.sin(2 * math.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * math.pi * df["month"] / 12.0)
    df["day_of_year_sin"] = np.sin(2 * math.pi * df["day_of_year"] / 366.0)
    df["day_of_year_cos"] = np.cos(2 * math.pi * df["day_of_year"] / 366.0)

    columns = [
        "station_code",
        "latitude",
        "longitude",
        "altitude",
        "year",
        "month",
        "day_of_year",
        "wind_daily_mean_ms",
        "wind_daily_std_ms",
        "solar_daily_kwh_m2_day",
        "missing_rate_wind",
        "missing_rate_solar",
        "month_sin",
        "month_cos",
        "day_of_year_sin",
        "day_of_year_cos",
    ]
    ml = df[columns].sort_values(["station_code", "year", "day_of_year"])
    config.gold_dir.mkdir(parents=True, exist_ok=True)
    output = config.gold_dir / "ml_dataset_station_day.csv"
    ml.to_csv(output, index=False)
    normalized = normalize_ml_dataset(ml, config)
    LOGGER.info("Dataset ML salvo em %s (%s linhas)", output, len(ml))
    LOGGER.info("Dataset ML normalizado salvo com %s linhas", len(normalized))
    return ml


def normalize_ml_dataset(ml: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Create a min-max normalized gold dataset and persist scaler parameters."""

    normalized = ml.copy()
    params = []
    for column in NUMERIC_FEATURES_TO_NORMALIZE:
        series = pd.to_numeric(normalized[column], errors="coerce")
        min_value = series.min(skipna=True)
        max_value = series.max(skipna=True)
        params.append({"feature": column, "method": "min_max", "min": min_value, "max": max_value})
        if pd.isna(min_value) or pd.isna(max_value) or max_value == min_value:
            normalized[f"{column}_norm"] = 0.0
        else:
            normalized[f"{column}_norm"] = (series - min_value) / (max_value - min_value)

    normalized_output = config.gold_dir / "ml_dataset_station_day_normalized.csv"
    params_output = config.gold_dir / "normalization_params.csv"
    normalized.to_csv(normalized_output, index=False)
    pd.DataFrame(params).to_csv(params_output, index=False)
    return normalized
