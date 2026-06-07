"""Build a station-day dataset ready for machine learning."""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

from .config import PipelineConfig

LOGGER = logging.getLogger(__name__)


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
    output = config.processed_dir / "ml_dataset_station_day.csv"
    ml.to_csv(output, index=False)
    LOGGER.info("Dataset ML salvo em %s (%s linhas)", output, len(ml))
    return ml

