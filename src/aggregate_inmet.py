"""Daily, annual and historical aggregations for cleaned INMET data."""

from __future__ import annotations

import logging

import pandas as pd

from .config import PipelineConfig

LOGGER = logging.getLogger(__name__)

STATION_COLUMNS = ["station_code", "station_name", "city", "state", "latitude", "longitude", "altitude"]


def aggregate_daily(hourly: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Generate daily station-level wind and solar metrics."""

    day_mask = hourly["hour"].between(config.solar_day_start_hour, config.solar_day_end_hour, inclusive="both")
    df = hourly.copy()
    df["solar_for_daily"] = df["solar_radiation_kj_m2"].where(day_mask)
    expected_solar_hours = config.solar_day_end_hour - config.solar_day_start_hour + 1

    daily = (
        df.groupby(STATION_COLUMNS + ["date", "year", "month", "day"], dropna=False)
        .agg(
            solar_daily_kj_m2=("solar_for_daily", "sum"),
            wind_daily_mean_ms=("wind_speed_10m_ms", "mean"),
            wind_daily_std_ms=("wind_speed_10m_ms", "std"),
            valid_hours_wind=("wind_speed_10m_ms", "count"),
            valid_hours_solar=("solar_for_daily", "count"),
        )
        .reset_index()
    )
    daily["solar_daily_kj_m2"] = daily["solar_daily_kj_m2"].where(daily["valid_hours_solar"] > 0)
    daily["solar_daily_kwh_m2_day"] = daily["solar_daily_kj_m2"] / 3600.0
    daily["missing_rate_wind"] = 1.0 - (daily["valid_hours_wind"] / 24.0)
    daily["missing_rate_solar"] = 1.0 - (daily["valid_hours_solar"] / expected_solar_hours)
    daily["missing_rate_wind"] = daily["missing_rate_wind"].clip(0, 1)
    daily["missing_rate_solar"] = daily["missing_rate_solar"].clip(0, 1)

    output = config.processed_dir / "inmet_pe_daily.csv"
    daily.to_csv(output, index=False)
    LOGGER.info("Base diaria salva em %s (%s linhas)", output, len(daily))
    return daily


def aggregate_annual(daily: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Generate annual station-level summary."""

    annual = (
        daily.groupby(STATION_COLUMNS + ["year"], dropna=False)
        .agg(
            wind_annual_mean_ms=("wind_daily_mean_ms", "mean"),
            wind_annual_std_ms=("wind_daily_mean_ms", "std"),
            solar_annual_mean_kwh_m2_day=("solar_daily_kwh_m2_day", "mean"),
            solar_annual_std_kwh_m2_day=("solar_daily_kwh_m2_day", "std"),
            valid_days=("date", "nunique"),
            missing_rate_wind=("missing_rate_wind", "mean"),
            missing_rate_solar=("missing_rate_solar", "mean"),
        )
        .reset_index()
    )
    output = config.processed_dir / "inmet_pe_station_annual_summary.csv"
    annual.to_csv(output, index=False)
    LOGGER.info("Resumo anual salvo em %s", output)
    return annual


def aggregate_historical(daily: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Generate historical station-level summary."""

    historical = (
        daily.groupby(STATION_COLUMNS, dropna=False)
        .agg(
            years_available=("year", lambda values: ",".join(str(int(v)) for v in sorted(pd.Series(values).dropna().unique()))),
            wind_historical_mean_ms=("wind_daily_mean_ms", "mean"),
            wind_historical_std_ms=("wind_daily_mean_ms", "std"),
            solar_historical_mean_kwh_m2_day=("solar_daily_kwh_m2_day", "mean"),
            solar_historical_std_kwh_m2_day=("solar_daily_kwh_m2_day", "std"),
            missing_rate_wind=("missing_rate_wind", "mean"),
            missing_rate_solar=("missing_rate_solar", "mean"),
        )
        .reset_index()
    )
    output = config.processed_dir / "inmet_pe_station_historical_summary.csv"
    historical.to_csv(output, index=False)
    LOGGER.info("Resumo historico salvo em %s", output)
    return historical


def run_aggregations(hourly: pd.DataFrame, config: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run and persist all aggregation outputs."""

    daily = aggregate_daily(hourly, config)
    annual = aggregate_annual(daily, config)
    historical = aggregate_historical(daily, config)
    return daily, annual, historical

