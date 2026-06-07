"""Quality reports and simple plots for INMET Pernambuco outputs."""

from __future__ import annotations

import logging
import os

import pandas as pd

from .config import PROJECT_ROOT, PipelineConfig

MPL_CACHE_DIR = PROJECT_ROOT / ".matplotlib-cache"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib.pyplot as plt

LOGGER = logging.getLogger(__name__)


def save_plot(path, close: bool = True) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    if close:
        plt.close()
    LOGGER.info("Grafico salvo em %s", path)


def generate_quality_reports(hourly: pd.DataFrame, daily: pd.DataFrame, annual: pd.DataFrame, config: PipelineConfig) -> None:
    """Write CSV quality tables and PNG diagnostic charts."""

    config.reports_dir.mkdir(parents=True, exist_ok=True)

    counts = (
        hourly.groupby(["station_code", "station_name", "city", "state"], dropna=False)
        .agg(
            hourly_records=("timestamp_local", "count"),
            first_timestamp=("timestamp_local", "min"),
            last_timestamp=("timestamp_local", "max"),
            valid_wind=("wind_speed_10m_ms", "count"),
            valid_solar=("solar_radiation_kj_m2", "count"),
        )
        .reset_index()
    )
    counts.to_csv(config.reports_dir / "data_count_by_station.csv", index=False)

    missing = (
        hourly.groupby(["station_code", "station_name", "city", "state"], dropna=False)
        .agg(
            missing_rate_wind=("wind_speed_10m_ms", lambda values: values.isna().mean()),
            missing_rate_solar=("solar_radiation_kj_m2", lambda values: values.isna().mean()),
        )
        .reset_index()
    )
    missing.to_csv(config.reports_dir / "missing_rate_by_station.csv", index=False)

    if not annual.empty:
        by_year = annual.groupby("year", as_index=False).agg(
            wind_mean=("wind_annual_mean_ms", "mean"),
            solar_mean=("solar_annual_mean_kwh_m2_day", "mean"),
        )
        plt.figure(figsize=(9, 4))
        plt.plot(by_year["year"], by_year["wind_mean"], marker="o")
        plt.xlabel("Ano")
        plt.ylabel("Vento medio anual (m/s)")
        plt.title("Serie temporal anual media de vento")
        save_plot(config.reports_dir / "annual_mean_wind.png")

        plt.figure(figsize=(9, 4))
        plt.plot(by_year["year"], by_year["solar_mean"], marker="o", color="#c98400")
        plt.xlabel("Ano")
        plt.ylabel("Irradiacao solar media (kWh/m2/dia)")
        plt.title("Serie temporal anual media de irradiacao solar")
        save_plot(config.reports_dir / "annual_mean_solar.png")

    plt.figure(figsize=(8, 4))
    hourly["wind_speed_10m_ms"].dropna().hist(bins=50, color="#3f6c9f")
    plt.xlabel("Velocidade do vento a 10 m (m/s)")
    plt.ylabel("Frequencia")
    plt.title("Histograma da velocidade do vento")
    save_plot(config.reports_dir / "hist_wind_speed.png")

    plt.figure(figsize=(8, 4))
    daily["solar_daily_kwh_m2_day"].dropna().hist(bins=50, color="#d79a2b")
    plt.xlabel("Irradiacao solar diaria (kWh/m2/dia)")
    plt.ylabel("Frequencia")
    plt.title("Histograma da irradiacao solar diaria")
    save_plot(config.reports_dir / "hist_daily_solar.png")

    stations = daily.drop_duplicates("station_code")
    plt.figure(figsize=(7, 7))
    plt.scatter(stations["longitude"], stations["latitude"], c="#1f7a5c", s=45)
    for _, row in stations.iterrows():
        label = row.get("station_code", "")
        plt.annotate(label, (row["longitude"], row["latitude"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Mapa simples das estacoes selecionadas")
    plt.grid(True, alpha=0.25)
    save_plot(config.reports_dir / "selected_stations_map.png")

    LOGGER.info("Relatorios de qualidade gerados em %s", config.reports_dir)
