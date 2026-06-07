"""Command line entry point for the INMET Pernambuco pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .aggregate_inmet import run_aggregations
from .build_ml_dataset import build_ml_dataset
from .clean_inmet import apply_geographic_selection, clean_hourly_data
from .config import PE_BBOX, PipelineConfig, ensure_directories
from .download_inmet import download_inmet_archives
from .extract_inmet import extract_archives
from .parse_inmet import parse_many_csvs
from .quality_report import generate_quality_reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline INMET para dados horarios de Pernambuco.")
    parser.add_argument("--start-year", type=int, default=2003, help="Ano inicial.")
    parser.add_argument("--end-year", type=int, default=2025, help="Ano final.")
    parser.add_argument("--region", default="PE", help="UF alvo para o modo uf.")
    parser.add_argument("--mode", choices=["uf", "buffer"], default="uf", help="Modo de selecao geografica.")
    parser.add_argument("--skip-download", action="store_true", help="Usa apenas ZIPs/CSVs ja presentes em data/raw.")
    parser.add_argument("--solar-max-kj-m2", type=float, default=5000.0, help="Limite fisico/configuravel para radiacao horaria.")
    parser.add_argument("--wind-max-ms", type=float, default=75.0, help="Limite fisico/configuravel para vento horario.")
    parser.add_argument("--nighttime-solar-policy", choices=["nan", "zero"], default="nan")
    parser.add_argument("--outlier-method", choices=["iqr", "zscore", "none"], default="iqr")
    parser.add_argument("--keep-outliers", action="store_true", help="Mantem outliers na base clean, apenas com flags.")
    parser.add_argument("--bbox", nargs=4, type=float, metavar=("MIN_LAT", "MAX_LAT", "MIN_LON", "MAX_LON"))
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> PipelineConfig:
    bbox = PE_BBOX.copy()
    if args.bbox:
        bbox = {
            "min_lat": args.bbox[0],
            "max_lat": args.bbox[1],
            "min_lon": args.bbox[2],
            "max_lon": args.bbox[3],
        }
    return PipelineConfig(
        start_year=args.start_year,
        end_year=args.end_year,
        region=args.region,
        mode=args.mode,
        bbox=bbox,
        solar_max_kj_m2=args.solar_max_kj_m2,
        wind_max_ms=args.wind_max_ms,
        nighttime_solar_policy=args.nighttime_solar_policy,
        outlier_method=args.outlier_method,
        drop_outliers_in_clean=not args.keep_outliers,
    )


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    config = build_config(args)
    ensure_directories()

    logging.info("Iniciando pipeline INMET: %s-%s, modo=%s", config.start_year, config.end_year, config.mode)
    archives = download_inmet_archives(config, skip_download=args.skip_download)
    csv_files = extract_archives(config, archives)
    if not csv_files:
        raise RuntimeError(
            "Nenhum CSV encontrado. Baixe os ZIPs anuais no portal do INMET e coloque-os em data/raw/."
        )

    hourly_raw = parse_many_csvs([Path(path) for path in csv_files], config)
    hourly_selected, _stations = apply_geographic_selection(hourly_raw, config)
    hourly_clean = clean_hourly_data(hourly_selected, config)
    daily, annual, _historical = run_aggregations(hourly_clean, config)
    build_ml_dataset(daily, config)
    generate_quality_reports(hourly_clean, daily, annual, config)
    logging.info("Pipeline concluido com sucesso.")


if __name__ == "__main__":
    main()

