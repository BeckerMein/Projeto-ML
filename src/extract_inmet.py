"""Extract INMET ZIP archives into data/extracted."""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path

from tqdm import tqdm

from .config import PipelineConfig

LOGGER = logging.getLogger(__name__)


def extract_archives(config: PipelineConfig, archives: list[Path] | None = None) -> list[Path]:
    """Extract ZIPs from raw to extracted and return all CSV files available."""

    config.extracted_dir.mkdir(parents=True, exist_ok=True)
    archives = archives or sorted(config.raw_dir.glob("*.zip"))
    if not archives:
        LOGGER.warning("Nenhum ZIP encontrado em %s. Procurando CSVs ja extraidos/manual.", config.raw_dir)

    for archive in tqdm(archives, desc="Extraindo ZIPs"):
        year_dir = config.extracted_dir / archive.stem
        year_dir.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(archive) as zip_handle:
                zip_handle.extractall(year_dir)
            LOGGER.info("Extraido: %s -> %s", archive, year_dir)
        except zipfile.BadZipFile:
            LOGGER.exception("Arquivo ZIP invalido: %s", archive)

    csv_files = sorted(config.extracted_dir.rglob("*.csv")) + sorted(config.raw_dir.glob("*.csv"))
    if not csv_files:
        LOGGER.warning("Nenhum CSV encontrado em data/extracted ou data/raw.")
    return csv_files

