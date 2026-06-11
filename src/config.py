"""Central configuration for the INMET Pernambuco pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
EXTRACTED_DIR = DATA_DIR / "extracted"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"
REPORTS_DIR = PROJECT_ROOT / "reports"

INMET_HISTORICAL_URL = "https://portal.inmet.gov.br/dadoshistoricos"
INMET_UPLOADS_BASE_URL = "https://portal.inmet.gov.br/uploads/dadoshistoricos"

PE_BBOX = {
    "min_lat": -9.75,
    "max_lat": -6.75,
    "min_lon": -42.25,
    "max_lon": -34.50,
}

MISSING_VALUES = ["", " ", "Null", "NULL", "null", "NaN", "nan", "9999", "9999.0", "-9999", "-9999.0"]


@dataclass(frozen=True)
class PipelineConfig:
    """Runtime options shared by all pipeline steps."""

    start_year: int = 2003
    end_year: int = 2025
    region: str = "PE"
    mode: str = "uf"
    include_neighbors: bool = False
    bbox: dict[str, float] = field(default_factory=lambda: PE_BBOX.copy())
    solar_day_start_hour: int = 5
    solar_day_end_hour: int = 18
    nighttime_solar_policy: str = "nan"
    solar_max_kj_m2: float = 5000.0
    wind_max_ms: float = 75.0
    outlier_method: str = "iqr"
    iqr_multiplier: float = 1.5
    zscore_threshold: float = 4.0
    drop_outliers_in_clean: bool = True
    request_timeout: int = 60
    download_retries: int = 3
    raw_dir: Path = RAW_DIR
    extracted_dir: Path = EXTRACTED_DIR
    interim_dir: Path = INTERIM_DIR
    processed_dir: Path = PROCESSED_DIR
    silver_dir: Path = SILVER_DIR
    gold_dir: Path = GOLD_DIR
    reports_dir: Path = REPORTS_DIR

    @property
    def years(self) -> range:
        return range(self.start_year, self.end_year + 1)


def ensure_directories() -> None:
    """Create the project data/report directories if they do not exist."""

    for path in [RAW_DIR, EXTRACTED_DIR, INTERIM_DIR, PROCESSED_DIR, SILVER_DIR, GOLD_DIR, REPORTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)
