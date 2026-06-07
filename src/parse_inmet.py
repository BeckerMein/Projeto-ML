"""Robust parsing for INMET automatic station CSV files."""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import MISSING_VALUES, PipelineConfig

LOGGER = logging.getLogger(__name__)

ENCODINGS = ("utf-8-sig", "latin1", "iso-8859-1", "cp1252")


@dataclass
class InmetMetadata:
    station_code: str | None = None
    station_name: str | None = None
    city: str | None = None
    state: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    altitude: float | None = None


def strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text))
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def normalize_text(text: str) -> str:
    text = strip_accents(text).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def read_text_with_fallback(path: Path) -> tuple[str, str]:
    for encoding in ENCODINGS:
        try:
            return path.read_text(encoding=encoding), encoding
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="latin1", errors="replace"), "latin1"


def parse_decimal(value: object) -> float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip().replace(",", ".")
    text = re.sub(r"[^0-9+\-.]", "", text)
    if text in {"", "+", "-", "."}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def find_table_start(lines: list[str]) -> int:
    """Find the header line that starts the semicolon-separated table."""

    for idx, line in enumerate(lines):
        normalized = normalize_text(line)
        if "data" in normalized and ("hora" in normalized or "utc" in normalized) and ";" in line:
            return idx
    for idx, line in enumerate(lines):
        if line.count(";") >= 5:
            return idx
    raise ValueError("Nao foi possivel detectar a linha inicial da tabela.")


def metadata_key_to_field(key: str) -> str | None:
    key_norm = normalize_text(key)
    if key_norm in {"regiao"}:
        return None
    if key_norm in {"uf", "estado"}:
        return "state"
    if key_norm in {"estacao", "nome", "nome_da_estacao"}:
        return "station_name"
    if key_norm in {"codigo_wmo", "codigo", "cod_wmo", "codigo_da_estacao"}:
        return "station_code"
    if key_norm in {"latitude"}:
        return "latitude"
    if key_norm in {"longitude"}:
        return "longitude"
    if key_norm in {"altitude"}:
        return "altitude"
    if key_norm in {"municipio", "cidade"}:
        return "city"
    return None


def extract_metadata(lines: list[str], table_start: int) -> InmetMetadata:
    metadata = InmetMetadata()
    for line in lines[:table_start]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        field = metadata_key_to_field(key)
        if field is None:
            continue
        value = value.strip().strip(";")
        if field in {"latitude", "longitude", "altitude"}:
            setattr(metadata, field, parse_decimal(value))
        else:
            setattr(metadata, field, value.upper() if field == "state" else value)
    return metadata


def standardize_column_name(column: str) -> str:
    normalized = normalize_text(column)
    normalized = normalized.replace("deg_c", "c")
    if normalized in {"data", "data_medicao"} or normalized.startswith("data_"):
        return "date_raw"
    if "hora" in normalized and ("utc" in normalized or "medicao" in normalized or normalized == "hora"):
        return "hour_raw"
    if "vento_velocidade" in normalized and ("10m" in normalized or "m_s" in normalized):
        return "wind_speed_10m_ms"
    if "vento" in normalized and "velocidade" in normalized:
        return "wind_speed_10m_ms"
    if "radiacao_global" in normalized or "radiacao" in normalized and "kj_m" in normalized:
        return "solar_radiation_kj_m2"
    return normalized


def coerce_hour(hour_value: object) -> str | None:
    if pd.isna(hour_value):
        return None
    text = str(hour_value).strip().replace("UTC", "").replace("utc", "").strip()
    text = text.replace(":", "")
    digits = re.sub(r"\D", "", text)
    if not digits:
        return None
    if len(digits) <= 2:
        return f"{int(digits):02d}:00:00"
    if len(digits) in {3, 4}:
        return f"{int(digits[:-2]):02d}:{int(digits[-2:]):02d}:00"
    return f"{int(digits[:2]):02d}:{int(digits[2:4]):02d}:00"


def build_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    if "date_raw" not in df.columns:
        raise ValueError("Coluna de data nao encontrada.")
    if "hour_raw" not in df.columns:
        df["hour_raw"] = "00:00"

    date_text = df["date_raw"].astype(str).str.strip()
    hour_text = df["hour_raw"].map(coerce_hour)
    timestamp_text = date_text + " " + hour_text.fillna("00:00:00")
    timestamp_utc = pd.to_datetime(timestamp_text, dayfirst=True, errors="coerce", utc=False)
    if timestamp_utc.isna().all():
        timestamp_utc = pd.to_datetime(timestamp_text, errors="coerce", utc=False)

    df["timestamp_utc"] = timestamp_utc
    df["timestamp_local"] = df["timestamp_utc"] - pd.Timedelta(hours=3)
    df["date"] = df["timestamp_local"].dt.date
    df["year"] = df["timestamp_local"].dt.year
    df["month"] = df["timestamp_local"].dt.month
    df["day"] = df["timestamp_local"].dt.day
    df["hour"] = df["timestamp_local"].dt.hour
    return df


def read_inmet_csv(path: Path) -> pd.DataFrame:
    """Read one INMET CSV into the standardized hourly schema."""

    text, encoding = read_text_with_fallback(path)
    lines = text.splitlines()
    table_start = find_table_start(lines)
    metadata = extract_metadata(lines, table_start)

    df = pd.read_csv(
        path,
        sep=";",
        skiprows=table_start,
        encoding=encoding,
        na_values=MISSING_VALUES,
        dtype=str,
        engine="python",
        on_bad_lines="skip",
    )
    df = df.dropna(axis=1, how="all")
    df.columns = [standardize_column_name(col) for col in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]

    for required in ["wind_speed_10m_ms", "solar_radiation_kj_m2"]:
        if required not in df.columns:
            df[required] = np.nan

    for column in ["wind_speed_10m_ms", "solar_radiation_kj_m2"]:
        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
            .replace({value: np.nan for value in MISSING_VALUES})
            .str.replace(",", ".", regex=False)
        )
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = build_timestamps(df)
    df["station_code"] = metadata.station_code
    df["station_name"] = metadata.station_name
    df["city"] = metadata.city
    df["state"] = metadata.state
    df["latitude"] = metadata.latitude
    df["longitude"] = metadata.longitude
    df["altitude"] = metadata.altitude
    df["source_file"] = str(path)

    output_columns = [
        "station_code",
        "station_name",
        "city",
        "state",
        "latitude",
        "longitude",
        "altitude",
        "timestamp_utc",
        "timestamp_local",
        "date",
        "year",
        "month",
        "day",
        "hour",
        "wind_speed_10m_ms",
        "solar_radiation_kj_m2",
        "source_file",
    ]
    return df[output_columns].dropna(subset=["timestamp_utc"])


def parse_many_csvs(csv_files: list[Path], config: PipelineConfig) -> pd.DataFrame:
    """Parse many INMET CSV files and write a raw standardized parquet."""

    frames: list[pd.DataFrame] = []
    for path in tqdm(csv_files, desc="Lendo CSVs INMET"):
        try:
            frames.append(read_inmet_csv(path))
        except Exception:
            LOGGER.exception("Falha ao processar CSV: %s", path)

    if not frames:
        raise RuntimeError("Nenhum CSV do INMET foi lido com sucesso.")

    hourly = pd.concat(frames, ignore_index=True)
    hourly = hourly.drop_duplicates(subset=["station_code", "timestamp_utc"])
    config.interim_dir.mkdir(parents=True, exist_ok=True)
    output = config.interim_dir / "inmet_hourly_standardized.parquet"
    hourly.to_parquet(output, index=False)
    LOGGER.info("Base horaria padronizada salva em %s (%s linhas)", output, len(hourly))
    return hourly
