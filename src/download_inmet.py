"""Download annual historical ZIP files from INMET."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from .config import INMET_HISTORICAL_URL, INMET_UPLOADS_BASE_URL, PipelineConfig

LOGGER = logging.getLogger(__name__)


def discover_annual_links(config: PipelineConfig) -> dict[int, str]:
    """Discover annual ZIP links from the official INMET historical page."""

    LOGGER.info("Consultando pagina oficial do INMET: %s", INMET_HISTORICAL_URL)
    try:
        response = requests.get(INMET_HISTORICAL_URL, timeout=config.request_timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            "Nao foi possivel acessar a pagina oficial de dados historicos do INMET. "
            "Use o modo manual colocando os ZIPs em data/raw/ ou tente novamente depois."
        ) from exc

    soup = BeautifulSoup(response.text, "html.parser")
    links: dict[int, str] = {}
    for anchor in soup.find_all("a", href=True):
        text = anchor.get_text(" ", strip=True)
        href = anchor["href"]
        candidate = f"{text} {href}"
        match = re.search(r"(20\d{2}|19\d{2})", candidate)
        if not match:
            continue
        year = int(match.group(1))
        if not href.lower().endswith(".zip") and "dadoshistoricos" not in href.lower():
            continue
        links[year] = urljoin(INMET_HISTORICAL_URL, href)

    # The current official page exposes /uploads/dadoshistoricos/YYYY.zip links.
    # Validate the predictable URL only for requested years not already found.
    for year in config.years:
        if year in links:
            continue
        candidate_url = f"{INMET_UPLOADS_BASE_URL}/{year}.zip"
        try:
            head = requests.head(candidate_url, timeout=config.request_timeout, allow_redirects=True)
            if head.ok:
                links[year] = candidate_url
        except requests.RequestException:
            LOGGER.debug("Falha ao validar URL candidata %s", candidate_url, exc_info=True)

    discovered = {year: links[year] for year in config.years if year in links}
    missing = sorted(set(config.years) - set(discovered))
    if missing:
        LOGGER.warning("Links anuais nao encontrados na pagina oficial: %s", missing)
    return discovered


def download_file(url: str, destination: Path, config: PipelineConfig) -> Path:
    """Download one file with retries and a progress bar."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        LOGGER.info("Arquivo ja existe, pulando download: %s", destination)
        return destination

    last_error: Exception | None = None
    for attempt in range(1, config.download_retries + 1):
        try:
            LOGGER.info("Baixando %s (tentativa %s/%s)", url, attempt, config.download_retries)
            with requests.get(url, stream=True, timeout=config.request_timeout) as response:
                response.raise_for_status()
                total = int(response.headers.get("content-length", 0))
                with destination.open("wb") as handle:
                    progress = tqdm(total=total, unit="B", unit_scale=True, desc=destination.name)
                    for chunk in response.iter_content(chunk_size=1024 * 256):
                        if chunk:
                            handle.write(chunk)
                            progress.update(len(chunk))
                    progress.close()
            return destination
        except requests.RequestException as exc:
            last_error = exc
            LOGGER.warning("Falha no download de %s: %s", url, exc)
            if destination.exists() and destination.stat().st_size == 0:
                destination.unlink()

    raise RuntimeError(f"Download falhou para {url}") from last_error


def download_inmet_archives(config: PipelineConfig, skip_download: bool = False) -> list[Path]:
    """Download requested INMET annual archives, or use files already in raw."""

    config.raw_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(config.raw_dir.glob("*.zip"))
    if skip_download:
        LOGGER.info("Download desativado; usando arquivos ja presentes em %s", config.raw_dir)
        return existing

    links = discover_annual_links(config)
    downloaded: list[Path] = []
    for year, url in sorted(links.items()):
        destination = config.raw_dir / f"{year}.zip"
        downloaded.append(download_file(url, destination, config))

    if not downloaded and existing:
        LOGGER.warning("Nenhum download concluido; usando ZIPs ja presentes em data/raw.")
        return existing
    return sorted(set(downloaded + existing))

