"""Baseline climatologica e features historicas derivadas apenas da camada gold."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from src.modeling.gold_energy import (
    FEATURE_COLUMNS,
    METADATA_COLUMNS,
    TARGET_COLUMNS,
    assert_no_forbidden_features,
)

HISTORICAL_LEVELS = ["station_doy", "station_month", "station", "global_doy", "global_month", "global"]
HISTORICAL_STATS = ["mean", "median", "std", "count"]


def target_prefix(target: str) -> str:
    """Retorna o prefixo compacto usado nas colunas historicas."""

    return target.replace("_generation_kwh_day", "")


def make_historical_feature_columns(target_columns: list[str] | None = None) -> list[str]:
    """Lista todas as features historicas geradas para os alvos informados."""

    columns: list[str] = []
    for target in target_columns or TARGET_COLUMNS:
        prefix = target_prefix(target)
        for level in HISTORICAL_LEVELS:
            for stat in HISTORICAL_STATS:
                columns.append(f"hist_{prefix}_{level}_{stat}")
    return columns


HISTORICAL_FEATURE_COLUMNS = make_historical_feature_columns()


@dataclass
class HistoricalFeatureReference:
    """Agregados historicos usados para criar features ou predicoes de baseline."""

    tables: dict[str, pd.DataFrame]
    min_observations_day: int
    min_observations_month: int
    source_years: list[int]

    def metadata(self) -> dict[str, Any]:
        return {
            "min_observations_day": self.min_observations_day,
            "min_observations_month": self.min_observations_month,
            "source_years": self.source_years,
            "feature_columns": HISTORICAL_FEATURE_COLUMNS,
        }


def _validate_input_columns(table: pd.DataFrame, require_targets: bool) -> None:
    required = {"station_code", "month", "day_of_year"}
    if require_targets:
        required.update(TARGET_COLUMNS)
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Tabela sem colunas obrigatorias para historico: {missing}")


def _clean_table(table: pd.DataFrame, require_targets: bool) -> pd.DataFrame:
    _validate_input_columns(table, require_targets=require_targets)
    clean = table.copy()
    clean["station_code"] = clean["station_code"].astype(str)
    clean["month"] = pd.to_numeric(clean["month"], errors="coerce")
    clean["day_of_year"] = pd.to_numeric(clean["day_of_year"], errors="coerce")
    if require_targets:
        for target in TARGET_COLUMNS:
            clean[target] = pd.to_numeric(clean[target], errors="coerce")
        clean = clean.dropna(subset=["station_code", "month", "day_of_year", *TARGET_COLUMNS])
    else:
        clean = clean.dropna(subset=["station_code", "month", "day_of_year"])
    if clean.empty:
        raise ValueError("Tabela vazia para calculo de historico.")
    clean["month"] = clean["month"].astype(int)
    clean["day_of_year"] = clean["day_of_year"].astype(int)
    return clean


def _aggregate_targets(
    table: pd.DataFrame,
    group_columns: list[str],
    level: str,
    min_observations: int | None = None,
) -> pd.DataFrame:
    named_aggs = {}
    for target in TARGET_COLUMNS:
        prefix = target_prefix(target)
        for stat in HISTORICAL_STATS:
            column = f"hist_{prefix}_{level}_{stat}"
            named_aggs[column] = (target, stat)

    if group_columns:
        aggregated = table.groupby(group_columns, dropna=False).agg(**named_aggs).reset_index()
    else:
        aggregated = pd.DataFrame({name: [getattr(table[target], stat)()] for name, (target, stat) in named_aggs.items()})

    for target in TARGET_COLUMNS:
        prefix = target_prefix(target)
        count_column = f"hist_{prefix}_{level}_count"
        if count_column in aggregated.columns:
            aggregated[count_column] = pd.to_numeric(aggregated[count_column], errors="coerce").fillna(0)
        if min_observations is not None:
            low_count = aggregated[count_column] < min_observations
            for stat in ["mean", "median", "std"]:
                aggregated.loc[low_count, f"hist_{prefix}_{level}_{stat}"] = np.nan
    return aggregated


def fit_historical_feature_reference(
    table: pd.DataFrame,
    *,
    min_observations_day: int = 3,
    min_observations_month: int = 3,
) -> HistoricalFeatureReference:
    """Ajusta agregados historicos usando somente as linhas recebidas."""

    clean = _clean_table(table, require_targets=True)
    source_years = sorted(clean["year"].dropna().astype(int).unique().tolist()) if "year" in clean.columns else []
    tables = {
        "station_doy": _aggregate_targets(
            clean,
            ["station_code", "day_of_year"],
            "station_doy",
            min_observations=min_observations_day,
        ),
        "station_month": _aggregate_targets(
            clean,
            ["station_code", "month"],
            "station_month",
            min_observations=min_observations_month,
        ),
        "station": _aggregate_targets(clean, ["station_code"], "station"),
        "global_doy": _aggregate_targets(
            clean,
            ["day_of_year"],
            "global_doy",
            min_observations=min_observations_day,
        ),
        "global_month": _aggregate_targets(
            clean,
            ["month"],
            "global_month",
            min_observations=min_observations_month,
        ),
        "global": _aggregate_targets(clean, [], "global"),
    }
    return HistoricalFeatureReference(
        tables=tables,
        min_observations_day=min_observations_day,
        min_observations_month=min_observations_month,
        source_years=source_years,
    )


def _merge_reference_table(frame: pd.DataFrame, reference: HistoricalFeatureReference, key: str, on: list[str]) -> pd.DataFrame:
    return frame.merge(reference.tables[key], how="left", on=on)


def _fill_feature_chain(frame: pd.DataFrame, columns: list[str], default: float = 0.0) -> None:
    for position, column in enumerate(columns):
        if position + 1 < len(columns):
            fallback = frame[columns[position + 1 :]].bfill(axis=1).iloc[:, 0]
            frame[column] = frame[column].fillna(fallback)
        frame[column] = frame[column].fillna(default)


def _fill_target_history(frame: pd.DataFrame, target: str) -> None:
    prefix = target_prefix(target)
    levels = HISTORICAL_LEVELS
    for stat in ["mean", "median", "std"]:
        columns = [f"hist_{prefix}_{level}_{stat}" for level in levels]
        _fill_feature_chain(frame, columns)
    for level in levels:
        column = f"hist_{prefix}_{level}_count"
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def transform_with_historical_features(
    frame: pd.DataFrame,
    reference: HistoricalFeatureReference,
) -> pd.DataFrame:
    """Adiciona features historicas a uma tabela sem olhar para os alvos dela."""

    result = _clean_table(frame, require_targets=False)
    result = result.drop(columns=[column for column in HISTORICAL_FEATURE_COLUMNS if column in result.columns])
    result = _merge_reference_table(result, reference, "station_doy", ["station_code", "day_of_year"])
    result = _merge_reference_table(result, reference, "station_month", ["station_code", "month"])
    result = _merge_reference_table(result, reference, "station", ["station_code"])
    result = _merge_reference_table(result, reference, "global_doy", ["day_of_year"])
    result = _merge_reference_table(result, reference, "global_month", ["month"])
    global_table = reference.tables["global"].copy()
    global_table["_historical_join_key"] = 1
    result["_historical_join_key"] = 1
    result = result.merge(global_table, how="left", on="_historical_join_key").drop(columns=["_historical_join_key"])

    for column in HISTORICAL_FEATURE_COLUMNS:
        if column not in result.columns:
            result[column] = np.nan
        result[column] = pd.to_numeric(result[column], errors="coerce")
    for target in TARGET_COLUMNS:
        _fill_target_history(result, target)
    return result


def make_expanding_historical_feature_frame(
    table: pd.DataFrame,
    *,
    min_observations_day: int = 3,
    min_observations_month: int = 3,
    drop_rows_without_history: bool = True,
) -> pd.DataFrame:
    """Cria features historicas para treino usando apenas anos anteriores a cada linha."""

    if "year" not in table.columns:
        raise ValueError("A tabela precisa da coluna year para features historicas expansivas.")

    clean = _clean_table(table, require_targets=True)
    frames = []
    years = sorted(clean["year"].dropna().astype(int).unique().tolist())
    for year in years:
        current = clean[clean["year"].astype(int) == year].copy()
        history = clean[clean["year"].astype(int) < year].copy()
        if history.empty:
            if drop_rows_without_history:
                continue
            for column in HISTORICAL_FEATURE_COLUMNS:
                current[column] = np.nan
            current["historical_reference_years"] = ""
            frames.append(current)
            continue
        reference = fit_historical_feature_reference(
            history,
            min_observations_day=min_observations_day,
            min_observations_month=min_observations_month,
        )
        current = transform_with_historical_features(current, reference)
        current["historical_reference_years"] = f"{min(reference.source_years)}-{max(reference.source_years)}"
        frames.append(current)

    if not frames:
        raise ValueError("Nao ha anos anteriores suficientes para criar features historicas.")
    return pd.concat(frames, ignore_index=True)


def prepare_historical_train_test_frames(
    modeling_table: pd.DataFrame,
    train_years: list[int],
    test_years: list[int],
    *,
    min_observations_day: int = 3,
    min_observations_month: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, HistoricalFeatureReference]:
    """Prepara treino expansivo e teste com referencia fixa do periodo de treino."""

    train_source = modeling_table[modeling_table["year"].astype(int).isin(train_years)].copy()
    test_source = modeling_table[modeling_table["year"].astype(int).isin(test_years)].copy()
    train_frame = make_expanding_historical_feature_frame(
        train_source,
        min_observations_day=min_observations_day,
        min_observations_month=min_observations_month,
    )
    train_reference = fit_historical_feature_reference(
        train_source,
        min_observations_day=min_observations_day,
        min_observations_month=min_observations_month,
    )
    test_frame = transform_with_historical_features(test_source, train_reference)
    return train_frame, test_frame, train_reference


def split_feature_target_metadata_with_columns(
    table: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Retorna X, y e metadados usando a lista explicita de features."""

    assert_no_forbidden_features(feature_columns)
    missing_features = sorted(set(feature_columns) - set(table.columns))
    if missing_features:
        raise ValueError(f"Features ausentes na tabela historica: {missing_features}")
    x = table[feature_columns].copy()
    y = table[TARGET_COLUMNS].copy()
    metadata = table[[column for column in METADATA_COLUMNS if column in table.columns]].copy()
    return x, y, metadata


def predict_climatology_baseline(
    frame: pd.DataFrame,
    reference: HistoricalFeatureReference,
) -> np.ndarray:
    """Prediz com a media historica por estacao/dia com fallback hierarquico."""

    features = transform_with_historical_features(frame, reference)
    predictions = []
    for target in TARGET_COLUMNS:
        prefix = target_prefix(target)
        predictions.append(features[f"hist_{prefix}_station_doy_mean"].to_numpy(dtype=float))
    return np.column_stack(predictions)


class ClimatologyBaselineRegressor(BaseEstimator, RegressorMixin):
    """Regressor deterministico baseado em medias historicas da camada gold."""

    def __init__(self, min_observations_day: int = 3, min_observations_month: int = 3):
        self.min_observations_day = min_observations_day
        self.min_observations_month = min_observations_month

    def fit(self, x: pd.DataFrame, y: pd.DataFrame | np.ndarray) -> "ClimatologyBaselineRegressor":
        y_frame = pd.DataFrame(y, columns=TARGET_COLUMNS) if not isinstance(y, pd.DataFrame) else y.reset_index(drop=True)
        table = pd.concat([x.reset_index(drop=True), y_frame[TARGET_COLUMNS].reset_index(drop=True)], axis=1)
        self.reference_ = fit_historical_feature_reference(
            table,
            min_observations_day=self.min_observations_day,
            min_observations_month=self.min_observations_month,
        )
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "reference_"):
            raise ValueError("A baseline precisa ser ajustada antes de prever.")
        return predict_climatology_baseline(x, self.reference_)


def compare_metric_tables(
    model_metrics: pd.DataFrame,
    baseline_metrics: pd.DataFrame,
    model_name: str,
    baseline_name: str = "baseline_climatology",
) -> pd.DataFrame:
    """Compara metricas do modelo contra a baseline alvo a alvo."""

    merged = model_metrics.merge(baseline_metrics, on="target", suffixes=(f"_{model_name}", f"_{baseline_name}"))
    rows = []
    for _, row in merged.iterrows():
        target = row["target"]
        for metric in ["mae", "rmse", "nrmse", "medae", "smape"]:
            model_value = float(row[f"{metric}_{model_name}"])
            baseline_value = float(row[f"{metric}_{baseline_name}"])
            rows.append(
                {
                    "target": target,
                    "metric": metric,
                    "baseline": baseline_value,
                    "model": model_value,
                    "model_minus_baseline": model_value - baseline_value,
                    "improvement_pct": ((baseline_value - model_value) / baseline_value * 100.0)
                    if baseline_value
                    else np.nan,
                }
            )
        model_r2 = float(row[f"r2_{model_name}"])
        baseline_r2 = float(row[f"r2_{baseline_name}"])
        rows.append(
            {
                "target": target,
                "metric": "r2",
                "baseline": baseline_r2,
                "model": model_r2,
                "model_minus_baseline": model_r2 - baseline_r2,
                "improvement_pct": np.nan,
            }
        )
    return pd.DataFrame(rows)


def save_historical_reference(reference: HistoricalFeatureReference, output_dir: Path) -> list[Path]:
    """Salva os agregados historicos em CSV e a configuracao em JSON."""

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    metadata_path = output_dir / "historical_reference_metadata.json"
    metadata_path.write_text(json.dumps(reference.metadata(), ensure_ascii=True, indent=2), encoding="utf-8")
    saved_paths.append(metadata_path)
    for name, table in reference.tables.items():
        path = output_dir / f"{name}.csv"
        table.to_csv(path, index=False)
        saved_paths.append(path)
    return saved_paths
