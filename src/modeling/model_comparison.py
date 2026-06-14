"""Consolidacao das predicoes dos modelos treinados."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

from src.config import PROJECT_ROOT
from src.modeling.gold_energy import (
    ENERGY_RESULT_COLUMNS,
    METADATA_COLUMNS,
    TARGET_COLUMNS,
    WIND_HUB_HEIGHT_COLUMN,
    evaluate_predictions,
)

MODEL_PREDICTION_FILES = {
    "baseline": {
        "directory": "baseline_climatology",
        "pattern": "baseline_climatology_test_predictions_*.csv",
    },
    "random_forest": {
        "directory": "random_forest",
        "pattern": "random_forest_test_predictions_*.csv",
    },
    "mlp": {
        "directory": "mlp",
        "pattern": "mlp_test_predictions_*.csv",
    },
}
MODEL_ORDER = list(MODEL_PREDICTION_FILES)
VALUE_COLUMNS = TARGET_COLUMNS + [WIND_HUB_HEIGHT_COLUMN] + ENERGY_RESULT_COLUMNS
PHYSICAL_COMPARISON_COLUMNS = TARGET_COLUMNS


def _resolve_path_from_env(project_root: Path, env_name: str) -> Path | None:
    raw_value = os.environ.get(env_name)
    if not raw_value:
        return None

    path = Path(raw_value)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def find_latest_modeling_artifacts_dir(project_root: Path = PROJECT_ROOT) -> Path:
    """Encontra a ultima pasta de artefatos com resultados de modelagem."""

    modeling_root = project_root / "artifacts" / "modeling"
    if not modeling_root.exists():
        raise FileNotFoundError(f"Nenhuma pasta de modelagem encontrada em: {modeling_root}")

    candidates = [
        path
        for path in modeling_root.iterdir()
        if path.is_dir() and (path / "evaluation").exists() and path.name != "manual"
    ]
    if not candidates:
        raise FileNotFoundError(
            "Nenhuma execucao de modelagem com pasta evaluation foi encontrada em "
            f"{modeling_root}."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime).resolve()


def resolve_comparison_artifacts_dir(project_root: Path = PROJECT_ROOT) -> Path:
    """Resolve a pasta de artefatos da execucao atual ou a ultima disponivel."""

    artifacts_dir = _resolve_path_from_env(project_root, "MODEL_ARTIFACTS_DIR")
    if artifacts_dir:
        return artifacts_dir
    return find_latest_modeling_artifacts_dir(project_root)


def _latest_prediction_file(directory: Path, pattern: str) -> Path:
    files = sorted(
        path
        for path in directory.glob(pattern)
        if "_sample_" not in path.name and path.is_file()
    )
    if not files:
        raise FileNotFoundError(f"Nenhum arquivo encontrado em {directory} com padrao {pattern}.")
    return files[-1]


def load_model_prediction_tables(
    artifacts_dir: Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, Path]]:
    """Carrega os CSVs de predicao de teste dos modelos esperados."""

    evaluation_dir = artifacts_dir / "evaluation"
    tables: dict[str, pd.DataFrame] = {}
    file_paths: dict[str, Path] = {}

    for model_name, file_config in MODEL_PREDICTION_FILES.items():
        model_dir = evaluation_dir / file_config["directory"]
        prediction_path = _latest_prediction_file(model_dir, file_config["pattern"])
        tables[model_name] = pd.read_csv(prediction_path)
        file_paths[model_name] = prediction_path

    return tables, file_paths


def _validate_prediction_table(model_name: str, df: pd.DataFrame) -> None:
    missing_keys = [column for column in METADATA_COLUMNS if column not in df.columns]
    if missing_keys:
        raise ValueError(f"{model_name} sem colunas de identificacao: {missing_keys}")

    missing_actuals = [f"{column}_actual" for column in VALUE_COLUMNS if f"{column}_actual" not in df.columns]
    if missing_actuals:
        raise ValueError(f"{model_name} sem colunas reais esperadas: {missing_actuals}")

    missing_predictions = [f"{column}_pred" for column in VALUE_COLUMNS if f"{column}_pred" not in df.columns]
    if missing_predictions:
        raise ValueError(f"{model_name} sem colunas preditas esperadas: {missing_predictions}")

    duplicated_keys = int(df.duplicated(METADATA_COLUMNS).sum())
    if duplicated_keys:
        raise ValueError(f"{model_name} possui {duplicated_keys} linhas duplicadas por estacao/data.")


def build_consolidated_predictions(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Monta uma tabela wide com real, predicao de cada modelo e erros."""

    missing_models = [model for model in MODEL_ORDER if model not in tables]
    if missing_models:
        raise ValueError(f"Predicoes ausentes para consolidacao: {missing_models}")

    for model_name, df in tables.items():
        _validate_prediction_table(model_name, df)

    base = tables[MODEL_ORDER[0]]
    actual_columns = [f"{column}_actual" for column in VALUE_COLUMNS]
    consolidated = base[METADATA_COLUMNS + actual_columns].copy()

    for model_name in MODEL_ORDER:
        model_predictions = tables[model_name][
            METADATA_COLUMNS + [f"{column}_pred" for column in VALUE_COLUMNS]
        ].copy()
        model_predictions = model_predictions.rename(
            columns={f"{column}_pred": f"{column}_{model_name}" for column in VALUE_COLUMNS}
        )
        consolidated = consolidated.merge(
            model_predictions,
            on=METADATA_COLUMNS,
            how="left",
            validate="one_to_one",
        )

    prediction_columns = [
        f"{column}_{model_name}"
        for model_name in MODEL_ORDER
        for column in VALUE_COLUMNS
    ]
    missing_after_merge = consolidated[prediction_columns].isna().sum()
    missing_after_merge = missing_after_merge[missing_after_merge > 0]
    if not missing_after_merge.empty:
        raise ValueError(
            "Algumas predicoes nao foram pareadas por estacao/data: "
            f"{missing_after_merge.to_dict()}"
        )

    for column in VALUE_COLUMNS:
        actual_column = f"{column}_actual"
        for model_name in MODEL_ORDER:
            prediction_column = f"{column}_{model_name}"
            consolidated[f"{column}_error_{model_name}"] = (
                consolidated[prediction_column] - consolidated[actual_column]
            )
            consolidated[f"{column}_abs_error_{model_name}"] = (
                consolidated[f"{column}_error_{model_name}"].abs()
            )

    ordered_columns = list(METADATA_COLUMNS)
    for column in VALUE_COLUMNS:
        ordered_columns.append(f"{column}_actual")
        ordered_columns.extend(f"{column}_{model_name}" for model_name in MODEL_ORDER)
        ordered_columns.extend(f"{column}_error_{model_name}" for model_name in MODEL_ORDER)
        ordered_columns.extend(f"{column}_abs_error_{model_name}" for model_name in MODEL_ORDER)

    return consolidated[ordered_columns]


def build_long_predictions(consolidated: pd.DataFrame) -> pd.DataFrame:
    """Monta uma tabela long com uma linha por estacao, data e modelo."""

    frames = []
    for model_name in MODEL_ORDER:
        model_frame = consolidated[METADATA_COLUMNS].copy()
        model_frame["model"] = model_name
        for column in VALUE_COLUMNS:
            model_frame[f"{column}_actual"] = consolidated[f"{column}_actual"]
            model_frame[f"{column}_pred"] = consolidated[f"{column}_{model_name}"]
            model_frame[f"{column}_error"] = consolidated[f"{column}_error_{model_name}"]
            model_frame[f"{column}_abs_error"] = consolidated[f"{column}_abs_error_{model_name}"]
        frames.append(model_frame)
    return pd.concat(frames, ignore_index=True)


def build_model_metrics_summary(consolidated: pd.DataFrame) -> pd.DataFrame:
    """Recalcula metricas por modelo usando a tabela consolidada."""

    frames = []
    for model_name in MODEL_ORDER:
        for metric_group, columns in {
            "physical": PHYSICAL_COMPARISON_COLUMNS,
            "energy": ENERGY_RESULT_COLUMNS,
        }.items():
            y_true = consolidated[[f"{column}_actual" for column in columns]].copy()
            y_pred = consolidated[[f"{column}_{model_name}" for column in columns]].copy()
            y_true.columns = columns
            y_pred.columns = columns

            metrics, _ = evaluate_predictions(y_true, y_pred, target_columns=columns)
            metrics.insert(0, "metric_group", metric_group)
            metrics.insert(0, "model", model_name)
            metrics["balanced_nrmse_group"] = metrics["nrmse"].mean()
            frames.append(metrics)

    return pd.concat(frames, ignore_index=True)


def save_consolidated_model_comparison(
    artifacts_dir: Path | None = None,
    project_root: Path = PROJECT_ROOT,
    log_to_mlflow: bool = False,
) -> dict[str, Any]:
    """Gera e salva o consolidado wide, long e resumo de metricas."""

    resolved_artifacts_dir = artifacts_dir or resolve_comparison_artifacts_dir(project_root)
    resolved_artifacts_dir = Path(resolved_artifacts_dir).resolve()
    tables, source_files = load_model_prediction_tables(resolved_artifacts_dir)

    consolidated = build_consolidated_predictions(tables)
    long_predictions = build_long_predictions(consolidated)
    metrics_summary = build_model_metrics_summary(consolidated)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = resolved_artifacts_dir / "evaluation" / "model_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    wide_path = output_dir / f"model_predictions_consolidated_wide_{timestamp}.csv"
    long_path = output_dir / f"model_predictions_consolidated_long_{timestamp}.csv"
    metrics_path = output_dir / f"model_comparison_metrics_{timestamp}.csv"
    manifest_path = output_dir / f"model_comparison_manifest_{timestamp}.json"

    consolidated.to_csv(wide_path, index=False)
    long_predictions.to_csv(long_path, index=False)
    metrics_summary.to_csv(metrics_path, index=False)

    manifest = {
        "artifacts_dir": str(resolved_artifacts_dir),
        "created_at": timestamp,
        "models": MODEL_ORDER,
        "row_count_wide": int(len(consolidated)),
        "row_count_long": int(len(long_predictions)),
        "value_columns": VALUE_COLUMNS,
        "source_files": {model: str(path) for model, path in source_files.items()},
        "outputs": {
            "wide": str(wide_path),
            "long": str(long_path),
            "metrics": str(metrics_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    if log_to_mlflow:
        with mlflow.start_run(run_name="model_comparison_consolidated"):
            mlflow.log_param("models", ",".join(MODEL_ORDER))
            mlflow.log_param("source_artifacts_dir", str(resolved_artifacts_dir))
            mlflow.log_metric("row_count_wide", len(consolidated))
            mlflow.log_metric("row_count_long", len(long_predictions))
            mlflow.log_artifacts(str(output_dir), artifact_path="evaluation/model_comparison")

    return {
        "artifacts_dir": resolved_artifacts_dir,
        "output_dir": output_dir,
        "wide_path": wide_path,
        "long_path": long_path,
        "metrics_path": metrics_path,
        "manifest_path": manifest_path,
        "wide_rows": len(consolidated),
        "long_rows": len(long_predictions),
        "metrics": metrics_summary,
    }
