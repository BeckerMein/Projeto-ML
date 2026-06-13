"""Funcoes auxiliares para modelar potencial solar e eolico usando apenas tabelas gold."""

from __future__ import annotations

import json
import math
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import PROJECT_ROOT

GOLD_DAILY_FILENAME = "inmet_pe_daily.csv"
TARGET_COLUMNS = ["solar_generation_kwh_day", "wind_generation_kwh_day"]
FEATURE_COLUMNS = [
    "station_code",
    "latitude",
    "longitude",
    "altitude",
    "month",
    "day_of_year",
    "month_sin",
    "month_cos",
    "day_of_year_sin",
    "day_of_year_cos",
]
CATEGORICAL_FEATURES = ["station_code"]
NUMERIC_FEATURES = [column for column in FEATURE_COLUMNS if column not in CATEGORICAL_FEATURES]
METADATA_COLUMNS = ["station_code", "station_name", "city", "state", "latitude", "longitude", "altitude", "date", "year"]
DERIVED_AUDIT_COLUMNS = ["wind_daily_mean_hub_height_ms"]
FORBIDDEN_FEATURE_COLUMNS = {
    "solar_daily_kwh_m2_day",
    "solar_daily_kj_m2",
    "wind_daily_mean_ms",
    "wind_daily_std_ms",
    "valid_hours_wind",
    "valid_hours_solar",
    "missing_rate_wind",
    "missing_rate_solar",
    "wind_daily_mean_hub_height_ms",
}
REQUIRED_DAILY_COLUMNS = {
    "station_code",
    "latitude",
    "longitude",
    "altitude",
    "date",
    "year",
    "month",
    "wind_daily_mean_ms",
    "solar_daily_kwh_m2_day",
}
REQUIRED_ENERGY_CONFIG = [
    "SOLAR_PANEL_AREA_M2",
    "SOLAR_PANEL_EFFICIENCY",
    "WIND_ROTOR_AREA_M2",
    "WIND_TURBINE_EFFICIENCY",
    "AIR_DENSITY_KG_M3",
    "WIND_REFERENCE_HEIGHT_M",
    "WIND_HUB_HEIGHT_M",
    "WIND_SHEAR_EXPONENT_ALPHA",
]


def load_gold_daily(project_root: Path = PROJECT_ROOT) -> pd.DataFrame:
    """Carrega a tabela gold diaria e falha claramente se ela nao existir."""

    gold_path = project_root / "data" / "gold" / GOLD_DAILY_FILENAME
    if not gold_path.exists():
        raise FileNotFoundError(
            f"Tabela gold nao encontrada: {gold_path}. "
            "Gere a camada gold antes de treinar os modelos."
        )

    df = pd.read_csv(gold_path)
    missing = sorted(REQUIRED_DAILY_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Tabela gold sem colunas obrigatorias para modelagem: {missing}")
    return df


def validate_energy_config(config: dict[str, Any]) -> dict[str, float]:
    """Valida constantes fisicas fornecidas pelo usuario sem inventar padroes."""

    missing = [key for key in REQUIRED_ENERGY_CONFIG if key not in config or config[key] is None]
    if missing:
        raise ValueError(
            "Preencha as constantes fisicas antes do treino. "
            f"Valores ausentes: {missing}"
        )

    validated: dict[str, float] = {}
    for key in REQUIRED_ENERGY_CONFIG:
        value = float(config[key])
        if key == "WIND_SHEAR_EXPONENT_ALPHA":
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} deve ser um numero finito maior ou igual a zero.")
        elif not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} deve ser um numero positivo finito.")
        validated[key] = value

    for key in ["SOLAR_PANEL_EFFICIENCY", "WIND_TURBINE_EFFICIENCY"]:
        if validated[key] > 1:
            raise ValueError(f"{key} deve estar no intervalo (0, 1].")
    if validated["WIND_SHEAR_EXPONENT_ALPHA"] > 1:
        raise ValueError("WIND_SHEAR_EXPONENT_ALPHA deve estar no intervalo [0, 1].")
    return validated


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona variaveis ciclicas de calendario derivadas da coluna de data da gold."""

    result = df.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    if result["date"].isna().any():
        raise ValueError("A coluna date da gold contem valores invalidos.")
    result["year"] = result["date"].dt.year
    result["month"] = result["date"].dt.month
    result["day_of_year"] = result["date"].dt.dayofyear
    result["month_sin"] = np.sin(2 * math.pi * result["month"] / 12.0)
    result["month_cos"] = np.cos(2 * math.pi * result["month"] / 12.0)
    result["day_of_year_sin"] = np.sin(2 * math.pi * result["day_of_year"] / 366.0)
    result["day_of_year_cos"] = np.cos(2 * math.pi * result["day_of_year"] / 366.0)
    return result


def add_generation_targets(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Calcula alvos de geracao potencial a partir das variaveis meteorologicas da gold."""

    constants = validate_energy_config(config)
    result = df.copy()
    result["solar_daily_kwh_m2_day"] = pd.to_numeric(result["solar_daily_kwh_m2_day"], errors="coerce")
    result["wind_daily_mean_ms"] = pd.to_numeric(result["wind_daily_mean_ms"], errors="coerce")

    if (result["solar_daily_kwh_m2_day"].dropna() < 0).any():
        raise ValueError("A gold contem irradiacao solar diaria negativa.")
    if (result["wind_daily_mean_ms"].dropna() < 0).any():
        raise ValueError("A gold contem velocidade media de vento negativa.")

    result["solar_generation_kwh_day"] = (
        result["solar_daily_kwh_m2_day"]
        * constants["SOLAR_PANEL_AREA_M2"]
        * constants["SOLAR_PANEL_EFFICIENCY"]
    )
    wind_height_factor = (
        constants["WIND_HUB_HEIGHT_M"] / constants["WIND_REFERENCE_HEIGHT_M"]
    ) ** constants["WIND_SHEAR_EXPONENT_ALPHA"]
    result["wind_daily_mean_hub_height_ms"] = result["wind_daily_mean_ms"] * wind_height_factor
    result["wind_generation_kwh_day"] = (
        0.5
        * constants["AIR_DENSITY_KG_M3"]
        * constants["WIND_ROTOR_AREA_M2"]
        * constants["WIND_TURBINE_EFFICIENCY"]
        * result["wind_daily_mean_hub_height_ms"].pow(3)
        * 24.0
        / 1000.0
    )
    return result


def prepare_energy_modeling_table(df: pd.DataFrame, energy_config: dict[str, Any]) -> pd.DataFrame:
    """Cria uma tabela estacao-dia de modelagem com variaveis permitidas e alvos."""

    table = add_generation_targets(add_calendar_features(df), energy_config)
    for column in FEATURE_COLUMNS + TARGET_COLUMNS:
        if column != "station_code":
            table[column] = pd.to_numeric(table[column], errors="coerce")
    table["station_code"] = table["station_code"].astype(str)

    selected_columns = [column for column in METADATA_COLUMNS if column in table.columns]
    selected_columns += [column for column in DERIVED_AUDIT_COLUMNS if column in table.columns]
    selected_columns += FEATURE_COLUMNS + TARGET_COLUMNS
    selected_columns = list(dict.fromkeys(selected_columns))
    table = table[selected_columns].dropna(subset=FEATURE_COLUMNS + TARGET_COLUMNS).copy()
    if table.empty:
        raise ValueError("A tabela de modelagem ficou vazia apos remover valores ausentes.")
    return table.sort_values(["year", "station_code", "date"]).reset_index(drop=True)


def split_feature_target_metadata(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Retorna X, y e metadados a partir da tabela de modelagem preparada."""

    assert_no_forbidden_features(FEATURE_COLUMNS)
    x = table[FEATURE_COLUMNS].copy()
    y = table[TARGET_COLUMNS].copy()
    metadata = table[[column for column in METADATA_COLUMNS if column in table.columns]].copy()
    return x, y, metadata


def assert_no_forbidden_features(features: list[str] | pd.Index) -> None:
    """Garante que clima e qualidade do mesmo dia nao sejam usados como entradas do modelo."""

    forbidden = sorted(set(features) & FORBIDDEN_FEATURE_COLUMNS)
    if forbidden:
        raise ValueError(f"Features proibidas detectadas em X: {forbidden}")


def temporal_train_test_split(
    x: pd.DataFrame,
    y: pd.DataFrame,
    metadata: pd.DataFrame,
    test_year_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[int], list[int]]:
    """Separa dados por ano, reservando os anos mais recentes para teste final."""

    years = sorted(metadata["year"].dropna().astype(int).unique().tolist())
    if len(years) < 3:
        raise ValueError("Sao necessarios pelo menos 3 anos na gold para treino, validacao temporal e teste.")
    test_count = max(1, math.ceil(len(years) * test_year_fraction))
    test_years = years[-test_count:]
    train_years = years[:-test_count]
    if len(train_years) < 2:
        raise ValueError("Sao necessarios pelo menos 2 anos de treino para folds temporais.")

    train_mask = metadata["year"].astype(int).isin(train_years).to_numpy()
    test_mask = metadata["year"].astype(int).isin(test_years).to_numpy()
    return (
        x.loc[train_mask].reset_index(drop=True),
        x.loc[test_mask].reset_index(drop=True),
        y.loc[train_mask].reset_index(drop=True),
        y.loc[test_mask].reset_index(drop=True),
        metadata.loc[train_mask].reset_index(drop=True),
        metadata.loc[test_mask].reset_index(drop=True),
        train_years,
        test_years,
    )


def make_temporal_cv_splits(metadata_train: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """Cria folds temporais expansivos por ano dentro do periodo de treino."""

    years = sorted(metadata_train["year"].dropna().astype(int).unique().tolist())
    if len(years) < 2:
        raise ValueError("A busca de hiperparametros precisa de pelo menos 2 anos no treino.")

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    year_values = metadata_train["year"].astype(int).to_numpy()
    for position in range(1, len(years)):
        train_years = years[:position]
        validation_year = years[position]
        train_index = np.flatnonzero(np.isin(year_values, train_years))
        validation_index = np.flatnonzero(year_values == validation_year)
        if len(train_index) and len(validation_index):
            splits.append((train_index, validation_index))
    if not splits:
        raise ValueError("Nao foi possivel criar folds temporais para validacao.")
    return splits


def balanced_negative_nrmse(y_true: Any, y_pred: Any) -> float:
    """Pontuacao para selecao de modelo: NRMSE medio negativo entre os dois alvos."""

    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.ndim == 1:
        true = true.reshape(-1, 1)
        pred = pred.reshape(-1, 1)

    target_scores = []
    for index in range(true.shape[1]):
        observed = true[:, index]
        predicted = pred[:, index]
        rmse = float(np.sqrt(np.mean((observed - predicted) ** 2)))
        scale = float(np.nanmax(observed) - np.nanmin(observed))
        if not math.isfinite(scale) or scale == 0:
            scale = float(np.nanstd(observed))
        if not math.isfinite(scale) or scale == 0:
            scale = 1.0
        target_scores.append(rmse / scale)
    return -float(np.mean(target_scores))


BALANCED_NRMSE_SCORER = make_scorer(balanced_negative_nrmse, greater_is_better=True)


def evaluate_predictions(y_true: pd.DataFrame, y_pred: np.ndarray) -> tuple[pd.DataFrame, dict[str, float]]:
    """Calcula metricas por alvo no conjunto de teste temporal separado."""

    rows = []
    metrics: dict[str, float] = {}
    for index, target in enumerate(TARGET_COLUMNS):
        observed = y_true[target].to_numpy(dtype=float)
        predicted = np.asarray(y_pred, dtype=float)[:, index]
        mae = float(mean_absolute_error(observed, predicted))
        rmse = float(np.sqrt(mean_squared_error(observed, predicted)))
        r2 = float(r2_score(observed, predicted))
        scale = float(np.nanmax(observed) - np.nanmin(observed))
        nrmse = rmse / scale if math.isfinite(scale) and scale else np.nan
        rows.append({"target": target, "mae": mae, "rmse": rmse, "nrmse": nrmse, "r2": r2})
        metric_prefix = target.replace("_generation_kwh_day", "")
        metrics[f"{metric_prefix}_mae"] = mae
        metrics[f"{metric_prefix}_rmse"] = rmse
        metrics[f"{metric_prefix}_nrmse"] = float(nrmse) if math.isfinite(nrmse) else np.nan
        metrics[f"{metric_prefix}_r2"] = r2
    metrics["balanced_nrmse"] = float(np.nanmean([row["nrmse"] for row in rows]))
    return pd.DataFrame(rows), metrics


def make_one_hot_encoder() -> OneHotEncoder:
    """Cria um encoder compativel com versoes recentes e antigas do scikit-learn."""

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_model_pipeline(estimator: Any, scale_numeric: bool) -> Pipeline:
    """Monta o pipeline de preprocessamento + estimador usado pelos dois notebooks."""

    numeric_transformer: str | StandardScaler = StandardScaler() if scale_numeric else "passthrough"
    preprocessor = ColumnTransformer(
        transformers=[
            ("station_code", make_one_hot_encoder(), CATEGORICAL_FEATURES),
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
        ],
        remainder="drop",
    )
    return Pipeline([("preprocess", preprocessor), ("model", estimator)])


def random_forest_base_pipeline(random_state: int) -> Pipeline:
    """Pipeline Random Forest configurado para busca paralela sem paralelismo aninhado."""

    estimator = RandomForestRegressor(random_state=random_state, n_jobs=1)
    return build_model_pipeline(estimator, scale_numeric=False)


def mlp_base_estimator(random_state: int) -> TransformedTargetRegressor:
    """Pipeline MLP com variaveis numericas escaladas e alvos multi-saida escalados."""

    estimator = MLPRegressor(random_state=random_state, max_iter=1000)
    pipeline = build_model_pipeline(estimator, scale_numeric=True)
    return TransformedTargetRegressor(regressor=pipeline, transformer=StandardScaler())


def random_forest_search_space() -> dict[str, list[Any]]:
    """Espaco amplo de hiperparametros do Random Forest."""

    return {
        "model__n_estimators": [300, 500, 800, 1200],
        "model__max_depth": [None, 8, 12, 18, 24, 32],
        "model__min_samples_split": [2, 5, 10, 20],
        "model__min_samples_leaf": [1, 2, 4, 8],
        "model__max_features": ["sqrt", 0.5, 0.7, 1.0],
        "model__bootstrap": [True, False],
        "model__criterion": ["squared_error"],
    }


def _neighbors(options: list[Any], value: Any, width: int = 1, max_items: int = 3) -> list[Any]:
    if value not in options:
        return [value]
    position = options.index(value)
    start = max(0, position - width)
    end = min(len(options), position + width + 1)
    return options[start:end][:max_items]


def random_forest_refinement_grid(best_params: dict[str, Any]) -> dict[str, list[Any]]:
    """Grid pequeno ao redor do melhor resultado do RandomizedSearchCV, limitado a 72 combinacoes."""

    space = random_forest_search_space()
    grid = {
        "model__n_estimators": _neighbors(space["model__n_estimators"], best_params["model__n_estimators"]),
        "model__max_depth": _neighbors(space["model__max_depth"], best_params["model__max_depth"]),
        "model__min_samples_split": _neighbors(
            space["model__min_samples_split"], best_params["model__min_samples_split"], max_items=2
        ),
        "model__min_samples_leaf": _neighbors(
            space["model__min_samples_leaf"], best_params["model__min_samples_leaf"], max_items=2
        ),
        "model__max_features": _neighbors(
            space["model__max_features"], best_params["model__max_features"], max_items=2
        ),
        "model__bootstrap": [best_params["model__bootstrap"]],
        "model__criterion": [best_params["model__criterion"]],
    }
    if grid_size(grid) > 72:
        raise ValueError(f"Grid de refino RF maior que o planejado: {grid_size(grid)} combinacoes.")
    return grid


def mlp_search_space() -> dict[str, list[Any]]:
    """Espaco amplo de hiperparametros da MLP usando listas numericas em escala log."""

    return {
        "regressor__model__hidden_layer_sizes": [
            (64,),
            (128,),
            (256,),
            (128, 64),
            (256, 128),
            (256, 128, 64),
            (512, 256),
        ],
        "regressor__model__activation": ["relu", "tanh"],
        "regressor__model__alpha": np.geomspace(1e-6, 1e-2, 80).tolist(),
        "regressor__model__learning_rate_init": np.geomspace(1e-4, 3e-3, 80).tolist(),
        "regressor__model__batch_size": [64, 128, 256, 512],
        "regressor__model__learning_rate": ["constant", "adaptive"],
        "regressor__model__early_stopping": [True],
        "regressor__model__validation_fraction": [0.15],
        "regressor__model__n_iter_no_change": [20, 40],
        "regressor__model__max_iter": [1000],
    }


def _geom_refinement(value: float, low: float, high: float, factor: float, points: int = 25) -> list[float]:
    start = max(low, value / factor)
    end = min(high, value * factor)
    if start == end:
        return [value]
    return np.geomspace(start, end, points).tolist()


def mlp_refinement_space(best_params: dict[str, Any]) -> dict[str, list[Any]]:
    """Espaco reduzido de busca da MLP ao redor do melhor resultado da busca ampla."""

    base_space = mlp_search_space()
    hidden = best_params["regressor__model__hidden_layer_sizes"]
    batch = best_params["regressor__model__batch_size"]
    batch_options = _neighbors(base_space["regressor__model__batch_size"], batch, max_items=3)
    hidden_options = [hidden]
    for candidate in base_space["regressor__model__hidden_layer_sizes"]:
        if candidate != hidden and len(hidden_options) < 3:
            hidden_options.append(candidate)
    return {
        "regressor__model__hidden_layer_sizes": hidden_options,
        "regressor__model__activation": [best_params["regressor__model__activation"]],
        "regressor__model__alpha": _geom_refinement(best_params["regressor__model__alpha"], 1e-7, 1e-1, 10),
        "regressor__model__learning_rate_init": _geom_refinement(
            best_params["regressor__model__learning_rate_init"], 1e-5, 1e-2, 3
        ),
        "regressor__model__batch_size": batch_options,
        "regressor__model__learning_rate": [best_params["regressor__model__learning_rate"]],
        "regressor__model__early_stopping": [True],
        "regressor__model__validation_fraction": [0.15],
        "regressor__model__n_iter_no_change": [best_params["regressor__model__n_iter_no_change"]],
        "regressor__model__max_iter": [1000],
    }


def grid_size(param_grid: dict[str, list[Any]]) -> int:
    """Retorna o numero de combinacoes em um grid."""

    return int(np.prod([len(values) for values in param_grid.values()]))


def to_jsonable(value: Any) -> Any:
    """Converte objetos com tipos numpy para valores amigaveis a JSON em logs."""

    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def train_random_search(
    estimator: Any,
    param_distributions: dict[str, list[Any]],
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    n_iter: int,
    n_jobs: int,
    random_state: int,
    verbose: int = 2,
) -> tuple[RandomizedSearchCV, float]:
    """Executa RandomizedSearchCV com a metrica do projeto."""

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=BALANCED_NRMSE_SCORER,
        cv=cv_splits,
        n_jobs=n_jobs,
        random_state=random_state,
        refit=True,
        verbose=verbose,
        error_score="raise",
    )
    start = perf_counter()
    search.fit(x_train, y_train)
    return search, perf_counter() - start


def train_grid_search(
    estimator: Any,
    param_grid: dict[str, list[Any]],
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    n_jobs: int,
    verbose: int = 2,
) -> tuple[GridSearchCV, float]:
    """Executa GridSearchCV com a metrica do projeto."""

    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=BALANCED_NRMSE_SCORER,
        cv=cv_splits,
        n_jobs=n_jobs,
        refit=True,
        verbose=verbose,
        error_score="raise",
    )
    start = perf_counter()
    search.fit(x_train, y_train)
    return search, perf_counter() - start


def fit_final_model(estimator: Any, best_params: dict[str, Any], x_train: pd.DataFrame, y_train: pd.DataFrame) -> tuple[Any, float]:
    """Clona, configura e ajusta o estimador final em todos os anos de treino."""

    final_model = clone(estimator).set_params(**best_params)
    start = perf_counter()
    final_model.fit(x_train, y_train)
    return final_model, perf_counter() - start


def make_future_feature_frame(table: pd.DataFrame, future_date: str | pd.Timestamp) -> pd.DataFrame:
    """Monta uma linha de variaveis por estacao conhecida para uma data futura."""

    date = pd.to_datetime(future_date, errors="raise")
    station_columns = ["station_code", "latitude", "longitude", "altitude"]
    optional_columns = [column for column in ["station_name", "city", "state"] if column in table.columns]
    stations = (
        table[station_columns + optional_columns]
        .drop_duplicates(subset=["station_code"])
        .sort_values("station_code")
        .reset_index(drop=True)
    )
    stations["date"] = date
    stations["year"] = date.year
    stations["month"] = date.month
    stations["day_of_year"] = date.dayofyear
    stations["month_sin"] = np.sin(2 * math.pi * stations["month"] / 12.0)
    stations["month_cos"] = np.cos(2 * math.pi * stations["month"] / 12.0)
    stations["day_of_year_sin"] = np.sin(2 * math.pi * stations["day_of_year"] / 366.0)
    stations["day_of_year_cos"] = np.cos(2 * math.pi * stations["day_of_year"] / 366.0)
    return stations


def predict_future_ranking(model: Any, table: pd.DataFrame, future_date: str | pd.Timestamp) -> pd.DataFrame:
    """Prediz potencial solar, eolico e hibrido para todas as estacoes conhecidas em uma data futura."""

    future = make_future_feature_frame(table, future_date)
    predictions = np.asarray(model.predict(future[FEATURE_COLUMNS]), dtype=float)
    result = future[[column for column in ["station_code", "station_name", "city", "state", "latitude", "longitude", "altitude", "date"] if column in future.columns]].copy()
    result["solar_generation_kwh_day_pred"] = predictions[:, 0]
    result["wind_generation_kwh_day_pred"] = predictions[:, 1]
    result["hybrid_generation_kwh_day_pred"] = (
        result["solar_generation_kwh_day_pred"] + result["wind_generation_kwh_day_pred"]
    )
    return result.sort_values("hybrid_generation_kwh_day_pred", ascending=False).reset_index(drop=True)


def predict_future_for_station(
    model: Any,
    table: pd.DataFrame,
    station_code: str,
    future_date: str | pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prediz uma estacao e retorna o ranking completo da mesma data como contexto."""

    station_code = str(station_code)
    ranking = predict_future_ranking(model, table, future_date)
    station_result = ranking[ranking["station_code"].astype(str) == station_code].copy()
    if station_result.empty:
        known = ", ".join(ranking["station_code"].astype(str).head(10).tolist())
        raise ValueError(f"station_code nao encontrado na gold: {station_code}. Exemplos conhecidos: {known}")
    station_result.insert(0, "ranking_position", station_result.index + 1)
    return station_result.reset_index(drop=True), ranking


def describe_search_space(param_space: dict[str, list[Any]]) -> str:
    """Serializa um resumo compacto do espaco de busca."""

    summary = {key: len(values) for key, values in param_space.items()}
    return json.dumps(summary, ensure_ascii=True, sort_keys=True)


def cartesian_preview(param_grid: dict[str, list[Any]], limit: int = 5) -> list[dict[str, Any]]:
    """Gera uma pequena previa das combinacoes do grid para exibicao no notebook."""

    keys = list(param_grid)
    rows = []
    for values in list(product(*(param_grid[key] for key in keys)))[:limit]:
        rows.append(dict(zip(keys, values)))
    return rows
