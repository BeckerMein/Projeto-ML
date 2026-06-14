"""Configuracao compartilhada dos notebooks de energia com dados tratados."""

# Dados medios de estruturas ja existentes em Pernambuco.
ENERGY_CONFIG = {
    "SOLAR_PANEL_AREA_M2": 807.65,
    "SOLAR_PANEL_EFFICIENCY": 0.2157,
    "WIND_ROTOR_AREA_M2": 9503.317,
    "WIND_TURBINE_EFFICIENCY": 0.46075,
    "AIR_DENSITY_KG_M3": 1.06,
    "WIND_REFERENCE_HEIGHT_M": 10.0,
    "WIND_HUB_HEIGHT_M": 105.0,
    "WIND_SHEAR_EXPONENT_ALPHA": 0.20,
}

# Padrao: 8 workers paralelos com ate 2 threads numericas cada.
# Isso mira uma carga de treino de 8 nucleos / 16 threads.
CPU_WORKERS = 8
BLAS_THREADS = 2

TEST_YEAR_FRACTION = 0.20
MLFLOW_EXPERIMENT_NAME = "gold_energy_potential"

USE_HISTORICAL_FEATURES = True
HISTORY_MIN_OBSERVATIONS_DAY = 3
HISTORY_MIN_OBSERVATIONS_MONTH = 3
PRODUCTION_REFIT_WITH_FULL_GOLD = True
BASELINE_MODEL_NAME = "baseline_climatology"

# Entradas opcionais de inferencia futura. Mantenha None ate executar essa celula.
FUTURE_STATION_CODE = None
FUTURE_DATE = None
