"""Configuracao compartilhada dos notebooks de energia da camada gold."""

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

# Padrao: 4 workers paralelos com ate 2 threads numericas cada.
# Isso mira uma carga de treino de 4 nucleos / 8 threads.
CPU_WORKERS = 4
BLAS_THREADS = 2

TEST_YEAR_FRACTION = 0.20
MLFLOW_EXPERIMENT_NAME = "gold_energy_potential"

# Entradas opcionais de inferencia futura. Mantenha None ate executar essa celula.
FUTURE_STATION_CODE = None
FUTURE_DATE = None
