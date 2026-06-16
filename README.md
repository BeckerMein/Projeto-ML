# Previsão de Potencial Solar, Eólico e Híbrido em Pernambuco

Este repositório apresenta uma solução de Aprendizado de Máquina para estimar o potencial de geração solar, eólica e híbrida em Pernambuco a partir de dados históricos horários das estações automáticas do Instituto Nacional de Meteorologia (INMET). A solução contempla aquisição e padronização de dados, análise exploratória, tratamento de qualidade, agregação em base analítica, treinamento e comparação de modelos, rastreamento de experimentos com MLflow, visualização interativa em dashboard e disponibilização em ambiente conteinerizado.

## Identificação Do Projeto

**Disciplina:** Machine Learning I e Projeto 3

**Instituição:** CESAR School

**Repositório:** https://github.com/BeckerMein/Projeto-ML

**Integrantes:**

| Nome | Usuário GitHub | E-mail |
|---|---|---|
| Matheus Henrique de Melo Araujo | @MathhAraujo | mhma@cesar.school |
| Matheus de Lucena Henriques | @Matheuslh | mlh@cesar.school |
| Miguel Chaves Becker | @BeckerMein | mcb4@cesar.school |

## Síntese Da Solução

O objetivo do projeto é apoiar a análise de potencial renovável em Pernambuco por meio de modelos supervisionados treinados sobre séries históricas meteorológicas. A unidade de observação adotada é `estação-dia`, preservando a dimensão temporal dos dados e permitindo avaliar a capacidade de generalização dos modelos para anos futuros.

Na implementação atual, os modelos são treinados para estimar duas variáveis físicas agregadas em escala diária:

| Alvo | Interpretação |
|---|---|
| `solar_daily_kwh_m2_day` | Irradiação solar diária por metro quadrado, calculada a partir da radiação solar horária do INMET. |
| `wind_daily_mean_ms` | Velocidade média diária do vento a 10 m. |

O alvo solar não representa a geração final de um sistema fotovoltaico. Ele é obtido no pipeline de agregação a partir da soma diária de `solar_radiation_kj_m2` no período diurno e da conversão de kJ/m² para kWh/m². Após a previsão desses alvos físicos, a geração solar, eólica e híbrida em kWh/dia é calculada deterministicamente com constantes físicas definidas em `src/modeling/training_config.py`. Essa decisão metodológica separa o aprendizado estatístico das premissas de infraestrutura, como área de painéis, eficiência fotovoltaica, área do rotor, densidade do ar e altura de cubo.

## Conformidade Com A Especificação

| Requisito técnico | Atendimento no projeto |
|---|---|
| Leitura de base de dados | O pipeline baixa, extrai e lê os arquivos históricos do INMET por meio dos módulos `src/download_inmet.py`, `src/extract_inmet.py` e `src/parse_inmet.py`. |
| Estatísticas descritivas | São produzidos relatórios de cobertura, taxas de ausência e resumos por estação na pasta `reports/`. |
| Tratamento de dados | O projeto padroniza colunas, datas, encodings, valores sentinela, variáveis físicas, flags de qualidade, registros inválidos e outliers em `src/clean_inmet.py`. |
| Visualizações analíticas | O pipeline gera visualizações estáticas de distribuição, séries anuais e distribuição espacial em `reports/`, além das visualizações interativas do dashboard. |
| Mínimo de cinco visualizações com interpretação | A seção "Análise Exploratória E Visualizações" documenta a finalidade analítica de cada visualização gerada. |
| Holdout | A avaliação utiliza holdout temporal, reservando os anos mais recentes para teste final. |
| Validação cruzada | A busca de hiperparâmetros usa validação cruzada temporal expansiva por ano. |
| Leave-One-Out quando aplicável | O Leave-One-Out não foi adotado por ser inadequado a séries temporais meteorológicas e computacionalmente custoso para o volume de dados utilizado. |
| Random Search e/ou Grid Search | O Random Forest utiliza `RandomizedSearchCV` e refino com `GridSearchCV`; a MLP utiliza busca ampla e etapa de refino. |
| Treinamento de ao menos dois modelos | São comparados Random Forest e Multilayer Perceptron, além de uma baseline climatológica determinística. |
| Métricas apropriadas | São calculadas MAE, RMSE, NRMSE, R2, viés médio, MedAE, sMAPE e `balanced_nrmse`. |
| MLflow | Os experimentos registram parâmetros, métricas, artefatos, modelos e múltiplas execuções em `artifacts/modeling/<run_id>/`. |
| Dashboard interativo | A aplicação Streamlit em `app/app_dashboard.py` apresenta mapas, rankings, erros e séries temporais por modelo. |
| Docker | O projeto disponibiliza `Dockerfile` e `docker-compose.yml` para execução da dashboard e do MLflow UI em ambiente conteinerizado. |
| Relatório acadêmico | O artigo no formato SBC está disponível em `dev-docs/artigo_projeto_ml_sbc.docx`. |

## Estrutura Do Repositório

```text
data/
  raw/          arquivos originais baixados ou adicionados manualmente
  extracted/    CSVs extraídos dos arquivos compactados do INMET
  silver/       dados horários padronizados, selecionados e tratados
  gold/         dados agregados e bases analíticas para modelagem
notebooks/
  modelos/      notebooks de baseline, Random Forest, MLP e comparação
src/             código-fonte do pipeline, tratamento e modelagem
app/             dashboard interativo em Streamlit
reports/         relatórios de qualidade e visualizações de EDA
runs/            notebooks executados por rodada de treinamento
artifacts/
  modeling/      métricas, predições, modelos, logs e artefatos MLflow
dev-docs/        especificação, artigo e documentos de referência
Dockerfile
docker-compose.yml
requirements.txt
README.md
```

## Fonte E Recorte Dos Dados

A fonte primária é a página oficial de dados históricos anuais do INMET:

https://portal.inmet.gov.br/dadoshistoricos

O projeto utiliza arquivos de estações automáticas e seleciona o recorte espacial de Pernambuco. A execução consolidada incluída no repositório utiliza a base diária tratada e os artefatos de modelagem referentes ao identificador `20260613-134558`.

Caso o download automático falhe por alteração no portal do INMET, os arquivos anuais podem ser baixados manualmente e colocados em `data/raw/`.

## Preparação Do Ambiente Local

A execução local requer a criação de um ambiente Python e a instalação das dependências.

No Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Em Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Execução Do Pipeline De Dados

Executar o pipeline completo para Pernambuco usando apenas `UF == PE`:

```powershell
python -m src.main --start-year 2003 --end-year 2025 --region PE --mode uf
```

Executar com caixa delimitadora ao redor de Pernambuco:

```powershell
python -m src.main --start-year 2003 --end-year 2025 --region PE --mode buffer
```

Executar com caixa delimitadora personalizada:

```powershell
python -m src.main --mode buffer --bbox -9.75 -6.75 -42.25 -34.50
```

Executar usando arquivos já baixados em `data/raw/`:

```powershell
python -m src.main --start-year 2003 --end-year 2025 --skip-download
```

Executar usando CSVs já extraídos em `data/extracted/`:

```powershell
python -m src.main --start-year 2003 --end-year 2025 --region PE --mode uf --use-extracted
```

Parâmetros opcionais de tratamento:

```powershell
python -m src.main --nighttime-solar-policy zero
python -m src.main --outlier-method zscore
python -m src.main --keep-outliers
```

## Saídas Do Pipeline

| Arquivo | Descrição |
|---|---|
| `data/silver/stations_selected.csv` | Catálogo das estações selecionadas, incluindo código, município, UF, coordenadas, altitude e anos disponíveis. |
| `data/silver/inmet_hourly_standardized.parquet` | Base horária padronizada após leitura robusta dos CSVs. |
| `data/silver/inmet_pe_hourly_flagged.parquet` | Base horária com flags de qualidade. |
| `data/silver/inmet_pe_hourly_clean.parquet` | Base horária limpa, com valores inválidos e outliers tratados conforme configuração. |
| `data/gold/inmet_pe_daily.csv` | Base diária por estação, com irradiação solar, vento médio, horas válidas e taxas de ausência. |
| `data/gold/inmet_pe_station_annual_summary.csv` | Resumo anual por estação. |
| `data/gold/inmet_pe_station_historical_summary.csv` | Resumo histórico por estação. |
| `data/gold/ml_dataset_station_day.csv` | Base de modelagem com uma linha por estação-dia e atributos de calendário. |
| `data/gold/ml_dataset_station_day_normalized.csv` | Versão normalizada da base de modelagem. |
| `data/gold/normalization_params.csv` | Parâmetros de normalização utilizados. |

## Tratamento E Organização Dos Dados

O pipeline segue uma organização inspirada em arquitetura medalhão:

| Camada | Conteúdo |
|---|---|
| Bronze | Arquivos originais em `data/raw/` e CSVs extraídos em `data/extracted/`. |
| Silver | Dados horários padronizados, filtrados para Pernambuco e enriquecidos com metadados e flags de qualidade. |
| Gold | Agregações diárias, resumos históricos e bases prontas para análise e modelagem. |

As principais decisões de tratamento são:

- detecção automática da linha inicial das tabelas do INMET;
- leitura com múltiplos encodings comuns;
- conversão de vírgula decimal para ponto;
- padronização de nomes de colunas entre anos;
- tratamento de valores sentinela como `9999`, `9999.0`, `-9999`, `Null` e strings vazias;
- invalidação de vento negativo ou acima do limite configurado;
- invalidação de radiação solar negativa ou acima do limite configurado;
- conversão dos horários UTC para horário local por subtração fixa de três horas;
- marcação ou remoção de outliers por estação e variável, usando IQR por padrão.

## Análise Exploratória E Visualizações

Os relatórios de análise exploratória e qualidade são gravados em `reports/`.

| Arquivo | Interpretação analítica |
|---|---|
| `data_count_by_station.csv` | Avalia a cobertura temporal e a quantidade de registros válidos por estação, permitindo identificar lacunas de disponibilidade. |
| `missing_rate_by_station.csv` | Resume as taxas de ausência para vento e radiação solar, orientando a avaliação de qualidade antes da modelagem. |
| `selected_stations_map.png` | Representa a distribuição espacial das estações selecionadas em Pernambuco, evidenciando que a análise se restringe aos municípios monitorados. |
| `hist_wind_speed.png` | Descreve a distribuição da velocidade do vento a 10 m, auxiliando a identificação de concentração de valores, caudas e possíveis anomalias. |
| `hist_daily_solar.png` | Descreve a distribuição da irradiação solar diária, permitindo observar variabilidade e dias de baixa disponibilidade solar. |
| `annual_mean_wind.png` | Apresenta a evolução anual média do vento, apoiando a análise de variações temporais agregadas. |
| `annual_mean_solar.png` | Apresenta a evolução anual média da irradiação solar, apoiando a análise de estabilidade e variação do recurso solar. |

Além das visualizações estáticas, a dashboard fornece mapas, rankings, análise de erro e séries temporais interativas.

## Modelagem

Os experimentos de modelagem estão organizados em notebooks:

| Notebook | Descrição |
|---|---|
| `notebooks/modelos/00_baseline_climatologica.ipynb` | Implementa uma baseline determinística baseada em médias históricas por estação, dia do ano e mês, com fallbacks globais. |
| `notebooks/modelos/01_random_forest.ipynb` | Treina um `RandomForestRegressor` multi-saída com busca e refino de hiperparâmetros. |
| `notebooks/modelos/02_mlp.ipynb` | Treina um `MLPRegressor` multi-saída com normalização de atributos e alvos. |
| `notebooks/modelos/03_comparacao_modelos.ipynb` | Consolida predições, métricas e comparações entre baseline, Random Forest e MLP. |

Os atributos utilizados incluem informações de localização, calendário e histórico, como `station_code`, latitude, longitude, altitude, mês, dia do ano, codificações cíclicas e estatísticas históricas dos alvos físicos.

## Estratégia De Validação

A validação foi definida de modo a respeitar a estrutura temporal das séries meteorológicas.

- **Holdout temporal:** os anos mais recentes são reservados para teste final.
- **Validação cruzada temporal expansiva:** no conjunto de treino, cada fold utiliza anos anteriores para treinamento e o ano seguinte para validação.
- **Leave-One-Out:** não foi aplicado por não representar adequadamente a previsão de anos futuros, além de apresentar custo elevado para a base utilizada.
- **Busca de hiperparâmetros:** Random Forest utiliza `RandomizedSearchCV` e refino por `GridSearchCV`; MLP utiliza busca ampla e etapa de refino.

Essa estratégia reduz o risco de vazamento temporal, pois evita que informações de anos futuros sejam usadas no treinamento de modelos avaliados em anos anteriores.

## Métricas

As métricas avaliadas são:

```text
MAE
RMSE
NRMSE
R2
viés médio
MedAE
sMAPE
balanced_nrmse
```

As métricas são calculadas em dois níveis:

- **Métricas físicas:** irradiação solar diária por metro quadrado e velocidade média diária do vento.
- **Métricas energéticas:** geração solar, eólica e híbrida em kWh/dia.

O critério principal de comparação é o `balanced_nrmse`, definido como a média do NRMSE entre os alvos avaliados.

## Resultados Da Execução Consolidada

A execução consolidada utilizada para demonstração possui identificador `20260613-134558`.

Os resultados indicam que o Random Forest apresentou melhor desempenho agregado nos alvos físicos e na geração híbrida. A baseline climatológica permaneceu competitiva e obteve melhor desempenho para irradiação solar isolada, enquanto Random Forest e MLP apresentaram melhor desempenho para vento.

Arquivos principais da comparação:

```text
artifacts/modeling/20260613-134558/evaluation/model_comparison/model_comparison_metrics_20260613-225146.csv
artifacts/modeling/20260613-134558/evaluation/model_comparison/model_predictions_consolidated_wide_20260613-225146.csv
artifacts/modeling/20260613-134558/evaluation/model_comparison/model_predictions_consolidated_long_20260613-225146.csv
```

## MLOps E MLflow

Os experimentos registram parâmetros, métricas, artefatos e modelos no MLflow. O runner de modelagem cria uma pasta própria para cada execução:

```text
runs/<run_id>/
artifacts/modeling/<run_id>/
```

Principais artefatos:

```text
artifacts/modeling/<run_id>/mlflow.db
artifacts/modeling/<run_id>/mlruns/
artifacts/modeling/<run_id>/evaluation/
artifacts/modeling/<run_id>/logs/
```

O banco `mlflow.db` e a pasta `mlruns/` permitem abrir o MLflow UI e consultar experimentos, métricas, parâmetros, modelos serializados e arquivos auxiliares.

## Execução Com Docker

A configuração Docker disponibiliza a solução já treinada, incluindo dashboard, MLflow UI, métricas, modelos e CSV consolidado da execução `20260613-134558`.

Construir a imagem e iniciar os serviços:

```powershell
docker compose up --build
```

Acessar os serviços:

```text
Dashboard Streamlit: http://localhost:8501
MLflow UI:           http://localhost:5000
```

Encerrar os serviços:

```powershell
docker compose down
```

## Treinamento Dos Modelos

O treinamento dos modelos é executado localmente por meio do script `scripts/run_training_notebooks.ps1`, que roda os notebooks de modelagem com `nbconvert`, cria um novo identificador de execução, salva os notebooks executados e registra métricas, parâmetros, modelos e artefatos no MLflow.

Antes de iniciar o treinamento, confirme que:

- o ambiente Python foi criado e as dependências de `requirements.txt` foram instaladas;
- a base tratada `data/gold/inmet_pe_daily.csv` existe;
- as constantes energéticas em `src/modeling/training_config.py` foram revisadas.

Executar todos os treinamentos e a etapa de comparação:

```powershell
.\scripts\run_training_notebooks.ps1
```

Executar omitindo etapas específicas:

```powershell
.\scripts\run_training_notebooks.ps1 -SkipBaseline
.\scripts\run_training_notebooks.ps1 -SkipRandomForest
.\scripts\run_training_notebooks.ps1 -SkipMlp
.\scripts\run_training_notebooks.ps1 -SkipComparison
```

Executar e desligar a máquina após conclusão bem-sucedida:

```powershell
.\scripts\run_training_notebooks.ps1 -ShutdownOnSuccess
```

Ao final, os principais resultados são salvos em:

```text
runs/<run_id>/notebooks/
artifacts/modeling/<run_id>/logs/
artifacts/modeling/<run_id>/evaluation/
artifacts/modeling/<run_id>/mlflow.db
artifacts/modeling/<run_id>/mlruns/
```

## Execução Da Aplicação Sem Docker

A interface interativa foi desenvolvida com Streamlit e está implementada em `app/app_dashboard.py`.

Funcionalidades principais:

- seleção da origem da previsão: valores reais, baseline, Random Forest ou MLP;
- mapas de potencial solar, eólico e híbrido;
- rankings municipais por potencial energético;
- análise de erro médio absoluto por cidade;
- séries temporais comparando valores reais e predições dos modelos.

Executar localmente:

```powershell
streamlit run app/app_dashboard.py
```

Por padrão, a dashboard consome o consolidado:

```text
artifacts/modeling/20260613-134558/evaluation/model_comparison/model_predictions_consolidated_wide_20260613-225146.csv
```

Para utilizar outro arquivo consolidado:

```powershell
$env:PREDICTIONS_CSV="C:\caminho\para\model_predictions_consolidated_wide.csv"
streamlit run app/app_dashboard.py
```

## Relatório Acadêmico

O artigo do projeto está disponível em:

```text
docs/artigo_ML.pdf
```
