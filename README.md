# Projeto ML - INMET Pernambuco

Pipeline Python para baixar, organizar, limpar e gerar bases meteorologicas horarias do INMET para o estado de Pernambuco, com foco em velocidade do vento a 10 m, radiacao solar global, metadados das estacoes e data/hora da medicao.

O projeto foi pensado para apoiar uma etapa de aquisicao/tratamento de dados para Machine Learning e analise espacial de potencial solar, eolico ou hibrido no Nordeste do Brasil.

## Fonte Dos Dados

A fonte principal e a pagina oficial de historicos anuais do INMET:

https://portal.inmet.gov.br/dadoshistoricos

Na data de criacao deste projeto, a pagina lista arquivos anuais de estacoes automaticas, por exemplo `ANO 2003 (AUTOMATICA)`, `ANO 2025 (AUTOMATICA)` e `ANO 2026 (AUTOMATICA) - Ate 31/05/2026`. O downloader tenta descobrir os links reais nessa pagina antes de baixar. Se o portal mudar, o pipeline registra erro claro e permite o modo manual.

## Estrutura

```text
data/
  raw/          ZIPs ou CSVs baixados manualmente ou pelo pipeline
  extracted/    CSVs extraidos dos ZIPs
  silver/       camada prata: dados padronizados, selecionados e limpos
  gold/         camada ouro: bases agregadas e normalizadas para analise/ML
  interim/      legado/espaco para bases intermediarias exploratorias
  processed/    legado/espaco para saidas finais antigas
notebooks/      espaco para exploracao opcional
reports/        CSVs e PNGs de qualidade
src/            codigo modular do pipeline
requirements.txt
README.md
```

## Instalacao

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Em Linux/macOS, use `source .venv/bin/activate`.

## Como Executar

Rodar o pipeline completo para Pernambuco usando somente `UF == PE`:

```bash
python -m src.main --start-year 2003 --end-year 2025 --region PE --mode uf
```

Rodar com bounding box ao redor de Pernambuco:

```bash
python -m src.main --start-year 2003 --end-year 2025 --region PE --mode buffer
```

Usar um bounding box personalizado:

```bash
python -m src.main --mode buffer --bbox -9.75 -6.75 -42.25 -34.50
```

Se o download automatico falhar, baixe os ZIPs anuais manualmente na pagina oficial do INMET, coloque-os em `data/raw/` e rode:

```bash
python -m src.main --start-year 2003 --end-year 2025 --skip-download
```

Tambem e aceito colocar CSVs ja extraidos diretamente em `data/raw/`.

Quando os CSVs ja estiverem em `data/extracted/`, pule download e extracao e gere diretamente as camadas silver/gold:

```bash
python -m src.main --start-year 2003 --end-year 2025 --region PE --mode uf --use-extracted
```

## Arquivos Gerados

`data/silver/stations_selected.csv`: estacoes selecionadas com codigo, nome, municipio, UF, latitude, longitude, altitude e anos disponiveis.

`data/silver/inmet_hourly_standardized.parquet`: base horaria padronizada apos leitura robusta dos CSVs do INMET.

`data/silver/inmet_pe_hourly_flagged.parquet`: base horaria padronizada com flags de qualidade.

`data/silver/inmet_pe_hourly_clean.parquet`: base horaria limpa. Por padrao, valores invalidos e outliers marcados sao convertidos para `NaN`.

`data/gold/inmet_pe_daily.csv`: agregacao diaria por estacao, com irradiacao solar diaria, vento medio, vento desvio padrao, horas validas e taxas de ausencia.

`data/gold/inmet_pe_station_annual_summary.csv`: resumo anual por estacao e ano.

`data/gold/inmet_pe_station_historical_summary.csv`: resumo historico por estacao.

`data/gold/ml_dataset_station_day.csv`: base pronta para ML, com uma linha por estacao-dia e features ciclicas de mes e dia do ano.

`data/gold/ml_dataset_station_day_normalized.csv`: versao gold com features numericas normalizadas por min-max.

`data/gold/normalization_params.csv`: parametros usados na normalizacao min-max.

## Modelo Medalhao

Neste projeto, a camada bronze corresponde aos arquivos originais em `data/raw/` e aos CSVs extraidos em `data/extracted/`. Esses arquivos devem ser preservados como vieram do INMET.

A camada silver fica em `data/silver/` e contem dados padronizados, selecionados para Pernambuco ou buffer, com metadados estruturados, timestamp local/UTC, variaveis principais e flags de qualidade.

A camada gold fica em `data/gold/` e contem dados agregados e prontos para consumo analitico ou Machine Learning, incluindo a base diaria, resumos anuais/historicos e a versao normalizada do dataset estação-dia.

## Decisoes De Tratamento

O parser detecta automaticamente a linha inicial da tabela dos CSVs do INMET, tenta encodings comuns (`utf-8-sig`, `latin1`, `iso-8859-1`, `cp1252`), converte virgula decimal para ponto e padroniza nomes de colunas levemente diferentes entre anos.

Valores `9999`, `9999.0`, `-9999`, `Null`, strings vazias e equivalentes sao tratados como ausentes. Tambem sao marcados como invalidos vento menor que zero, vento acima do limite configuravel, radiacao solar menor que zero e radiacao solar acima do limite configuravel.

Os horarios do INMET sao interpretados como UTC e convertidos para horario local de Pernambuco/Brasilia por subtracao fixa de 3 horas. Para estudos historicos muito sensiveis, verifique periodos antigos com horario de verao.

Por padrao, a radiacao solar fora de 05:00 a 18:00 no horario local e definida como `NaN` antes da agregacao diaria. Para usar zero durante a noite, execute com:

```bash
python -m src.main --nighttime-solar-policy zero
```

Outliers sao marcados por estacao e variavel. O metodo padrao e IQR; tambem ha z-score:

```bash
python -m src.main --outlier-method zscore
```

Para manter outliers na base limpa, use:

```bash
python -m src.main --keep-outliers
```

## Relatorios

Os relatorios sao salvos em `reports/`:

```text
data_count_by_station.csv
missing_rate_by_station.csv
annual_mean_wind.png
annual_mean_solar.png
hist_wind_speed.png
hist_daily_solar.png
selected_stations_map.png
```

## Limitacoes

Os dados do INMET podem ter falhas longas, mudancas de disponibilidade por estacao, alteracoes de nomenclatura entre anos, metadados incompletos e valores sentinela. A radiacao solar horaria pode exigir verificacao adicional por estacao antes de uso em modelos finais.

A conversao UTC para horario local usa offset fixo de -3 horas. Essa escolha e simples e reproduzivel, mas nao representa horario de verao em periodos historicos.

## Modelos De Machine Learning

A etapa de modelagem usa apenas os dados tratados disponiveis em `data/gold/inmet_pe_daily.csv`. Os modelos nao preveem diretamente a geracao em kWh. Eles preveem as variaveis fisicas usadas no calculo energetico:

```text
solar_daily_kwh_m2_day
wind_daily_mean_ms
```

Depois da predicao, a geracao solar, eolica e hibrida em kWh/dia e calculada de forma deterministica com as constantes configuradas em `src/modeling/training_config.py`.

### Modelos Implementados

Os notebooks ficam em `notebooks/modelos/`:

```text
00_baseline_climatologica.ipynb
01_random_forest.ipynb
02_mlp.ipynb
03_comparacao_modelos.ipynb
```

`00_baseline_climatologica.ipynb`: cria uma baseline climatologica baseada em medias historicas por estacao, dia do ano e mes, com fallbacks globais.

`01_random_forest.ipynb`: treina um `RandomForestRegressor` multi-saida com busca de hiperparametros por `RandomizedSearchCV` e refino por `GridSearchCV`.

`02_mlp.ipynb`: treina uma `MLPRegressor` multi-saida com normalizacao de features e alvos, busca ampla e busca refinada de hiperparametros.

`03_comparacao_modelos.ipynb`: consolida as predicoes da baseline, Random Forest e MLP em arquivos comparaveis por estacao e data.

### Validacao E Metricas

A validacao usa separacao temporal. Os anos mais recentes ficam como teste final, enquanto os anos anteriores sao usados para treino e validacao cruzada temporal expansiva.

Essa estrategia foi escolhida porque os dados sao series temporais por estacao. Um particionamento aleatorio poderia misturar passado e futuro, criando vazamento temporal e uma avaliacao otimista. O holdout temporal avalia a capacidade do modelo de generalizar para anos futuros, que e o uso esperado da solucao.

O Leave-One-Out nao foi aplicado porque nao e adequado para este contexto: alem do custo computacional alto para dezenas de milhares de linhas, ele quebraria a estrutura temporal do problema e nao representaria a previsao de anos futuros. Por isso, a validacao cruzada temporal expansiva e a alternativa metodologicamente mais coerente.

As metricas registradas incluem:

```text
MAE
RMSE
NRMSE
R2
Bias medio
MedAE
sMAPE
```

As metricas sao calculadas em dois grupos:

```text
metricas fisicas: irradiacao solar e velocidade media do vento
metricas energeticas: geracao solar, eolica e hibrida em kWh/dia
```

O criterio principal para escolher o melhor modelo e o menor `balanced_nrmse`, calculado como a media do NRMSE entre os alvos avaliados. Em caso de resultados proximos, sao observados RMSE, MAE, R2 e a simplicidade operacional do modelo. A baseline climatologica continua como referencia forte: se um modelo de ML nao superar a baseline de forma relevante, a baseline pode ser a alternativa mais adequada para producao inicial.

### MLflow E Artefatos

Os treinamentos registram parametros, metricas, artefatos e modelos no MLflow. Ao usar o runner, cada execucao cria uma pasta propria em:

```text
runs/<run_id>/
artifacts/modeling/<run_id>/
```

Os principais resultados ficam em:

```text
artifacts/modeling/<run_id>/evaluation/
artifacts/modeling/<run_id>/mlruns/
artifacts/modeling/<run_id>/mlflow.db
```

O consolidado dos modelos e salvo em:

```text
artifacts/modeling/<run_id>/evaluation/model_comparison/
```

com os arquivos:

```text
model_predictions_consolidated_wide_*.csv
model_predictions_consolidated_long_*.csv
model_comparison_metrics_*.csv
model_comparison_manifest_*.json
```

Os CSVs consolidados de predicoes incluem valores reais, predicoes, erro, erro absoluto e NRMSE por municipio, modelo e alvo.

### Demonstração Com Docker

O projeto inclui uma configuração Docker para demonstrar a solução ja treinada, sem baixar dados, processar bases brutas ou executar notebooks de treinamento. A imagem inclui a execução oficial `20260613-134558`, com `mlflow.db`, `mlruns`, metricas, modelos e CSV consolidado.

Para construir e subir a dashboard e o MLflow:

```powershell
docker compose up --build
```

Após a inicialização, acesse:

```text
Dashboard Streamlit: http://localhost:8501
MLflow UI:           http://localhost:5000
```

A dashboard usa por padrão:

```text
artifacts/modeling/20260613-134558/evaluation/model_comparison/model_predictions_consolidated_wide_20260613-225146.csv
```

Para apontar a dashboard para outro CSV consolidado, defina a variável `PREDICTIONS_CSV` no serviço `dashboard` do `docker-compose.yml`.

### Como Executar Os Modelos

Antes de treinar, confirme os valores em `ENERGY_CONFIG`, dentro de `src/modeling/training_config.py`.

Para executar o pipeline completo de modelagem:

```powershell
.\scripts\run_training_notebooks.ps1
```

Para executar e desligar a maquina ao finalizar com sucesso:

```powershell
.\scripts\run_training_notebooks.ps1 -ShutdownOnSuccess
```

Para pular alguma etapa:

```powershell
.\scripts\run_training_notebooks.ps1 -SkipBaseline
.\scripts\run_training_notebooks.ps1 -SkipRandomForest
.\scripts\run_training_notebooks.ps1 -SkipMlp
.\scripts\run_training_notebooks.ps1 -SkipComparison
```

### Observacoes Dos Resultados

Na execucao avaliada, a baseline foi mais forte para a previsao solar, enquanto o Random Forest teve melhor desempenho para vento e geracao hibrida. Isso e coerente com as features disponiveis: sem previsao meteorologica futura, a irradiacao solar diaria fica muito dependente do comportamento historico sazonal; ja o vento apresentou mais estrutura capturavel por localizacao, altitude, calendario e historico.

Para a proxima etapa, a dashboard deve consumir os CSVs consolidados em `evaluation/model_comparison/` para exibir metricas, previsoes e comparacoes entre baseline, Random Forest e MLP.
