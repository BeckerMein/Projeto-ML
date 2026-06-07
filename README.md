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
  interim/      bases intermediarias padronizadas
  processed/    bases finais tratadas
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

## Arquivos Gerados

`data/processed/stations_selected.csv`: estacoes selecionadas com codigo, nome, municipio, UF, latitude, longitude, altitude e anos disponiveis.

`data/processed/inmet_pe_hourly_flagged.parquet`: base horaria padronizada com flags de qualidade.

`data/processed/inmet_pe_hourly_clean.parquet`: base horaria limpa. Por padrao, valores invalidos e outliers marcados sao convertidos para `NaN`.

`data/processed/inmet_pe_daily.csv`: agregacao diaria por estacao, com irradiacao solar diaria, vento medio, vento desvio padrao, horas validas e taxas de ausencia.

`data/processed/inmet_pe_station_annual_summary.csv`: resumo anual por estacao e ano.

`data/processed/inmet_pe_station_historical_summary.csv`: resumo historico por estacao.

`data/processed/ml_dataset_station_day.csv`: base pronta para ML, com uma linha por estacao-dia e features ciclicas de mes e dia do ano.

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

## Proximos Passos

Para analise espacial, use `stations_selected.csv` e os resumos anuais/historicos como entrada para interpolacao por IDW, krigagem ou modelos espaciais. Para Machine Learning, comece por `ml_dataset_station_day.csv`, avalie taxas de ausencia, crie particoes temporais por ano e considere adicionar covariaveis externas como altitude refinada, distancia do litoral, cobertura do solo e reanalises climaticas.

