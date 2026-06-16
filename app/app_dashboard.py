import os
import plotly.express as px
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Painel de Energia Renovável PE", layout="wide", page_icon="⚡"
)

st.title("Sistema de Suporte à Decisão: Energia Renovável em PE")
st.markdown(
    "Esta aplicação consome as previsões consolidadas dos modelos de *Machine Learning* e exibe as estimativas de geração energética."
)

# --- CONFIGURAÇÃO DE CAMINHO ---
PATH_PREDICOES = "../artifacts/modeling/20260613-134558/evaluation/model_comparison/model_predictions_consolidated_wide_20260613-225146.csv"

# --- CARREGAR DADOS ---
@st.cache_data
def carregar_dados():
    if not os.path.exists(PATH_PREDICOES):
        st.error(f"Ficheiro não encontrado no caminho: `{PATH_PREDICOES}`")
        st.info("Coloque o ficheiro CSV no caminho correto ou atualize a variável PATH_PREDICOES no código.")
        st.stop()
        
    df = pd.read_csv(PATH_PREDICOES)
    # Converter a coluna de data para o formato datetime para o gráfico de linhas
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    return df

df_analise = carregar_dados()

# --- BARRA LATERAL: PARÂMETROS INTERATIVOS ---
st.sidebar.header("Configurações")
st.sidebar.subheader("Modelo Preditivo")

modelo_selecionado = st.sidebar.selectbox(
    "Escolha a origem da previsão (Para Mapas e Rankings):", 
    ["Valores Reais (Histórico)", "Random Forest", "MLP", "Baseline Climatológica"]
)

# --- MAPEAMENTO EXATO DAS COLUNAS (Baseado no seu CSV) ---
sufixos_modelo = {
    "Valores Reais (Histórico)": "_actual",
    "Random Forest": "_random_forest",
    "MLP": "_mlp",
    "Baseline Climatológica": "_baseline",
}
sufixo = sufixos_modelo[modelo_selecionado]

# Nomes exatos conforme o ficheiro CSV
col_solar_fisica = f"solar_daily_kwh_m2_day{sufixo}"
col_vento_fisico = f"wind_daily_mean_ms{sufixo}"

# Nomes exatos das colunas energéticas (já calculadas no CSV)
col_solar_gen = f"solar_generation_kwh_day{sufixo}"
col_eolico_gen = f"wind_generation_kwh_day{sufixo}"
col_hibrido_gen = f"hybrid_generation_kwh_day{sufixo}"

# --- AGREGAÇÃO POR MUNICÍPIO ---
df_municipio = df_analise.groupby('city').agg({
    col_solar_gen: 'mean',
    col_eolico_gen: 'mean',
    col_hibrido_gen: 'mean',
    'latitude': 'first',
    'longitude': 'first',
    col_solar_fisica: 'mean',
    col_vento_fisico: 'mean'
}).reset_index()


# --- CRIAÇÃO DAS ABAS ---
tab1, tab2, tab3, tab4 = st.tabs(["Mapas de Potencial", "Rankings de Municípios", "Análise de Erro do Modelo", "Evolução Temporal"])

with tab1:
    st.header(f"Distribuição Espacial - Fonte: {modelo_selecionado}")
    
    tipo_visualizacao = st.radio(
        "Selecione a matriz energética:",
        ["Energia Solar (kWh/dia)", "Energia Eólica (kWh/dia)", "Energia Híbrida (kWh/dia)"],
        horizontal=True,
    )

    mapa_colunas = {
        "Energia Solar (kWh/dia)": col_solar_gen,
        "Energia Eólica (kWh/dia)": col_eolico_gen,
        "Energia Híbrida (kWh/dia)": col_hibrido_gen,
    }
    coluna_alvo = mapa_colunas[tipo_visualizacao]

    # --- CONFIGURAÇÃO DINÂMICA DO HOVER E RÓTULOS AMIGÁVEIS ---
    if tipo_visualizacao == "Energia Solar (kWh/dia)":
        hover_config = {
            "latitude": False, 
            "longitude": False, 
            col_solar_gen: ":.2f",
            col_solar_fisica: ":.2f"
        }
        rotulos = {
            col_solar_gen: "Geração Solar Est. (kWh/dia)",
            col_solar_fisica: "Radiação Média (kWh/m²/dia)",
            "city": "Município"
        }
    elif tipo_visualizacao == "Energia Eólica (kWh/dia)":
        hover_config = {
            "latitude": False, 
            "longitude": False, 
            col_eolico_gen: ":.2f",
            col_vento_fisico: ":.2f"
        }
        rotulos = {
            col_eolico_gen: "Geração Eólica Est. (kWh/dia)",
            col_vento_fisico: "Velocidade do Vento (m/s)",
            "city": "Município"
        }
    else:  # Energia Híbrida (kWh/dia)
        hover_config = {
            "latitude": False, 
            "longitude": False, 
            col_hibrido_gen: ":.2f",
            col_solar_gen: ":.2f",
            col_eolico_gen: ":.2f"
        }
        rotulos = {
            col_hibrido_gen: "Geração Híbrida Total (kWh/dia)",
            col_solar_gen: "Contribuição Solar (kWh/dia)",
            col_eolico_gen: "Contribuição Eólica (kWh/dia)",
            "city": "Município"
        }

    fig = px.scatter_mapbox(
        df_municipio,
        lat="latitude",
        lon="longitude",
        size=coluna_alvo,
        color=coluna_alvo,
        hover_name="city",
        hover_data=hover_config,
        labels=rotulos,
        color_continuous_scale="Plasma" if "Híbrida" in tipo_visualizacao else ("OrYel" if "Solar" in tipo_visualizacao else "Viridis"),
        mapbox_style="carto-positron",
        zoom=6,
        center={"lat": -8.4, "lon": -37.9},
        size_max=35,
        opacity=0.85,
    )
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Top 10 Municípios com Maior Aptidão Energética")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Potencial Solar")
        top_s = df_municipio.sort_values(by=col_solar_gen, ascending=False).head(10).reset_index(drop=True)
        top_s.index = top_s.index + 1
        st.dataframe(top_s[["city", col_solar_gen]].rename(columns={ "city": "Município", col_solar_gen: "Geração Solar (kWh/dia)"}).style.format({"Geração Solar (kWh/dia)": "{:.2f}"}))

    with col2:
        st.subheader("Potencial Eólico")
        top_e = df_municipio.sort_values(by=col_eolico_gen, ascending=False).head(10).reset_index(drop=True)
        top_e.index = top_e.index + 1
        st.dataframe(top_e[["city", col_eolico_gen]].rename(columns={ "city": "Município", col_eolico_gen: "Geração Eólica (kWh/dia)"}).style.format({"Geração Eólica (kWh/dia)": "{:.2f}"}))

    with col3:
        st.subheader("Potencial Híbrido")
        top_h = df_municipio.sort_values(by=col_hibrido_gen, ascending=False).head(10).reset_index(drop=True)
        top_h.index = top_h.index + 1
        st.dataframe(top_h[["city", col_hibrido_gen]].rename(columns={ "city": "Município", col_hibrido_gen: "Geração Híbrida (kWh/dia)"}).style.format({"Geração Híbrida (kWh/dia)": "{:.2f}"}))

with tab3:
    st.header("Métricas de Erro do Modelo")
    
    if modelo_selecionado == "Valores Reais (Histórico)":
        st.info("Você selecionou os valores reais (Ground Truth). As métricas de erro são zero.")
    else:
        st.write(f"Análise dos erros absolutos diários para o modelo **{modelo_selecionado}**.")
        
        col_err_solar = f"solar_daily_kwh_m2_day_abs_error{sufixo}"
        col_err_vento = f"wind_daily_mean_ms_abs_error{sufixo}"
        
        df_erros = df_analise.groupby('city').agg({
            col_err_solar: 'mean',
            col_err_vento: 'mean'
        }).reset_index()
        
        c_err1, c_err2 = st.columns(2)
        with c_err1:
            st.write("Erro Médio Absoluto (MAE) - Irradiação Solar por Cidade")
            fig_err_s = px.bar(df_erros.sort_values(by=col_err_solar, ascending=False).head(15), x='city', y=col_err_solar, labels={'city': 'Cidade', col_err_solar: "MAE da Irradiação Solar (kWh/m²/dia)"})
            fig_err_s.update_layout(yaxis=dict(range=[0, 4]))
            st.plotly_chart(fig_err_s, use_container_width=True)
            
        with c_err2:
            st.write("Erro Médio Absoluto (MAE) - Velocidade do Vento por Cidade")
            fig_err_v = px.bar(df_erros.sort_values(by=col_err_vento, ascending=False).head(15), x='city', y=col_err_vento, labels={'city': 'Cidade', col_err_vento: "MAE da Velocidade do Vento (m/s)"})
            fig_err_v.update_layout(yaxis=dict(range=[0, 4]))
            st.plotly_chart(fig_err_v, use_container_width=True)

with tab4:
    st.header("Comparativo Temporal das Previsões")
    st.markdown("Acompanhe o comportamento das previsões ao longo do tempo (todos os modelos vs Real).")

    col_filt1, _ = st.columns(2)

    with col_filt1:
        cidades_disponiveis = ["Média do Estado"] + sorted(df_analise['city'].unique())
        cidade_filtro = st.selectbox("Localidade:", cidades_disponiveis)

    # Prepara o DataFrame temporário filtrado
    df_temp = df_analise.copy()
    if cidade_filtro != "Média do Estado":
        df_temp = df_temp[df_temp['city'] == cidade_filtro]

    
    df_temp['date'] = pd.to_datetime(df_temp['date'])

    df_temp['periodo'] = (
        df_temp['date']
        .dt.to_period('M')
        .dt.to_timestamp()
    )

    # Agrupa tirando a média do período
    df_plot = df_temp.groupby('periodo').mean(numeric_only=True).reset_index()

    # --- 1. Gráfico de Evolução Solar ---
    st.subheader("Geração Solar (kWh/dia)")
    

    df_solar = df_plot[['periodo', 'solar_generation_kwh_day_actual', 'solar_generation_kwh_day_baseline', 'solar_generation_kwh_day_random_forest', 'solar_generation_kwh_day_mlp']].rename(columns={
        'solar_generation_kwh_day_actual': 'Real',
        'solar_generation_kwh_day_baseline': 'Baseline',
        'solar_generation_kwh_day_random_forest': 'Random Forest',
        'solar_generation_kwh_day_mlp': 'MLP'
    })

    fig_line_solar = px.line(
        df_solar, 
        x='periodo', 
        y=['Real', 'Baseline', 'Random Forest', 'MLP'], # O Plotly agora lê os nomes limpos direto daqui
        labels={'periodo': 'Período', 'value': 'Geração Solar Média', 'variable': 'Modelo'}
    )
    st.plotly_chart(fig_line_solar, use_container_width=True)

    st.divider()

    # --- 2. Gráfico de Evolução Eólica ---
    st.subheader("Geração Eólica (kWh/dia)")
    
    df_wind = df_plot[['periodo', 'wind_generation_kwh_day_actual', 'wind_generation_kwh_day_baseline', 'wind_generation_kwh_day_random_forest', 'wind_generation_kwh_day_mlp']].rename(columns={
        'wind_generation_kwh_day_actual': 'Real',
        'wind_generation_kwh_day_baseline': 'Baseline',
        'wind_generation_kwh_day_random_forest': 'Random Forest',
        'wind_generation_kwh_day_mlp': 'MLP'
    })

    fig_line_wind = px.line(
        df_wind, 
        x='periodo', 
        y=['Real', 'Baseline', 'Random Forest', 'MLP'],
        labels={'periodo': 'Período', 'value': 'Geração Eólica Média', 'variable': 'Modelo'}
    )
    st.plotly_chart(fig_line_wind, use_container_width=True)

    st.divider()

    # --- 3. Gráfico de Evolução Híbrida ---
    st.subheader("Geração Híbrida (kWh/dia)")
    
    df_hybrid = df_plot[['periodo', 'hybrid_generation_kwh_day_actual', 'hybrid_generation_kwh_day_baseline', 'hybrid_generation_kwh_day_random_forest', 'hybrid_generation_kwh_day_mlp']].rename(columns={
        'hybrid_generation_kwh_day_actual': 'Real',
        'hybrid_generation_kwh_day_baseline': 'Baseline',
        'hybrid_generation_kwh_day_random_forest': 'Random Forest',
        'hybrid_generation_kwh_day_mlp': 'MLP'
    })

    fig_line_hybrid = px.line(
        df_hybrid, 
        x='periodo', 
        y=['Real', 'Baseline', 'Random Forest', 'MLP'],
        labels={'periodo': 'Período', 'value': 'Geração Híbrida Média', 'variable': 'Modelo'}
    )
    st.plotly_chart(fig_line_hybrid, use_container_width=True)