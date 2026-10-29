import altair as alt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import streamlit as st

# Garantir o download do léxico do VADER
try:
  nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
  nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

# Configuração da página do site
st.set_page_config(
    page_title='Painel de Avaliações e PLN', page_icon='📊', layout='wide'
)

st.title('📚 Painel de Avaliações & Validação de PLN (VADER)')
st.write(
    'Explore as avaliações organizadas em cartões e analise os gráficos'
    ' comparativos lado a lado.'
)


# Função para converter a polaridade do VADER em nota de 1 a 5
def calcular_nota_vader(texto):
  if not texto or pd.isna(texto):
    return 3.0
  compound = sia.polarity_scores(str(texto))['compound']
  if compound <= -0.5:
    return 1.0
  elif compound <= -0.1:
    return 2.0
  elif compound < 0.1:
    return 3.0
  elif compound < 0.5:
    return 4.0
  else:
    return 5.0


# Carregar dados com cache para otimizar a performance
@st.cache_data
def carregar_dados():
  df = pd.read_csv('books-15k.csv')
  df['review/text'] = df['review/text'].fillna('Sem texto')
  df['Title'] = df['Title'].fillna('Livro sem título')
  df['review/score'] = df['review/score'].fillna(3.0)

  # Processando uma amostra para garantir velocidade instantânea no app
  df_sample = df.head(15000).copy()

  # Calcula a nota do VADER baseada exclusivamente no texto
  df_sample['vader_score'] = df_sample['review/text'].apply(
      calcular_nota_vader
  )
  return df_sample


df = carregar_dados()

# Criando as Abas do Site
aba_avaliacoes, aba_comparativo = st.tabs(
    ['📖 Listagem de Avaliações', '📊 Comparativo e Gráficos']
)

with aba_avaliacoes:
  st.header('Listagem de Avaliações')
  st.write(
      'As informações de cada livro e suas notas estão agrupadas em cartões'
      ' individuais limpos e organizados.'
  )

  col_f1, col_f2 = st.columns([1, 2])
  with col_f1:
    filtro_nota = st.selectbox(
        'Filtrar por Nota VADER (Texto):', ['Todas', 1.0, 2.0, 3.0, 4.0, 5.0]
    )

  df_exibicao = df
  if filtro_nota != 'Todas':
    df_exibicao = df[df['vader_score'] == float(filtro_nota)]

  st.markdown(f'**Mostrando {len(df_exibicao)} avaliações:**')
  st.markdown('---')

  # Exibição em cartões encapsulados
  for idx, row in df_exibicao.head(30).iterrows():
    with st.container(border=True):
      st.subheader(f"📖 {row['Title']}")

      col1, col2 = st.columns(2)
      with col1:
        st.markdown(
            f"⭐ **Nota Original (Dataset):** `{row['review/score']}`"
        )
      with col2:
        st.markdown(f"🤖 **Nota VADER (Texto):** `{row['vader_score']}`")

      st.markdown(f"*{row['review/summary']}*")
      st.write(row['review/text'])

with aba_comparativo:
  st.header('📊 Painel Comparativo e Desempenho do Modelo')
  st.write(
      'Análise consolidada do volume geral de avaliações e da quantidade de'
      ' acertos exatos obtidos pelo algoritmo de PLN.'
  )

  # Pré-cálculos
  notas_possiveis = [1.0, 2.0, 3.0, 4.0, 5.0]
  contagem_original = (
      df['review/score'].value_counts().reindex(notas_possiveis, fill_value=0)
  )
  contagem_vader = (
      df['vader_score'].value_counts().reindex(notas_possiveis, fill_value=0)
  )

  df['acerto_exato'] = df['vader_score'] == df['review/score']
  acertos_nota = (
      df[df['acerto_exato']]['review/score']
      .value_counts()
      .reindex(notas_possiveis, fill_value=0)
  )

  # --- Bloco 1: Métricas de Resumo ---
  total_avaliacoes = len(df)
  total_acertos = df['acerto_exato'].sum()
  acuracia_geral = (total_acertos / total_avaliacoes * 100).round(2)

  m1, m2, m3 = st.columns(3)
  m1.metric('Total de Avaliações', f'{total_avaliacoes}')
  m2.metric('Acertos Exatos do VADER', f'{total_acertos}')
  m3.metric('Acurácia Geral', f'{acuracia_geral}%')

  st.markdown('---')

  # --- Bloco 2: Gráfico 1 - Volume Geral (Nota Original vs VADER) ---
  st.subheader('📈 1. Comparativo de Volume: Notas Originais vs. VADER')
  st.write(
      'Barras agrupadas lado a lado comparando a distribuição geral das notas'
      ' em cada categoria.'
  )

  df_longo_volume = pd.DataFrame({
      'Nota': [str(n) for n in notas_possiveis] * 2,
      'Quantidade': list(contagem_original) + list(contagem_vader),
      'Tipo': ['Nota Original (Score)'] * 5 + ['Nota VADER (Texto)'] * 5,
  })

  chart_volume = (
      alt.Chart(df_longo_volume)
      .mark_bar()
      .encode(
          x=alt.X('Nota:N', title='Estrelas / Nota', sort=None),
          y=alt.Y('Quantidade:Q', title='Número de Avaliações'),
          color=alt.Color(
              'Tipo:N',
              title='Legenda',
              scale=alt.Scale(range=['#1f77b4', '#ff7f0e']),
          ),
          xOffset='Tipo:N',
      )
      .properties(height=350)
  )

  st.altair_chart(chart_volume, use_container_width=True)

  st.markdown('---')

  # --- Bloco 3: Gráfico 2 - Acertos Exatos do VADER (No mesmo estilo lado a lado) ---
  st.subheader('🎯 2. Comparativo de Acertos Exatos do VADER por Nota')
  st.write(
      'Gráfico lado a lado mostrando o **Total de Avaliações** do dataset em'
      ' comparação direta com a **Quantidade de Acertos Exatos** obtidos pelo'
      ' VADER em cada nota.'
  )

  df_longo_acertos = pd.DataFrame({
      'Nota': [str(n) for n in notas_possiveis] * 2,
      'Quantidade': list(contagem_original) + list(acertos_nota),
      'Metrica': ['Total no Dataset'] * 5 + ['Acertos Exatos VADER'] * 5,
  })

  chart_acertos = (
      alt.Chart(df_longo_acertos)
      .mark_bar()
      .encode(
          x=alt.X('Nota:N', title='Estrelas / Nota', sort=None),
          y=alt.Y('Quantidade:Q', title='Número de Avaliações'),
          color=alt.Color(
              'Metrica:N',
              title='Legenda',
              scale=alt.Scale(
                  range=['#aec7e8', '#2ca02c']
              ),  # Azul claro e Verde para acertos
          ),
          xOffset='Metrica:N',
      )
      .properties(height=350)
  )

  st.altair_chart(chart_acertos, use_container_width=True)

  st.markdown('---')

  # --- Bloco 4: Tabela Detalhada de Desempenho ---
  st.subheader('📋 Tabela Resumo de Desempenho e Taxa de Acerto')

  df_detalhe = pd.DataFrame({
      'Total no Dataset': contagem_original,
      'Acertos do VADER': acertos_nota,
  })
  df_detalhe['Taxa de Acerto (%)'] = (
      df_detalhe['Acertos do VADER'] / df_detalhe['Total no Dataset'] * 100
  ).round(2)

  st.dataframe(df_detalhe, use_container_width=True)