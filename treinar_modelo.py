import os
import nltk
import joblib
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. Download das stopwords
nltk.download('stopwords')

NOME_ARQUIVO = 'books-15k.csv'

if not os.path.exists(NOME_ARQUIVO):
    raise FileNotFoundError(f"O arquivo {NOME_ARQUIVO} não foi encontrado na pasta atual.")

# 2. Carregar e tratar dados
df = pd.read_csv(NOME_ARQUIVO)
df['review/score'] = pd.to_numeric(df['review/score'], errors='coerce')
df['texto_completo'] = df['review/summary'].fillna('') + ' ' + df['review/text'].fillna('')
df = df.dropna(subset=['texto_completo', 'review/score'])

df['sentimento'] = df['review/score'].apply(lambda nota: 'boa' if nota >= 4 else 'ruim')

X = df['texto_completo']
y = df['sentimento']

# 3. Divisão Treino e Teste
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Pipeline corrigida (Ajuste o idioma das stopwords se o texto estiver em português)
stop_words_lang = stopwords.words('english') 

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=stop_words_lang, lowercase=True, ngram_range=(1, 2))),
    ('modelo', LogisticRegression(class_weight='balanced', max_iter=1000))
])

# 5. Treinamento
pipeline.fit(X_treino, y_treino)

# 6. Avaliação
predicoes_teste = pipeline.predict(X_teste)
print("\n--- Relatório de Precisão do Modelo ---")
print(classification_report(y_teste, predicoes_teste))

# 7. Salvar modelo
joblib.dump(pipeline, 'modelo_sentimento.pkl')
print("\nModelo ajustado e salvo como 'modelo_sentimento.pkl'!")