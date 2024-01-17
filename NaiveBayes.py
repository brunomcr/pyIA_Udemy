# Importando bibliotecas necessárias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from yellowbrick.classifier import ConfusionMatrix

# Importando os dados do CSV
csv_data = pd.read_csv('Src/jobs_in_data.csv')

# Definindo as Variáveis Dependentes (Y) e Independentes (X)
Y = csv_data['work_setting'].values  # Variável Dependente
X = csv_data.drop(columns=['work_setting', 'salary_currency', 'salary', 'job_title']).values  # Variáveis Independentes

# Aplicando Label Encoding para variáveis categóricas em X
label_encoder = LabelEncoder()
for i in range(X.shape[1]):
    if X[:, i].dtype == 'object':
        X[:, i] = label_encoder.fit_transform(X[:, i])

# Dividindo o conjunto de dados em conjuntos de treinamento e teste
# X_treinamento: Conjunto de características de treinamento
# X_teste: Conjunto de características de teste
# Y_treinamento: Conjunto de rótulos de treinamento
# Y_teste: Conjunto de rótulos de teste
# A função train_test_split divide os dados em uma proporção de 50% para treinamento e 50% para teste
# O uso de random_state=1 garante reprodutibilidade nos resultados
X_treinamento, X_teste, Y_treinamento, Y_teste = train_test_split(X, Y, test_size=0.5, random_state=1)

# Criando e treinando um modelo Naive Bayes
modelo = GaussianNB()
modelo.fit(X_treinamento, Y_treinamento)

# Fazendo previsões com o modelo treinado
previsoes = modelo.predict(X_teste)

# Avaliando o desempenho do modelo
accurace = accuracy_score(Y_teste, previsoes)
precisao = precision_score(Y_teste, previsoes, average='weighted')
recall = recall_score(Y_teste, previsoes, average='weighted')
f1 = f1_score(Y_teste, previsoes, average='weighted')

# Imprimindo as métricas de avaliação
print(f"""
accuracy = {accurace}
precision = {precisao}
recall = {recall}
f1 = {f1}
""")

# Gerando um relatório de classificação
report = classification_report(Y_teste, previsoes)
print(report)

# Visualizando a Matriz de Confusão
matrix_confusao = ConfusionMatrix(modelo, classes=['Hybrid', 'In-person', 'Remote'])
matrix_confusao.fit(X_treinamento, Y_treinamento)
matrix_confusao.score(X_teste, Y_teste)
matrix_confusao.show()
