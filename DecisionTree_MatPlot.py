# Importação de bibliotecas necessárias
import pandas as pd  # Biblioteca para manipulação de dados
from sklearn.model_selection import train_test_split  # Função para dividir o conjunto de dados
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree  # Classe para criar árvore de decisão e função para visualização
from sklearn.preprocessing import LabelEncoder  # Classe para codificar variáveis categóricas
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)  # Métricas de avaliação do modelo
import matplotlib.pyplot as plt

# Importando os dados do CSV
csv_data = pd.read_csv('Src/jobs_in_data.csv')

# Variáveis Dependentes (rótulos) e Independentes (características)
Y = csv_data['work_setting'].values  # Variável dependente (rótulo)
X = csv_data.drop(
    columns=['work_setting', 'salary_currency', 'salary', 'job_title', 'work_year']
).values  # Variáveis independentes (características)

# Aplicar Label Encoding para variáveis categóricas em X
label_encoder = LabelEncoder()
for i in range(X.shape[1]):
    if X[:, i].dtype == 'object':
        X[:, i] = label_encoder.fit_transform(X[:, i])

# Dividir o conjunto de dados em conjuntos de treinamento e teste (70% treinamento, 30% teste)
X_treinamento, X_teste, Y_treinamento, Y_teste = train_test_split(
    X, Y, test_size=0.5, random_state=1
)

# Criar um modelo de árvore de decisão com hiperparâmetros específicos
modelo = DecisionTreeClassifier(random_state=1)

# Treinar o modelo com os dados de treinamento
modelo.fit(X_treinamento, Y_treinamento)

# Gerar a representação da árvore de decisão em formato DOT
dot_data = export_graphviz(
    modelo,
    out_file=None,
    filled=True,
    feature_names=csv_data.drop(
        columns=['work_setting', 'salary_currency', 'salary', 'job_title', 'work_year']
    ).columns,
    class_names=True,
    rounded=True,
)

# Salvar o DOT em um arquivo (opcional)
with open("decision_tree.dot", "w") as f:
    f.write(dot_data)

plt.figure(figsize=(12, 6))
plot_tree(modelo, filled=True, feature_names=csv_data.drop(
    columns=['work_setting', 'salary_currency', 'salary', 'job_title', 'work_year']
).columns, class_names=True, rounded=True)
plt.show()  # Exibir a árvore de decisão na tela

# Fazer previsões com o modelo nos dados de teste
previsoes = modelo.predict(X_teste)

# Calcular métricas de avaliação: accuracy, precision, recall, f1-score
accurace = accuracy_score(Y_teste, previsoes)
precisao = precision_score(Y_teste, previsoes, average='weighted')
recall = recall_score(Y_teste, previsoes, average='weighted')
f1 = f1_score(Y_teste, previsoes, average='weighted')

# Imprimir as métricas de avaliação
print(f"""
accurace = {accurace}
precisao = {precisao}
recall = {recall}
f1 = {f1}
""")
