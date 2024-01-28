# Importação de bibliotecas necessárias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing  import  LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
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
modelo = RandomForestClassifier(random_state=1,
                                n_estimators=100,
                                max_depth=8,
                                max_leaf_nodes=8
                               )

# Treinar o modelo com os dados de treinamento
modelo.fit(X_treinamento, Y_treinamento)

# gera figura
tree_index = 0
tree_to_visualize = modelo.estimators_[tree_index]
plt.figure(figsize=(20,10))
plot_tree(tree_to_visualize,
          filled=True,
          feature_names=csv_data.drop(columns=['work_setting','salary_currency','salary','job_title', 'work_year']).columns,
          class_names=True,
          rounded=True
         )
plt.show()


# Fazer previsões com o modelo nos dados de teste
previsoes = modelo.predict(X_teste)

# Calcular métricas de avaliação: accuracy, precision, recall, f1-score
accurace = accuracy_score(Y_teste, previsoes)
precisao = precision_score(Y_teste, previsoes, average='weighted', zero_division=0)
recall = recall_score(Y_teste, previsoes, average='weighted')
f1 = f1_score(Y_teste, previsoes, average='weighted')

# Imprimir as métricas de avaliação
print(f"""
accurace = {accurace}
precisao = {precisao}
recall = {recall}
f1 = {f1}
""")
