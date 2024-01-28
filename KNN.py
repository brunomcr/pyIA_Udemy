from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import numpy as np

# Carrega os dados do CSV localizado em 'Src/Automobile.csv'
csv_data = pd.read_csv('Src/Automobile.csv')

# Remove as linhas com valor NaN na coluna "horsepower"
csv_data = csv_data.dropna(subset=['horsepower'])

# Variáveis Dependentes e Independentes
X = csv_data[['mpg', 'horsepower']].values
y = csv_data[['cylinders']].values

# Ajuste o formato de y usando ravel()
y = np.ravel(y)

# Modelo
knn = KNeighborsClassifier(n_neighbors=3)
modelo = knn.fit(X,y)

# Previsao
y_prev = modelo.predict(X)

accurace = accuracy_score(y, y_prev)
precisao = precision_score(y, y_prev, average='weighted', zero_division=0)
recall = recall_score(y, y_prev, average='weighted')
f1 = f1_score(y, y_prev, average='weighted')
cm = confusion_matrix(y, y_prev)

print(f"""
accurace = {accurace}
precisao = {precisao}
recall = {recall}
f1 = {f1}
CM = {cm}
""")

new_data = np.array([[16.3,185]])

previsao = modelo.predict(new_data)
print(f'previsao de cylinders = {previsao}')

distances, indices = modelo.kneighbors(new_data)
print(f'Distancia = {distances} \n indice = {indices}')

vizinhos_proximos = csv_data.loc[indices[0],['mpg', 'horsepower','cylinders']]
print(f'Vizinhos proximos = {vizinhos_proximos} ')