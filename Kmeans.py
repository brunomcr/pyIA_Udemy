from sklearn.datasets import load_iris
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np


def plot_clusters(data, labels, centroids=None, target_names=None, title="Clusters"):
    # Gerando uma paleta de cores aleatórias com base no número de labels únicos (incluindo ruído)
    unique_labels = set(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

    # Mapeando cada label único para uma cor, incluindo uma cor específica para ruído
    label_color = {label: color for label, color in zip(unique_labels, colors)}
    noise_color = [0, 0, 0, 1]  # Preto para ruído

    # Plotando as amostras com seus labels de cluster
    plt.figure(figsize=(10, 8))
    for label in unique_labels:
        # Selecionando cor para o cluster ou ruído
        color = noise_color if label == -1 else label_color[label]
        # Selecionando marcação para o cluster ou ruído
        marker = "x" if label == -1 else "o"
        # Plotando pontos do cluster ou ruído
        cluster_data = data[labels == label]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], c=[color],
                    label=f'Cluster {label}' if label != -1 else 'Noise', alpha=0.5, marker=marker)

    # Adicionando os centros dos clusters ao plot, se fornecidos
    if centroids is not None and target_names is not None:
        plt.scatter(centroids[:, 0], centroids[:, 1], color='yellow', edgecolor='k', marker='.', s=150,
                    label='Centroids', linewidths=1)
        # for i, center in enumerate(centroids):
        # plt.scatter(center[0], center[1], color='yellow', edgecolor='k', marker='.', s=150, label='Centroids', linewidths=1)

    plt.xlabel('Feature 0')  # Modifique para corresponder ao nome real da característica
    plt.ylabel('Feature 1')  # Modifique para corresponder ao nome real da característica
    plt.title(title)
    plt.legend()
    plt.show()

# Carregando o dataset Iris
iris = load_iris()

# Criando um DataFrame com as características
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Adicionando os labels verdadeiros ao DataFrame
iris_df['target'] = iris.target

# Adicionando os nomes dos targets (espécies) como uma nova coluna, mapeando os valores de 'target' para os nomes
iris_df['species'] = iris_df['target'].apply(lambda x: iris.target_names[x])

iris_df.head()  # Exibindo as primeiras linhas do DataFrame para verificação

X = iris.data

# iris.target contém os labels verdadeiros para cada instância do dataset,
# representando a espécie de cada amostra de íris (setosa, versicolor, virginica)
print(iris.target)

# Aplicando o algoritmo KMeans com n_init explicitamente definido
# kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
kmeans = KMeans(n_clusters=3, n_init=10)
kmeans.fit(X)

# Centroides dos clusters
centroids = kmeans.cluster_centers_
print(centroids)

# Labels dos clusters para cada instância no dataset
labels = kmeans.labels_
print(labels)

# Calculando e exibindo a matriz de confusão para avaliar o agrupamento em comparação com os labels verdadeiros
resultados = confusion_matrix(iris.target, kmeans.labels_)
print(resultados)

# Plotando os clusters com os nomes dos targets nos centros dos clusters
plot_clusters(X, kmeans.labels_, centroids, target_names=iris.target_names, title='Kmeans')

# Aplicando DBSCAN, um algoritmo baseado em densidade para agrupamento que pode formar clusters de formas complexas
dbscan = DBSCAN(eps=0.5, min_samples=3)
dbscan_labels = dbscan.fit_predict(X)
print(dbscan_labels)

# Matriz de confusão para os resultados do DBSCAN
resultados = confusion_matrix(iris.target, dbscan_labels)
print(resultados)

# Plotando os clusters com os nomes dos targets nos centros dos clusters
plot_clusters(X, dbscan_labels, target_names=iris.target_names, title='DBScan')

# Utilizando Agglomerative Clustering, um método hierárquico que agrupa os dados baseado na distância entre eles
agglo = AgglomerativeClustering(n_clusters=3)
agglo_labels = agglo.fit_predict(X)
print(agglo_labels)

# Matriz de confusão para os resultados do Agglomerative Clustering
resultados = confusion_matrix(iris.target, agglo_labels)
print(resultados)

# Plotando os clusters com os nomes dos targets nos centros dos clusters
plot_clusters(X, agglo_labels, target_names=iris.target_names, title='Agglomerative')

# Realizando o agrupamento hierárquico
linked = linkage(X, 'ward')

# Plotando o dendrograma
plt.figure(figsize=(18, 7))
dendrogram(linked,
           orientation='top',
           labels=np.array(iris.target),
           distance_sort='descending',
           show_leaf_counts=True,
          )
plt.axhline(y=7, c='grey', lw=1, linestyle='dashed')
plt.xlabel('Index of Iris data points')
plt.ylabel('Ward distance')
plt.title('Dendrogram of Iris Dataset')
plt.show()
