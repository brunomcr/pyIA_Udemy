import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import scipy.stats as stats
import seaborn as sns

# Carrega os dados do CSV localizado em 'Src/Automobile.csv'
csv_data = pd.read_csv('Src/Automobile.csv')

# Remove a primeira coluna (geralmente índice ou identificador único) e a última coluna
csv_data = csv_data.iloc[:, 1:-1]

# Calcula e visualiza a matriz de correlação para entender as relações entre variáveis
corr = csv_data.corr()
sns.heatmap(corr, cmap='coolwarm', annot=True, fmt='.2f')

# Cria subplots para visualizar a relação de cada variável com 'mpg' através de gráficos de dispersão
plt.figure(figsize=(15, 10))
for i, column in enumerate(csv_data.columns):
    if column != 'mpg':
        plt.subplot(3, 3, i)  # Ajusta o layout dos subplots para evitar sobreposição
        plt.scatter(csv_data['mpg'], csv_data[column])
        plt.xlabel('mpg')
        plt.ylabel(column)
plt.tight_layout()
plt.show()

# Prepara a string da fórmula para o modelo de regressão linear OLS
# 'mpg' é a variável dependente e as outras são variáveis independentes
formula = 'mpg ~ ' + ' + '.join(csv_data.columns.drop(['mpg', 'model_year', 'acceleration']))

# Cria e ajusta o modelo de regressão linear OLS com os dados
modelo = sm.ols(formula, data=csv_data)
resultado = modelo.fit()

# Exibe um resumo estatístico do modelo, incluindo coeficientes e métricas de qualidade
print(resultado.summary())

# Análise dos resíduos do modelo para verificar a normalidade
residuos = resultado.resid

# Histograma dos resíduos para visualizar a distribuição
plt.figure(figsize=(10, 6))
plt.hist(residuos, bins=30, edgecolor='k', alpha=0.7)
plt.xlabel('Resíduos')
plt.ylabel('Frequência')
plt.title('Histograma dos Resíduos')
plt.show()

# QQ-plot para avaliar se os resíduos seguem uma distribuição normal
stats.probplot(residuos, dist="norm", plot=plt)
plt.title('QQ-plot dos Resíduos')
plt.show()

# Teste de Shapiro-Wilk para avaliar a normalidade dos resíduos
stat, pval = stats.shapiro(residuos)
print(f'Shapiro-Wilk statistica: {stat:.3f}, p-value: {pval:.3f}')
