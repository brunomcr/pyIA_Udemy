import numpy as np

class RegressaoLinearSimples:
    def __init__(self):
        self.beta0 = 0
        self.beta1 = 0

    def ajustar(self, X, Y):
        """ Ajusta o modelo de regressão linear aos dados fornecidos. """
        x_media = np.mean(X)
        y_media = np.mean(Y)

        # Cálculo da inclinação (β1)
        numerador = sum((X - x_media) * (Y - y_media))
        denominador = sum((X - x_media) ** 2)
        self.beta1 = numerador / denominador

        # Cálculo da interceptação (β0)
        self.beta0 = y_media - self.beta1 * x_media

    def prever(self, X):
        """ Faz previsões usando o modelo ajustado. """
        return self.beta0 + self.beta1 * X

# Exemplo de uso da classe
# Supondo que você tenha os seguintes dados:
X = np.array([25, 30, 35, 40, 45, 50, 55, 60])  # Variável independente (Idade, por exemplo)
Y = np.array([200, 250, 300, 350, 400, 450, 500, 550])  # Variável dependente (Custo, por exemplo)

# Criação e ajuste do modelo
modelo = RegressaoLinearSimples()
modelo.ajustar(X, Y)

# Fazendo uma previsão
X_novo = np.array([58])  # Nova observação para a qual queremos prever Y
Y_previsto = modelo.prever(X_novo)

print(f"Previsão para X = {X_novo}: Y = {Y_previsto}")
