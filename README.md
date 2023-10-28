# pyIA_Udemy
Formação Inteligência Artificial e Machine Learning 2023

# Machine Learning

<hr>

# Algoritimos / Métodos

* ### 1.   Classificação (Supervisionado)
  * A classificação é uma técnica de Machine Learning usada para descrever ou prever um atributo categórico.
  Ela é frequentemente usada quando você deseja categorizar dados em grupos ou classes,
  como determinar se um e-mail é spam ou não.

* ### 2.   Regressão (Supervisionado)
  * A regressão é usada para prever um atributo numérico. Ao contrário da classificação, que lida com categorias,
  a regressão lida com valores contínuos. Por exemplo, prever o preço de uma casa com base em características
  como tamanho, localização e número de quartos é um problema de regressão.

* ### 3.   Agrupamentos (Não Supervisionado)
  * Agrupamento é uma técnica que visa agrupar dados com base em características ou semelhanças matemáticas.
    É útil quando você deseja encontrar padrões em seus dados e agrupar pontos de dados semelhantes juntos,
    mesmo que você não saiba de antemão quais grupos existem.

* ### 4.   Regras de Associação (Supervisionado ou Não Supervisionado)
  * As regras de associação são usadas para descobrir relações e associações entre diferentes itens
  em conjuntos de dados. Isso é frequentemente usado em sistemas de recomendação, como sugerir produtos 
  a clientes com base em seus históricos de compras.

* ### 5.   Detecção de Anomalias
* ### 6.   Aprendizado por reforço
* ### 7.   Processamento de Linguagem Natural (NLP)
* ### 8.   Redes Neurais
* ### 9.   Redução de dimensionalidade / Seleção de recursos
* ### 10.  Aprendizado semisupervisionado

<hr>

# Desempenho

### Matriz de Confusão (Classificação)
    Fornece uma visão detalhada do desempenho de um modelo, permitindo a análise de como ele está
    acertando ou errando na classificação de diferentes classes.


## Métricas

* ### Acurácia (Accuracy):
  * Mede a proporção de previsões corretas feitas pelo modelo.
  É fácil de entender, mas pode ser enganosa em problemas com classes desequilibradas.

  
    Acurácia = (VP + VN) / (VP + FP + VN + FN)

* ### Precisão (Precision):
  * Mede a proporção de instâncias classificadas como positivas que realmente são positivas.
  É fácil de entender e é útil quando o foco está em evitar falsos positivos.

  
    Precisão = VP / (VP + FP)

* ### Revocação (Recall ou Sensibilidade): 
  * Mede a proporção de instâncias positivas que foram corretamente identificadas pelo modelo.
  É fácil de entender e é útil quando o foco está em evitar falsos negativos.

  
    Revocação = VP / (VP + FN)

* ### Especificidade (Specificity): 
  * Mede a capacidade do modelo de evitar falsos positivos na classe negativa. 
  Também é relativamente fácil de entender.

    
    Especificidade = VN / (VN + FP)

* ### Taxa de Falsos Positivos (False Positive Rate): 
  * Mede a proporção de instâncias negativas que foram incorretamente classificadas como positivas.
  É um pouco mais complexa do que as métricas anteriores, mas ainda é compreensível.

  
    Taxa de Falsos Positivos = FP / (FP + VN)

* ### F1-Score: 
  * Uma métrica que combina precisão e revocação em uma única pontuação. 
  É um pouco mais complexa, pois considera ambas as métricas.

  
    F1-Score = 2 * (Precisão * Revocação) / (Precisão + Revocação)

* ### Área sob a Curva ROC (AUC-ROC): 
  * Esta métrica avalia a capacidade do modelo de distinguir entre classes positivas e negativas.
  É mais complexa, pois envolve o uso da curva ROC.

* ### Área sob a Curva PR (AUC-PR): 
  * Mede a área sob a curva da curva de precisão-recall (PR).
  É mais complexa do que a AUC-ROC e é útil em problemas com classes desequilibradas.


## Métricas de Erro

* ### Erro de Classificação (Classification Error):
  * Mede a taxa de classificações incorretas feitas pelo modelo.

    
    Fórmula: (FP + FN) / (TP + FP + TN + FN)

* ### Taxa de Falsos Positivos (False Positive Rate):
  * Mede a proporção de instâncias negativas que foram incorretamente
  classificadas como positivas.


      Fórmula: FP / (FP + TN)

* ### Taxa de Falsos Negativos (False Negative Rate):
  * Mede a proporção de instâncias positivas que foram incorretamente
  classificadas como negativas.

    
    Fórmula: FN / (TP + FN)

* ### ME (Mean Error):
  * Representa o erro médio, calculado como a média das diferenças entre
  as previsões do modelo e os valores reais.

    
    Fórmula: Σ (Valor Real - Previsão) / N

* ### Erro Absoluto Médio (Mean Absolute Error - MAE):
  * Usado em problemas de regressão, representa a média do valor absoluto
  das diferenças entre as previsões do modelo e os valores reais.

    
    Fórmula: Σ |Valor Real - Previsão| / N

* ### Erro Quadrático Médio (Mean Squared Error - MSE):
  * Outra métrica de regressão que mede a média dos quadrados das
  diferenças entre as previsões do modelo e os valores reais.

    
    Fórmula: Σ (Valor Real - Previsão)² / N

* ### Raiz do Erro Quadrático Médio (Root Mean Squared Error - RMSE):
  * O desvio padrão da amostra da diferenca entre o previsto e o teste.

    
    Fórmula: √(Σ (Valor Real - Previsão)² / N)

* ### Erro Médio Percentual Absoluto (Mean Absolute Percentage Error - MAPE):
  * Expressa o erro médio como uma porcentagem da média das observações reais, 
  utilizado em problemas de regressão.

    
    Fórmula: (Σ |(Valor Real - Previsão) / Valor Real|) / N

* ### Log Loss (Entropia Cruzada):
  * Amplamente usado em problemas de classificação, mede a diferença
  entre as probabilidades previstas pelo modelo e as classes reais.

    
    Fórmula: -Σ [y * log(p) + (1 - y) * log(1 - p)] / N

* ### Erro de Média Quadrática Logarítmica (Mean Squared Logarithmic Error - MSLE):
  * Usado em problemas de regressão, leva em consideração o logaritmo natural
  das previsões e valores reais.

    
    Fórmula: Σ (log(Valor Real + 1) - log(Previsão + 1))² / N

<hr>

#  Codificação Categórica (Categorical Encoding)
    A Codificação Categórica é um processo crucial na preparação de dados para análise ou modelagem
    em aprendizado de máquina. Ela lida com a representação de variáveis categóricas (ou qualitativas)
    em um formato numérico, uma vez que muitos algoritmos de aprendizado de máquina requerem entradas numéricas.
    Variáveis categóricas são aquelas que representam categorias ou rótulos, como tipos de produtos, cores, países, etc.

* ### Codificação One-Hot (One-Hot Encoding):
  * A Codificação One-Hot é um método popular que converte variáveis categóricas em vetores binários de 0s e 1s. 
  Cada categoria recebe uma coluna binária, e apenas uma coluna contém o valor "1" correspondente à categoria, 
  enquanto as outras são "0". Isso permite que o modelo trate cada categoria de forma independente.

* ### Codificação de Rótulos (Label Encoding):
  * A Codificação de Rótulos atribui um valor numérico a cada categoria, transformando-as em valores inteiros. 
  Essa abordagem é adequada para variáveis categóricas ordinais, onde a ordem entre as categorias tem importância.

* ### Codificação de Contagem (Count Encoding):
  * A Codificação de Contagem substitui cada categoria pelo número de vezes que ela aparece no conjunto de dados. 
  Essa técnica pode ser útil quando a frequência de uma categoria é informativa.

* ### Codificação de Frequência (Frequency Encoding):
  * A Codificação de Frequência é semelhante à Codificação de Contagem, mas divide o número de vezes que uma categoria
  aparece pelo total de observações. Isso transforma as categorias em proporções, o que pode ser útil em alguns casos.

* ### Codificação Ordinal (Ordinal Encoding):
  * A Codificação Ordinal atribui números inteiros às categorias com base em uma ordem predefinida.
  É apropriada para variáveis categóricas ordinais, onde a ordem entre as categorias é significativa.

* ### Codificação Target (Target Encoding):
  * A Codificação Target envolve a substituição das categorias pela média da variável de destino para aquela categoria
específica. Isso pode ser útil quando a relação entre a variável categórica e a variável de destino é importante.

* ### Codificação Binary (Binary Encoding):
  * A Codificação Binary converte cada valor inteiro em seu equivalente binário e cria várias colunas binárias. 
  Cada coluna representa um dígito binário, permitindo que as categorias sejam representadas em um espaço numérico.

* ### Codificação Helmert:
  * A Codificação Helmert calcula a média das variáveis dependentes para as categorias subsequentes em relação à 
  categoria atual. É útil quando a ordem das categorias é importante.

* ### Codificação Backward Difference:
  * A Codificação Backward Difference compara cada categoria com a categoria anterior. É útil para destacar a 
  diferença entre as categorias em relação à anterior.

<hr>

#  Dimensionamento de Características 
    No machine learning, o dimensionamento de características envolve normalizar ou padronizar as características
    do conjunto de dados. Isso é feito para garantir que todas as características tenham a mesma escala ou magnitude,
    evitando assim que algumas características dominem outras durante o treinamento do modelo. As técnicas comuns
    incluem a normalização min-max e a padronização (z-score).

* ### Normalização Min-Max:
  * A normalização Min-Max é uma técnica que ajusta os valores de características para que fiquem dentro
  de um intervalo específico, geralmente entre 0 e 1.
  Para realizar a normalização Min-Max, você subtrai o valor mínimo da característica de cada valor e,
  em seguida, divide pelo intervalo (a diferença entre o valor máximo e o valor mínimo).
    
  * Essa técnica é útil quando você deseja que todas as características tenham a mesma escala e variação,
  o que é importante para algoritmos sensíveis à escala, como redes neurais e algoritmos de gradiente descendente.
  
  
    F[ormula: X_norm = (X - X_min) / (X_max - X_min)


* ### Padronização (Z-Score):
  * A padronização, também conhecida como Z-Score, transforma os valores das características de modo que tenham
  uma média de 0 e um desvio padrão de 1.
  * A padronização é útil quando as características do conjunto de dados não seguem uma distribuição normal
  e têm escalas diferentes. Isso ajuda a tornar as características comparáveis e facilita a interpretação
  dos coeficientes dos modelos.
  * A padronização é mais robusta contra valores discrepantes do que a normalização Min-Max.

    
    Fórmula: X_standardized = (X - média) / desvio_padrão


<hr>

# Agrupamentos (Cluster)
    É uma técnica de aprendizado de máquina não supervisionado. Ela envolve a tarefa de dividir um conjunto de dados
    em grupos ou "clusters" com base em suas similaridades. O objetivo principal é criar grupos onde os pontos de dados
    dentro de um mesmo cluster sejam mais semelhantes entre si do que com os pontos de outros clusters.

## Algoritmos de Clustering

* ### Particional:

  * #### K-Means: 
    * Um dos algoritmos de clustering mais populares, onde os dados são divididos em um número
    específico de clusters (K) com base na minimização da variância intra-cluster.

  * #### K-Medoids: 
    * Similar ao K-Means, mas usa os medóides (pontos representativos) em vez das médias dos clusters
    para minimizar as distâncias intra-cluster.
  
  * #### Fuzzy C-Means:
    * Permite que um ponto de dados pertença parcialmente a múltiplos clusters, associando probabilidades
    de pertencimento a cada cluster.

* ### Hierárquico:

  * #### Aglomerativo: 
    * Começa com cada ponto de dados como um cluster individual e mescla gradualmente clusters vizinhos até que
    todos os dados estejam em um único cluster ou atenda a algum critério de parada.

  * #### Divisivo: 
    * Começa com todos os dados em um único cluster e divide-o em subclusters menores à medida que avança na hierarquia.

* ### Baseado em Densidade:

  * #### DBSCAN (Density-Based Spatial Clustering of Applications with Noise): 
    * Identifica clusters com densidades variáveis e é capaz de detectar ruído ou outliers.
  
  * #### OPTICS (Ordering Points to Identify the Clustering Structure): 
    * Similar ao DBSCAN, mas gera uma representação hierárquica de clusters.

* ### Modelo de Mistura:

  * #### Gaussian Mixture Models (GMM): 
    * Modela os dados como uma mistura de várias distribuições gaussianas, o que pode ser útil quando os dados
    não formam clusters distintos.

  * ### Baseado em Grade:

  * #### Self-Organizing Maps (SOM): 
    * Usa uma grade para organizar dados multidimensionais em clusters, frequentemente usado para visualização
    e redução de dimensionalidade.

### Outros:
  
  * #### Clustering Espectral: 
    * Usa técnicas de álgebra linear para agrupar dados com base na estrutura dos eigenvetores da matriz de afinidade.
    
  * #### Agglomerative Nests: 
    * Uma extensão do agglomerative clustering que permite identificar subclusters dentro dos principais clusters.

<hr>
<b>Obs:</b><br><br>
<b>Outliers:</b><br>
    Em estatística e análise de dados, são pontos de dados que se afastam significativamente do padrão ou do 
    comportamento esperado de um conjunto de dados. Eles são valores atípicos que são notavelmente diferentes dos 
    outros pontos de dados no conjunto. Outliers podem ser causados por erros de medição, anomalias reais nos dados
    ou representar eventos raros.
<hr>

## Regra de Associação
    É um conceito importante em mineração de dados e análise de padrões. Elas são usadas para descobrir relações 
    significativas entre diferentes itens em um conjunto de dados. Em termos simples, as regras de associação 
    ajudam a identificar quais itens costumam aparecer juntos. Um exemplo clássico é a análise de carrinhos de
    compras de clientes em um supermercado para entender quais produtos são frequentemente comprados em conjunto.

Uma regra de associação é representada na forma "Antecedente -> Consequente," com uma seta indicando a direção
da relação. A interpretação da regra é que se o antecedente estiver presente, é provável que o consequente também 
esteja presente.

* ### Antecedente (ou antecedentes): 
  * Esta é a parte esquerda da regra e representa os itens que estão sendo considerados como o "gatilho" ou "condição" 
  para a regra. Em um contexto de compras, o antecedente pode ser um conjunto de produtos, por exemplo, "pão e leite."

* ### Consequente (ou consequentes):
  * Esta é a parte direita da regra e representa os itens que são considerados como "resultados" ou "consequências" 
  do antecedente. No exemplo, o consequente pode ser "ovos."

* ### Suporte: 
  * O suporte mede a frequência com que o antecedente e o consequente aparecem juntos no conjunto de dados. 
  Ele indica a proporção de transações ou registros que contêm a combinação dos itens do antecedente e do consequente.

* ### Confiança: 
  * A confiança mede a probabilidade de que o consequente seja verdadeiro dado que o antecedente é verdadeiro. 
  Em outras palavras, indica a força da relação entre o antecedente e o consequente.

* ### Lift: 
  * Compara a confiança da regra com a probabilidade de ocorrer o consequente independentemente do antecedente. 

    * #### Lift Positivo:
      * O lift é maior que 1.
      Indica que a regra de associação tem uma influência positiva, ou seja, a presença do antecedente aumenta a
      probabilidade de o consequente ocorrer em comparação com o que ocorreria aleatoriamente.

    * #### Lift Igual a 1:
      * O lift é igual a 1.
      Nesse caso, a regra de associação não tem impacto significativo.A presença do antecedente não aumenta 
      nem diminui a probabilidade do consequente em comparação com o que ocorreria aleatoriamente.

    * #### Lift Negativo:
      * O lift é menor que 1.
      Indica que a regra de associação tem uma influência negativa, ou seja, a presença do antecedente diminui 
      a probabilidade do consequente em comparação com o que ocorreria aleatoriamente.