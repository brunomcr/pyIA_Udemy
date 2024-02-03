# pyIA_Udemy
Formação Inteligência Artificial e Machine Learning 2023

# Machine Learning

<hr>

# ### Fundamentos

<hr>


# Algoritimos / Métodos

* ### 1. Classificação (Supervisionado)
  * A classificação é uma técnica de Machine Learning usada para descrever ou prever um atributo categórico.
  Ela é frequentemente usada quando você deseja categorizar dados em grupos ou classes,
  como determinar se um e-mail é spam ou não.

* ### 2. Regressão (Supervisionado)
  * A regressão é usada para prever um atributo numérico. Ao contrário da classificação, que lida com categorias,
  a regressão lida com valores contínuos. Por exemplo, prever o preço de uma casa com base em características
  como tamanho, localização e número de quartos é um problema de regressão.

* ### 3. Agrupamentos (Não Supervisionado)
  * Agrupamento é uma técnica que visa agrupar dados com base em características ou semelhanças matemáticas.
    É útil quando você deseja encontrar padrões em seus dados e agrupar pontos de dados semelhantes juntos,
    mesmo que você não saiba de antemão quais grupos existem.

* ### 4. Regras de Associação (Supervisionado ou Não Supervisionado)
  * As regras de associação são usadas para descobrir relações e associações entre diferentes itens
  em conjuntos de dados. Isso é frequentemente usado em sistemas de recomendação, como sugerir produtos 
  a clientes com base em seus históricos de compras.

* ### 5. Detecção de Anomalias
* ### 6. Aprendizado por reforço
* ### 7. Processamento de Linguagem Natural (NLP)
* ### 8. Redes Neurais
* ### 9. Redução de dimensionalidade / Seleção de recursos
* ### 10. Aprendizado semisupervisionado

<hr>

## Correlação 
A correlação é um conceito estatístico que mede o grau de associação ou relação entre duas variáveis. É diferente da regressão, embora ambos sejam frequentemente usados para examinar relações entre variáveis, pois a correlação não implica causalidade. Existem diferentes tipos de coeficientes de correlação, sendo os mais comuns o coeficiente de correlação de Pearson e o coeficiente de correlação de Spearman. Vou explicar brevemente ambos:

* ### 1. Coeficiente de Correlação de Pearson (r):

  * **Descrição**: O coeficiente de correlação de Pearson, denotado como "r," mede a força e a direção da relação linear entre duas variáveis contínuas. Ele varia de -1 a 1. Um valor de 1 indica uma correlação positiva perfeita, -1 indica uma correlação negativa perfeita, e 0 indica ausência de correlação.
  * **Aplicações**: É comumente usado para avaliar a relação entre variáveis quantitativas, como idade e pressão arterial, ou renda e despesas.

* ### 2. Coeficiente de Correlação de Spearman (ρ):

  * **Descrição**: O coeficiente de correlação de Spearman, denotado como "ρ" (rho), é usado para medir a relação estatística entre duas variáveis. No entanto, ele não se limita a relações lineares. Ele se baseia na classificação das observações em vez dos valores brutos.
  * **Aplicações**: É útil quando as relações não são necessariamente lineares, ou quando os dados não seguem uma distribuição normal.
  
A correlação, seja de Pearson ou Spearman, fornece informações sobre como duas variáveis estão relacionadas. No entanto, é importante destacar que a correlação não implica causalidade. Dois eventos podem estar correlacionados sem que um cause o outro. Portanto, é necessário ter cuidado ao interpretar os resultados da correlação e considerar outros fatores e análises para estabelecer relações de causa e efeito.

## Coeficiente de Determinação (R²):
O coeficiente de determinação, frequentemente denotado como R² (R ao quadrado), é uma métrica estatística que não é um algoritmo por si só, mas sim uma medida usada para avaliar o desempenho de modelos de regressão. Ele é uma métrica importante para entender o quão bem um modelo de regressão se ajusta aos dados. Portanto, assim como a correlação, o coeficiente de determinação não faz parte diretamente da lista de algoritmos e métodos de aprendizado de máquina.

  * Descrição: O coeficiente de determinação é uma medida que varia de 0 a 1 (ou de 0% a 100%) e indica a proporção da variabilidade na variável dependente que é explicada pelas variáveis independentes incluídas no modelo de regressão. Quanto mais próximo de 1, melhor o modelo se ajusta aos dados, indicando que as variáveis independentes explicam uma grande parte da variabilidade na variável dependente.
  * Interpretação: Um valor de R² igual a 0 indica que o modelo não explica nenhuma variabilidade na variável dependente, enquanto um valor de 1 indica que o modelo explica toda a variabilidade. Valores intermediários representam a quantidade de variabilidade explicada pelo modelo.
  * Aplicações: É comumente usado na análise de regressão para avaliar a qualidade do ajuste do modelo aos dados.

Embora o coeficiente de determinação não seja um algoritmo, ele desempenha um papel fundamental na avaliação de modelos de regressão, ajudando a determinar quão bem um modelo está se ajustando aos dados observados. É uma métrica importante para compreender a qualidade de um modelo de regressão, mas não é uma técnica ou algoritmo de aprendizado de máquina por si só.

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

<hr>

## Algoritmos de Mineração de Dados 

* ### Apriori:
  * **Descrição**: O Apriori é um algoritmo clássico de mineração de regras de associação. 
  Ele é usado para encontrar relações frequentes entre itens em um conjunto de dados, como transações de compras.
  O objetivo é identificar itens que são frequentemente comprados juntos.
  * **Funcionamento**: O Apriori opera em duas etapas. Primeiro, ele gera conjuntos de itens frequentes
  (itens que aparecem juntos com uma frequência maior que um limite definido). Em seguida, utiliza 
  esses conjuntos para extrair regras de associação, que descrevem a probabilidade de um item ser
  comprado dado que outro item foi comprado.
  * **Aplicações comuns**: Recomendação de produtos, análise de mercado, otimização de layout de lojas, entre outros.

* ### FP-Growth (Frequent Pattern Growth):
  * **Descrição**: O FP-Growth é outro algoritmo de mineração de regras de associação. 
  Ele foi projetado para superar algumas limitações do Apriori, especialmente em relação ao desempenho
  em conjuntos de dados grandes.
  * **Funcionamento**: O FP-Growth utiliza uma estrutura de árvore de prefixo (árvore de frequência) para representar
  os padrões frequentes no conjunto de dados. Isso permite uma exploração mais eficiente dos padrões frequentes
  sem a necessidade de geração explícita de todos os subconjuntos de itens.
  * **Aplicações comuns**: Semelhantes às do Apriori, como recomendação de produtos, análise de mercado e 
  mineração de dados em geral.

<hr>

# ### ESTUDO

<hr>

# Correlação e Regressão Linear

### Relação entre Correlação e Regressão
- Enquanto a correlação mede apenas a força e a direção da relação linear, a regressão linear fornece a equação 
  para estimar a variável dependente com base nos valores da variável independente.
- Uma correlação forte (próxima de 1 ou -1) sugere que um modelo de regressão linear pode ser adequado para prever
  uma variável a partir da outra.

## Correlação

### Definição
A correlação mede a força e a direção da relação linear entre duas variáveis quantitativas. É comumente expressa 
através do coeficiente de correlação de Pearson.

### Coeficiente de Correlação de Pearson (r)
- Varia entre -1 e 1.
- **r = 1**: Correlação positiva perfeita.
- **r = -1**: Correlação negativa perfeita.
- **r = 0**: Sem correlação linear.

### Interpretação
- Valores próximos de 1 ou -1 indicam uma forte relação linear.
- Valores próximos de 0 indicam uma fraca ou nenhuma relação linear.

### Fórmula do Coeficiente de Correlação de Pearson
O Coeficiente de Correlação de Pearson, denotado como (r), é calculado pela fórmula:

* #### Usando Covariância e Variância
  `r = cov(X,Y) / √(var(X) * var(Y))`
<br><br>
  - `cov(X,Y)` é a covariância entre as variáveis X e Y.
  - `var(X)` e `var(Y)` são as variâncias de X e Y, respectivamente.

* #### Usando a Fórmula de Soma de Produtos
  `r = Σ((xi - x̄)(yi - ȳ)) / √(Σ(xi - x̄)²Σ(yi - ȳ)²)`
<br><br>
  - `xi` e `yi` são os valores individuais das variáveis X e Y, respectivamente.
  - `x̄` e `ȳ` são as médias dos valores de X e Y.
  - `Σ` representa a soma sobre todos os valores.
  - A expressão `√` denota a raiz quadrada.

Esta fórmula calcula a covariância entre as variáveis X e Y e a divide pelo produto de seus desvios padrão, fornecendo 
uma medida normalizada da relação linear entre as duas variáveis.


## Regressão Linear

### Definição
A regressão linear é usada para modelar a relação entre uma variável dependente e uma ou mais variáveis independentes.
O modelo mais simples é a regressão linear simples, que utiliza uma única variável independente.

### Equação da Regressão Linear Simples
`Y = β0 + β1X + ε`
- `Y`: Variável dependente.
- `X`: Variável independente.
- `β0`: Intercepto.
- `β1`: Coeficiente da variável independente.
- `ε`: Termo de erro.

### Interpretação
- O coeficiente `β1` indica a mudança na variável dependente para uma unidade de mudança na variável independente.
- O intercepto `β0` é o valor esperado de `Y` quando `X` é 0.

## Variáveis Simples e Múltiplas em Modelos de Regressão

### Variáveis Simples (Univariadas)

### Definição
- **Variáveis Simples**: Em estatística e modelagem, uma variável simples refere-se a uma única variável independente 
  usada em análises ou modelos.

### Uso em Regressão
- **Regressão Linear Simples**: Um modelo de regressão que usa apenas uma variável independente para prever uma 
  variável dependente.
- **Exemplo de Equação**:
  - `Y = β0 + β1X + ε`
  - Onde `Y` é a variável dependente, `X` é a variável independente, `β0` é o intercepto, `β1` é o coeficiente da 
    variável independente, e `ε` é o termo de erro.

### Variáveis Múltiplas (Multivariadas)

### Definição
- **Variáveis Múltiplas**: Referem-se ao uso de duas ou mais variáveis independentes em análises ou modelos.

### Uso em Regressão
- **Regressão Linear Múltipla**: Um modelo que utiliza várias variáveis independentes para prever uma 
  variável dependente.
- **Exemplo de Equação**:
  - `Y = β0 + β1X1 + β2X2 + ... + βnXn + ε`
  - Onde `Y` é a variável dependente, `X1, X2, ..., Xn` são as variáveis independentes, `β0` é o intercepto, 
    `β1, β2, ..., βn` são os coeficientes para cada variável independente, e `ε` é o termo de erro.

### Considerações Importantes

- **Complexidade**: Modelos com múltiplas variáveis são mais complexos e podem requerer cuidados adicionais na 
  interpretação e na verificação das suposições do modelo.
- **Risco de Overfitting**: Modelos com muitas variáveis podem se ajustar demais aos dados de treinamento, reduzindo
  a capacidade de generalização para novos dados.
- **Análise de Causalidade**: Em ambos os tipos de modelos, a correlação não implica causalidade. É importante realizar
  análises adicionais para inferências causais.



## Coeficiente de Determinação (R²)

O coeficiente de determinação, representado por (R²) (R-quadrado), é uma medida estatística usada na análise
de regressão para avaliar o ajuste de um modelo aos dados observados.

### Características do Coeficiente de Determinação

### Faixa de Valores
- Varia de 0 a 1.
  - **0**: O modelo não explica nenhuma variação na variável dependente.
  - **1**: O modelo explica completamente a variação na variável dependente.

### Interpretação
- Um valor mais alto de (R^2) indica um melhor ajuste do modelo aos dados.
- Por exemplo, um (R²) de 0.80 sugere que 80% da variação na variável dependente é explicada pelas variáveis
  independentes no modelo.

### Limitações
- Não indica causalidade.
- Pode ser enganoso em modelos com muitas variáveis.
- Um alto (R²) não necessariamente indica um bom ajuste preditivo.

### Exemplo de Cálculo
O (R²) é calculado como:

`R² = SSR/SST = 1 - SSE/SST`

Onde:
- `SSR`: Soma dos quadrados da regressão (explicada).
- `SST`: Soma total dos quadrados (total).
- `SSE`: Soma dos quadrados dos erros (não explicada).

### Uso em Análise de Regressão
O (R²) é usado para avaliar a adequação de um modelo de regressão linear, indicando quão bem as variáveis independentes
explicam a variação na variável dependente. Contudo, é importante considerar outras métricas e análises para uma
avaliação completa do modelo.


## Residuais em Análise de Regressão

### Definição de Residuais
Residuais são as diferenças entre os valores reais observados de uma variável dependente e os valores previstos por
um modelo de regressão.

### Importância dos Residuais
- **Avaliação do Modelo**: Residuais são usados para avaliar se um modelo de regressão se ajusta adequadamente aos dados.
- **Padrões nos Residuais**: A análise dos padrões nos residuais pode indicar problemas no modelo, como a
    não-linearidade ou heterocedasticidade.

### Cálculo dos Residuais
O residual de uma observação é calculado como:

`Residual = Valor Observado - Valor Previsto`

### Uso dos Residuais
- **Diagnóstico do Modelo**: A análise dos residuais ajuda a identificar se as suposições da regressão linear 
    foram violadas.
- **Identificação de Outliers**: Residuais grandes podem indicar outliers ou pontos influentes que afetam a qualidade
    do ajuste do modelo.
- **Melhoria do Modelo**: A análise dos residuais pode fornecer insights para aperfeiçoar o modelo de regressão.

### Representação Gráfica
- Gráficos de residuais, como gráficos de dispersão dos residuais versus valores ajustados, são ferramentas visuais
  úteis para avaliar a adequação do modelo de regressão.


## Inclinação em Modelos de Regressão Linear

### Definição de Inclinação
A inclinação, no contexto de um modelo de regressão linear, refere-se ao coeficiente que quantifica a relação entre a 
variável independente e a variável dependente. Ela indica a mudança esperada na variável dependente para cada unidade
de mudança na variável independente.

### Fórmula da Inclinação 
A inclinação em um modelo de regressão linear simples é calculada pela fórmula:

* #### Fórmula Baseada na Soma dos Produtos
  `β1 = Σ((xi - x̄) * (yi - ȳ)) / Σ(xi - x̄)²`

Onde:
- `β1` é a inclinação.
- `xi` e `yi` são os valores individuais das variáveis independentes (X) e dependentes (Y), respectivamente.
- `x̄` e `ȳ` são as médias dos valores de X e Y.
- `Σ` representa a soma sobre todos os valores.

### Interpretação da Inclinação
- Um valor positivo de `β1` indica uma relação positiva entre X e Y, onde um aumento em X está associado a um 
  aumento em Y.
- Um valor negativo de `β1` indica uma relação negativa, onde um aumento em X está associado a uma diminuição em Y.
- O valor de `β1

### Fórmula Alternativa para a Inclinação
A inclinação (m) em uma regressão linear simples também pode ser calculada usando a correlação entre as variáveis e o
desvio padrão de cada variável:

* #### Fórmula Baseada na Correlação e Desvios Padrão
  `m = r × (Sy / Sx)`

Onde:
- `m` é a inclinação.
- `r` é o coeficiente de correlação de Pearson entre as variáveis X e Y.
- `Sy` é o desvio padrão da variável dependente (Y).
- `Sx` é o desvio padrão da variável independente (X).

### Interpretação
- Esta fórmula mostra que a inclinação é diretamente proporcional à correlação entre X e Y e ao quão dispersos estão os
  valores de Y em relação aos de X.
- A inclinação indica quanto Y muda, em média, para cada unidade de mudança em X, ajustada pela relação entre os seus
  desvios padrão.

### Aplicação
- Esta abordagem é particularmente útil quando se tem o coeficiente de correlação e os desvios padrão das variáveis
  prontamente disponíveis, facilitando o cálculo da inclinação sem a necessidade dos cálculos detalhados da 
  covariância e variância.


### Importância no Modelo de Regressão
A inclinação é um componente essencial na análise de regressão, pois fornece insights diretos sobre a natureza da
relação entre as variáveis estudadas. Ela é crucial para compreender como uma variável independente afeta a variável
dependente, sendo fundamental em diversas aplicações, desde previsões econômicas até análises científicas.


## Interceptação em Modelos de Regressão Linear

### Definição de Interceptação
A interceptação, frequentemente denotada como `β0` em modelos de regressão linear, é o ponto onde a linha de 
regressão intercepta o eixo Y. Isto é, é o valor estimado da variável dependente (Y) quando a variável independente (X)
é igual a zero.

### Fórmula da Interceptação
Em um modelo de regressão linear simples, a interceptação é calculada pela fórmula:

`β0 = ȳ - β1 * x̄`

Onde:
- `β0` é a interceptação.
- `ȳ` é a média da variável dependente (Y).
- `β1` é a inclinação da linha de regressão.
- `x̄` é a média da variável independente (X).

### Interpretação da Interceptação
- A interceptação representa o valor esperado de Y quando X é zero.
- Em muitos contextos, a interceptação pode ter um significado prático, especialmente se um valor de X igual a zero é
  possível ou relevante para os dados observados.
- Em outros casos, a interceptação pode não ter um significado prático (por exemplo, se X = 0 não é um valor possível
  nos dados), mas ainda é uma parte vital da equação da linha de regressão.

### Importância no Modelo de Regressão
A interceptação é crucial para posicionar corretamente a linha de regressão no gráfico e é essencial para fazer
previsões precisas dentro do contexto do modelo. Ela é um componente fundamental da equação da linha de regressão linear
e desempenha um papel importante na interpretação do modelo.


## Previsão em Modelos de Regressão Linear

### Definição de Previsão
A previsão em modelos de regressão linear é o processo de usar a equação do modelo para estimar o valor da variável 
dependente (Y) com base em valores conhecidos ou novos das variáveis independentes (X).

### Fórmula de Previsão
A fórmula para realizar previsões em um modelo de regressão linear simples é:

`Ŷ = β0 + β1X`

Onde:
- `Ŷ` é o valor previsto da variável dependente.
- `β0` é a interceptação da linha de regressão.
- `β1` é a inclinação da linha de regressão.
- `X` é o valor da variável independente para a qual estamos fazendo a previsão.

### Processo de Previsão
1. **Determinar os Coeficientes**: Primeiro, identifique ou calcule os coeficientes `β0` (interceptação) e 
   `β1` (inclinação) usando os dados de treinamento.
2. **Aplicar a Fórmula**: Substitua os coeficientes e o valor de X na fórmula para obter a previsão.
3. **Interpretação da Previsão**: O resultado `Ŷ` é a estimativa do valor de Y dado o valor específico de X.

### Aplicações da Previsão
- Previsões são amplamente usadas em diversas áreas, como economia, ciência, engenharia e ciência de dados.
- Elas são úteis para tomar decisões baseadas em dados, planejar estratégias futuras e entender tendências e padrões.

### Importância da Previsão
- A capacidade de fazer previsões precisas é um aspecto fundamental da análise de regressão linear.
- Previsões baseadas em modelos de regressão ajudam a quantificar o impacto esperado de mudanças em variáveis 
  independentes sobre a variável dependente.

### Limitações
- As previsões são apenas tão precisas quanto o modelo utilizado. Se o modelo não se ajustar bem aos dados ou 
  se basear em suposições incorretas, as previsões podem ser imprecisas.
- É importante considerar o intervalo dos dados usados para treinar o modelo ao fazer previsões, pois prever fora
  desse intervalo (extrapolação) pode levar a resultados menos confiáveis.

  

## Segue link de calclulo simples de Regressao Linear
* Exemplo 1: [RegressaoLinearSimples.py](RegressaoLinearSimples.py)
* Exemplo 2: [Analise_Regressao_Automoveis.py](Analise_Regressao_Automoveis.py)


<hr>


# Probabilidade para Classificação
A probabilidade para classificação refere-se à aplicação de princípios probabilísticos para classificar dados em
diferentes categorias ou classes.

* ## Teorema de Bayes

  O Teorema de Bayes é fundamental na classificação probabilística. Ele ajuda a calcular a probabilidade de uma classe, dado um conjunto de atributos (features).

  ### Fórmula

  `P(Classe | Dados) = (P(Dados | Classe) × P(Classe)) / P(Dados)`
<br>
<br>
  Onde:
  - `P(Classe | Dados)` é a probabilidade posterior de uma classe (saída) dado um conjunto de dados (entrada).
  - `P(Dados | Classe)` é a probabilidade de observar os dados dado que a classe é verdadeira.
  - `P(Classe)` é a probabilidade prévia da classe.
  - `P(Dados)` é a probabilidade dos dados.

* ## Classificador Bayesiano Ingênuo (Naive Bayes)

  O Classificador Bayesiano Ingênuo é um classificador probabilístico baseado no Teorema de Bayes. É chamado de "ingênuo" porque assume que todas as características são independentes entre si.

  ### Fórmula

  `P(Classe|Dados) ∝ P(Classe) × ∏_{i} P(Dado_i|Classe`
<br>
<br>
  Onde:
  - `P(Classe|Dados)` é a probabilidade posterior de uma classe, dado um conjunto de dados.
  - `P(Classe)` é a probabilidade prévia da classe.
  - `∏_{i} P(Dado_i|Classe)` é o produto das probabilidades de cada característica individual, dado a classe.


## Segue link de calclulo de Naive Bayes
* Exemplo 1: [NaiveBayes.py](NaiveBayes.py)

<hr>

## Árvores de Decisão

As Árvores de Decisão são modelos preditivos que utilizam regras binárias (sim/não ou verdadeiro/falso) para calcular uma decisão. A estrutura lembra uma árvore, com um nó raiz, nós internos e folhas.

### Como funcionam?

- **Nó Raiz:** Ponto de partida da árvore, onde é feita a primeira decisão.
- **Nós Internos:** Representam decisões ou testes.
- **Folhas (ou Nós Finais):** Onde a árvore chega a uma conclusão, representando a saída ou a classificação final.

### Processo de Decisão

A partir do nó raiz, a árvore faz perguntas binárias, guiando-se pelos nós internos com base nas respostas, até chegar a um nó folha, onde é dada a saída final.

### Fórmulas e Critérios de Divisão

Em Árvores de Decisão, é crucial escolher os melhores pontos de divisão (ou "splits"). Existem várias métricas para isso:

- **Índice de Gini:** Usado para medir a impureza (ou pureza) de um nó.
  
  - **Fórmula:** `Gini = 1 - ∑(p_i)^2`
  - **Onde:**
    - `p_i` é a proporção de amostras que pertencem à classe i no conjunto.


- **Entropia:** A Entropia é outra métrica que mede a impureza de um nó, baseada na teoria da informação. Quanto maior a entropia, mais desordenado é o nó.

  - **Fórmula da Entropia:** `Entropia = -∑(p_i * log2(p_i))`
  - **Onde:**
    - `p_i` é a proporção de amostras que pertencem à classe i no conjunto.


- **Ganho de Informação:** Baseado na entropia, uma medida da desordem ou impureza.
  
  - **Fórmula:** `IG(T, a) = Entropia(T) - ∑(|T_v| / |T|) * Entropia(T_v)`
  - **Onde:**
    - `IG(T, a)` é o ganho de informação do conjunto total T após a divisão baseada no atributo a.
    - `T` é o conjunto de dados total antes da divisão.
    - `a` é o atributo que está sendo considerado para a divisão.
    - `T_v` são os subconjuntos resultantes da divisão.


- **Redução da Variância:** Usado principalmente para problemas de regressão. Baseia-se na redução da variância do alvo antes e depois da divisão.

## Métricas de Complexidade em Árvores de Decisão

As métricas de complexidade são importantes para avaliar a eficiência e eficácia de uma árvore na tomada de decisões. Elas ajudam a entender se a árvore está muito simples ou muito complexa.

- **Profundidade da Árvore:** Refere-se ao comprimento do caminho mais longo da raiz até uma folha. Uma árvore muito profunda pode ser um sinal de sobreajuste.


- **Número de Folhas:** Quantidade de nós finais (folhas) na árvore. Mais folhas podem indicar uma árvore mais complexa.


- **Número de Nós:** Contagem total de nós na árvore, incluindo nós internos e folhas. Um número maior de nós geralmente indica maior complexidade.


- **Taxa de Erro de Classificação:** Mede a proporção de previsões incorretas em relação ao total de previsões. Uma taxa de erro baixa em dados de treinamento, mas alta em dados de teste, pode indicar sobreajuste.


- **Redução Média da Impureza:** Indica quão eficaz cada característica (ou nó) é para reduzir a incerteza.


- **Índice de Gini ou Entropia nos Nós:** Medidas de impureza ou desordem nos nós. Nós com altos valores de Gini ou Entropia podem indicar que a árvore não está fazendo boas divisões.


- **Alpha de Poda (α):** Usado no algoritmo de poda de custo-complexidade. Determina o equilíbrio entre a complexidade da árvore e o ajuste aos dados de treinamento.


- **Tamanho do Código (para Árvores de Decisão baseadas em regras):** Mede a complexidade em termos do tamanho do código necessário para expressar a árvore. Uma métrica relevante quando as árvores são convertidas em conjuntos de regras.

## Dados Discretos

### Definição

Dados discretos são aqueles que podem assumir apenas valores específicos e distintos. Eles são contáveis e geralmente representam informações que podem ser categorizadas.

### Exemplos

- Número de usuários registrados em um site.
- Quantidade de produtos vendidos.
- Números inteiros como 1, 2, 3, etc.

### Características

- Geralmente representados por números inteiros.
- Não podem ter valores fracionários entre dois pontos de dados adjacentes.
- Frequentemente usados em cenários onde a contagem é relevante.

## Dados Contínuos

### Definição

Dados contínuos são aqueles que podem assumir qualquer valor dentro de um intervalo. Eles são mensuráveis e representam quantidades que podem ser divididas em unidades menores, incluindo frações ou decimais.

### Exemplos

- Tempo necessário para completar uma tarefa.
- Altura ou peso de uma pessoa.
- Temperatura ou velocidade.

### Características

- Podem assumir qualquer valor dentro de um intervalo.
- Frequentemente representados por números reais.
- Usados em cenários onde a medição precisa é essencial.

## Etapas da Indução de uma Árvore de Decisão

### Seleção do Atributo para Divisão

- A indução começa no nó raiz com todo o conjunto de dados.
- O algoritmo seleciona o melhor atributo para dividir os dados, baseando-se em critérios como ganho de informação, índice de Gini ou redução de variância.

### Divisão Baseada no Atributo Escolhido

- O conjunto de dados é dividido em subconjuntos menores, onde cada subconjunto corresponde a um dos possíveis valores do atributo escolhido.
- Para atributos contínuos, o algoritmo determina um ponto de corte.

### Repetição do Processo para Cada Subconjunto

- O algoritmo repete o processo para cada subconjunto, escolhendo o melhor atributo para cada um e dividindo-o mais.
- Este processo continua recursivamente.

### Critério de Parada

- A indução de cada ramo da árvore continua até que um critério de parada seja atingido, como todos os dados no nó pertencem à mesma classe, não há mais atributos para considerar, a profundidade máxima é atingida ou o número de instâncias no nó é menor que um limiar mínimo.

### Criação de um Nó Folha

- Quando o critério de parada é atingido, um nó folha é criado.
- Em tarefas de classificação, o nó folha representa a classe mais comum entre as instâncias no nó.
- Em tarefas de regressão, ele representa a média ou mediana dos valores alvo.

## Considerações Importantes

### Sobreajuste

Um risco comum na indução de árvores de decisão é o sobreajuste aos dados de treinamento. Técnicas como poda da árvore são usadas para mitigar isso.

### Poda da Árvore

A poda pode ser feita durante a construção da árvore (poda prévia) ou após a construção completa (poda posterior), envolvendo a remoção de partes da árvore que podem estar causando sobreajuste.

### Importância do Alinhamento com os Dados

A eficácia de uma árvore de decisão depende fortemente de quão bem suas divisões refletem as relações subjacentes nos dados.

## Segue link de calclulo de Arvore de Decisão
* Utilizando o Graphviz [DecisionTree_Graphviz.py](DecisionTree_Graphviz.py)
* Utilizando o MatPlot [DecisionTree_MatPlot.py](DecisionTree_MatPlot.py)

<hr>



## Aprendizado Ensemble (Aprendizado em Grupo / Métodos de Conjunto)
É um método de Machine Learning que utiliza a combinação de múltiplos modelos de aprendizado para
melhorar o desempenho em tarefas como classificação, regressão e detecção de anomalias. Ao invés de usar um único modelo
preditivo, o aprendizado ensemble emprega um conjunto de modelos para obter melhores resultados.

## Características Principais do Aprendizado Ensemble

* ### Diversidade de Modelos
  O aprendizado ensemble é baseado na diversidade, seja por diferentes algoritmos, conjuntos de dados ou parâmetros.

* ### Redução de Erro
  Visa combinar as forças de vários modelos para minimizar os erros.

* ### Métodos de Ensemble
  * #### Bagging
    - Cria versões variadas do conjunto de dados de treinamento, treina modelos separados nesses conjuntos e combina suas saídas.

  * #### Boosting
    - Constrói modelos de forma sequencial, onde cada novo modelo foca em corrigir os erros dos modelos anteriores.

  * ### Votação e Agregação
  Os modelos no ensemble contribuem para a decisão final através de votação (classificação) ou agregação (regressão).

  * ### Redução de Overfitting
  Pode reduzir o overfitting de modelos complexos.

## Inclusão de Random Forest, Bagging, Boosting e AdaBoost
  * ### Random Forest
    - Um exemplo clássico de bagging.
    - Consiste em múltiplas árvores de decisão, cada uma treinada em um subconjunto aleatório dos dados.
    - Reduz overfitting e melhora a precisão em comparação com uma única árvore de decisão.

  * ### Bagging
    - Técnica usada no Random Forest.
    - Envolve treinar modelos independentemente em diferentes subconjuntos dos dados e depois combinar suas previsões.
    - Reduz a variância dos modelos individuais.

  * ### Boosting
    - Diferente do bagging, boosting treina modelos de forma sequencial.
    - Cada modelo subsequente tenta corrigir os erros do modelo anterior.
    - Aumenta a precisão, mas pode ser mais suscetível a overfitting em comparação com o bagging.

  * ### AdaBoost (Adaptive Boosting)
    - Uma forma específica de boosting.
    - Começa com um modelo fraco e aumenta iterativamente o peso das instâncias mal classificadas.
    - Os modelos subsequentes focam mais nas instâncias difíceis, refinando continuamente o modelo.

## Segue link de calclulo de Random Forest
* [RandomForest.py](RandomForest.py)

<hr>

* ## Classificação Baseada em Modelo (Model-based Classification):
Nesse contexto, a classificação baseada em modelo se refere à criação de um modelo de aprendizado de máquina que é treinado com dados de treinamento para aprender a classificar diferentes tipos de entradas. Isso envolve a construção de um modelo estatístico ou algoritmo que pode fazer previsões ou classificações com base nos padrões identificados nos dados de treinamento. Os modelos podem incluir regressão logística, árvores de decisão, redes neurais, entre outros.

* ## Classificação Baseada em Instância (Instance-based Classification):
A classificação baseada em instância, por outro lado, refere-se a um método de classificação que não cria um modelo global. Em vez disso, ele classifica novas instâncias (dados de entrada) com base na similaridade com instâncias de treinamento conhecidas. O algoritmo considera as instâncias de treinamento como referências e classifica a nova instância com base em quais instâncias de treinamento são mais semelhantes à nova entrada. O algoritmo de classificação k-NN (k-nearest neighbors) é um exemplo desse tipo de abordagem, onde a classificação é feita com base nas k instâncias de treinamento mais próximas à nova instância.

<hr>

# K-Nearest Neighbors (KNN)
O K-Nearest Neighbors (KNN) é um algoritmo de aprendizado supervisionado amplamente utilizado para classificação e regressão. Ele se baseia no princípio de que objetos semelhantes tendem a estar próximos uns dos outros em um espaço de características.

### Distância Euclidiana
A distância Euclidiana é usada para calcular a proximidade entre pontos no espaço de características. A fórmula da distância Euclidiana entre dois pontos A e B em um espaço de n dimensões é:

* #### Formula
`dist(A, B) = sqrt((x1 - x2)^2 + (y1 - y2)^2 + ... + (xn - yn)^2)`


## Classificação no KNN

Para classificar um novo ponto, siga estas etapas:

1. Calcule a distância entre o novo ponto e todos os pontos de treinamento.
2. Selecione os "k" pontos mais próximos com base na distância calculada.
3. Atribua a classe mais frequente entre esses "k" vizinhos ao novo ponto.

## Regressão no KNN

Para regressão, siga estas etapas:

1. Calcule a distância entre o novo ponto e todos os pontos de treinamento.
2. Selecione os "k" pontos mais próximos com base na distância calculada.
3. Calcule a média (ou ponderação) dos valores alvo dos "k" pontos mais próximos como o valor de saída.

## Escolha de "k"

A escolha adequada do valor de "k" é essencial. Valores muito baixos podem levar a overfitting, enquanto valores muito altos podem levar a underfitting. Normalmente, valores ímpares são escolhidos para evitar empates na votação.

O KNN é uma ferramenta valiosa em Machine Learning para problemas de classificação e regressão quando a distribuição dos dados é desconhecida.

## Segue link de calclulo de KNN
* [KNN.py](KNN.py)

<hr>

# KMeans

## Definição

KMeans é um algoritmo de agrupamento (*clustering*) que visa particionar `n` observações em `k` agrupamentos (ou clusters) nos quais cada observação pertence ao cluster com a média mais próxima. É um método utilizado em análise de dados para identificar padrões e organizar os dados em categorias semelhantes com base em suas características.

## Funcionamento

1. **Inicialização**: Escolha de `k` pontos aleatórios dos dados como centros iniciais dos clusters (centroides).
2. **Atribuição**: Atribua cada ponto ao centroide mais próximo, formando `k` clusters.
3. **Atualização**: Recalcule o centroide de cada cluster tomando a média de todos os pontos que foram atribuídos ao cluster.
4. **Iteração**: Repita os passos 2 e 3 até que a posição dos centroides não mude significativamente ou até que um critério de parada seja atingido (por exemplo, um número máximo de iterações).

## Fórmulas

- **Distância Euclidiana** 
  - Usada para medir a proximidade entre pontos e centroides):
    `d(p, q) = \sqrt{\sum_{i=1}^{n} (q_i - p_i)^2}`
  - Onde: 
    - `p` e `q` são dois pontos no espaço de características `n`-dimensional.

- **Atualização do Centroide**:
  `C_i = \frac{1}{|S_i|} \sum_{x_j \in S_i} x_j`
  - Onde: 
    - `C_i` é o centro do cluster `i`, `S_i` é o conjunto de pontos no cluster `i`, e `x_j` são os pontos pertencentes a `S_i`.

## Objetivo

O objetivo do KMeans é minimizar a soma das distâncias quadradas de cada ponto a seu centroide atribuído, conhecida como a **inércia** ou **variância intra-cluster**:
\[
\text{Inércia} = \sum_{i=1}^{k} \sum_{x \in S_i} ||x - C_i||^2
\]

## Considerações

- A escolha de `k` é crucial para o desempenho do algoritmo e pode ser determinada usando métodos como o **método do cotovelo** ou análise de **silhueta**.
- KMeans é sensível à inicialização dos centroides. Métodos como KMeans++ podem ser usados para melhorar a seleção inicial dos centroides.

## Segue link de calclulo de Kmeans
* [Kmeans.py](Kmeans.py)


<hr>

# Algoritmo Apriori

## Introdução
O algoritmo Apriori é uma técnica clássica de mineração de dados usada para extrair conjuntos de itens frequentes e regras de associação. É amplamente utilizado para análise de cesta de compras, onde o objetivo é encontrar combinações de produtos frequentemente comprados juntos. Desenvolvido por Agrawal e Srikant em 1994, o algoritmo Apriori usa uma abordagem iterativa baseada em candidatos, onde conjuntos de itens frequentes de tamanho `k` são usados para explorar candidatos de tamanho `k+1`.

## Como Funciona
1. **Definir o suporte mínimo**: O primeiro passo é definir um limiar de suporte mínimo para os itens. O suporte de um item é definido como a proporção de transações na base de dados que contêm esse item.

2. **Gerar conjuntos de itens frequentes**:
    - Inicialmente, calcula-se o suporte de todos os itens individuais na base de dados e compara-se com o suporte mínimo. Itens que atendem ou excedem o suporte mínimo são considerados itens frequentes de tamanho 1.
    - Pares de itens frequentes são então formados, e seu suporte é calculado. Pares que satisfazem o suporte mínimo avançam para a próxima rodada.
    - O processo continua iterativamente, formando conjuntos maiores de itens frequentes até que não se possam formar mais conjuntos.

3. **Gerar regras de associação**: A partir dos conjuntos de itens frequentes, as regras de associação que satisfazem os limiares de suporte e confiança mínimos são geradas.

## Aplicações
- **Análise de cesta de compras**: Identificar produtos que são frequentemente comprados juntos.
- **Cross-selling**: Desenvolver estratégias de vendas cruzadas com base na compra combinada de produtos.
- **Recomendação de produtos**: Sugerir produtos com base no histórico de compras do cliente.

## Exemplo de Código em Python
Um exemplo simples de implementação do algoritmo Apriori em Python usando a biblioteca `mlxtend`.

* [Apriori.py](Apriori.py)
```python
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd

# Exemplo de conjunto de dados de transações
dataset = [['Leite', 'Pão', 'Cerveja'],
           ['Pão', 'Frango', 'Manteiga'],
           ['Leite', 'Pão'],
           ['Pão', 'Frango'],
           ['Leite', 'Pão', 'Frango', 'Cerveja'],
           ['Leite', 'Pão', 'Frango']]

# Preparação dos dados
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)
print(df)

# Aplicação do algoritmo Apriori
frequent_itemsets = apriori(df, min_support=0.5, use_colnames=True)

# Geração de regras de associação
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

print(frequent_itemsets)
print(rules)
```

<hr>

# Engenharia de Atributos

A Engenharia de Atributos é uma etapa crucial no desenvolvimento de modelos de machine learning, focada em transformar dados brutos em características que melhor representam o problema a ser previsto, melhorando assim o desempenho do modelo.

## Etapas da Engenharia de Atributos

### 1. Compreensão do Domínio e Identificação de Requisitos
- Compreender o problema e o domínio para identificar quais características podem ser relevantes.

### 2. Pré-processamento de Dados
- Preparar os dados, incluindo limpeza e normalização, para facilitar a criação e transformação de atributos.

### 3. Geração e Extração de Características
- **Geração de Características:** Criar novos atributos a partir dos dados existentes que capturam informações importantes não evidentes nos dados brutos.
- **Extração de Características:** Aplicar técnicas para extrair informações de conjuntos de dados complexos (e.g., textos, imagens, séries temporais).
- **Binning de Variáveis Numéricas:** Agrupar valores de atributos numéricos em intervalos discretos para simplificar a representação dos dados.

### 4. Transformação de Atributos
- Modificar atributos existentes para melhorar sua contribuição para o modelo, como aplicar normalizações ou transformações.

### 5. Redução de Dimensionalidade
- Aplicar técnicas como PCA para reduzir o número de características, mantendo a maior parte da informação original.

### 6. Avaliação e Seleção das Características Criadas
- Avaliar a relevância das características criadas e selecionar as mais úteis com base na análise exploratória e modelos iniciais.

### 7. Iteração e Refinamento
- Revisar e ajustar as características com base no feedback do desempenho dos modelos e iterar sobre o processo para melhorar a qualidade das características.

## Conclusão

A Engenharia de Atributos é um processo iterativo e criativo que exige conhecimento do domínio e compreensão profunda dos dados. As características bem projetadas podem significativamente melhorar a eficácia dos modelos de machine learning, tornando esta etapa fundamental para o sucesso de projetos de análise preditiva.
