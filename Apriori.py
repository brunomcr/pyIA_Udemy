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