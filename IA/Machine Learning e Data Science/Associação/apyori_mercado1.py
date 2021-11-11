import pandas as pd

base = pd.read_csv('mercado.csv', header = None)

transacoes = []  # Os dados serão convertidos de DataFrame para Lista
for i in range(0, base.shape[0]):
    transacoes.append([str(base.values[i, j]) for j in range(0,base.shape[1])])
    
from apyori import apriori
regras = apriori(transacoes, min_support = 0.3, min_confidence = 0.8, min_lift = 2.0, min_lenght = 2) 

resultados = list(regras) # Lista com as regras de associação

resultados2 = [list(x) for x in resultados]

resultado_formatado = []

for j in range(0, len(resultados2)):
    resultado_formatado.append([list(x) for x in resultados2[j][2]])

resultado_formatado