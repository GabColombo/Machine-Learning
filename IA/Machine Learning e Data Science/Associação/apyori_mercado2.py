import pandas as pd

base = pd.read_csv('mercado2.csv', header = None)
transacoes = []
for i in range(0, base.shape[0]):
    transacoes.append([str(base.values[i, j]) for j in range(0, base.shape[1])])

from apyori import apriori
# No suporte estão os produtos que foram vendido pelo menos 4 vezes por dia 4*7 / 7500 = 0.003
# Um valor ideal para o lift é entre 5 e 7
regras = apriori(transacoes, min_support = 0.003, min_confidence = 0.2, min_lift = 3.0, min_lenght = 2)

resultados = list(regras)

resultados2 = [list(x) for x in resultados]

resultado_formatado = []
for j in range(0, 5):
    resultado_formatado.append([list(x) for x in resultados2[j][2]])

resultado_formatado