import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np

base = pd.read_csv('credit-card-clients.csv', header = 1)

# Cria um novo atributo na base que é a soma das dívidas de cada pessoa
base['BILL_TOTAL'] = base['BILL_AMT1'] + base['BILL_AMT2'] + base['BILL_AMT3'] + base['BILL_AMT4'] + base['BILL_AMT5'] + base['BILL_AMT6']

# Variável com os atributos que serão utilizados na aprendizagem (Limite de crédito e dívidas totais)
X = base.iloc[:, [1, 25]].values

# Escalonamento
scaler = StandardScaler()
X = scaler.fit_transform(X)

dbscan = DBSCAN(eps = 0.37, min_samples = 4)
previsoes = dbscan.fit_predict(X)
unicos, quantidade = np.unique(previsoes, return_counts = True)

plt.scatter(X[previsoes == 0, 0], X[previsoes == 0, 1], s = 100, c = 'red', label = 'Cluster 1') # plota somente os pontos com rótulo 0
plt.scatter(X[previsoes == 1, 0], X[previsoes == 1, 1], s = 100, c = 'orange', label = 'Cluster 2') # plota somente os pontos com rótulo 0
plt.scatter(X[previsoes == 2, 0], X[previsoes == 2, 1], s = 100, c = 'green', label = 'Cluster 3') # plota somente os pontos com rótulo 0
#plt.scatter(X[previsoes == 3, 0], X[previsoes == 3, 1], s = 100, c = 'blue', label = 'Cluster 4') # plota somente os pontos com rótulo 0
plt.xlabel('Limite')
plt.ylabel('Gastos')
plt.legend()

# Une a base de dados com os rótulos
lista_clientes = np.column_stack((base, previsoes))
# Ordena a base de dados de acordo com o rótulo (Nesse caso está no atributo 26)
lista_clientes = lista_clientes[lista_clientes[:, 26].argsort()]