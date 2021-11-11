import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

base = pd.read_csv('credit-card-clients.csv', header = 1)

# Cria um novo atributo na base que é a soma das dívidas de cada pessoa
base['BILL_TOTAL'] = base['BILL_AMT1'] + base['BILL_AMT2'] + base['BILL_AMT3'] + base['BILL_AMT4'] + base['BILL_AMT5'] + base['BILL_AMT6']

# Variável com os atributos que serão utilizados na aprendizagem
X = base.iloc[:, [1, 2, 3, 4, 5, 25]].values

# Escalonamento
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Elbow Method (Define o melhor número de clusters para a base de dados)
wcss = [] # Variavel que irá armazenar as distancias
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, random_state = 0) # Testes com clusters entre 1 e 10
    kmeans.fit(X)
    wcss.append(kmeans.inertia_) # Salva a distância

# Gráfico do Elbow Method
plt.plot(range(1, 11), wcss)
plt.xlabel('Número de clusters')
plt.ylabel('WCSS')
plt.show()

# Aprendizagem com o número de cluster que mais se adequou pelo Elbow Method
kmeans = KMeans(n_clusters = 4, random_state = 0)
previsoes = kmeans.fit_predict(X) # Faz a aprendizagem e mostr os rótulos

# Une a base de dados com os rótulos
lista_clientes = np.column_stack((base, previsoes))
# Ordena a base de dados de acordo com o rótulo (Nesse caso está no atributo 26)
lista_clientes = lista_clientes[lista_clientes[:, 26].argsort()]