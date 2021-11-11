import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

'''x = [20, 27, 21, 37, 46, 53, 55, 47, 52, 32, 39, 41, 39, 48, 48]
y = [1000, 1200, 2900, 1850, 900, 950, 2000, 2100, 3000, 5900, 4100, 5100, 7000, 5000, 6500]

plt.scatter(x, y)'''

# Criação da base de dados
base = np.array([[20, 1000], [27, 1200], [21, 2900], [37, 1850], [46, 900], 
                 [53, 950], [55, 2000], [47, 2100], [52, 3000], [32, 5900], 
                 [39, 4100], [41, 5100], [39, 7000], [48, 5000], [48, 6500]])

# Escalonamento dos dados
scaler = StandardScaler()
base = scaler.fit_transform(base)

#Aprendizagem
kmeans = KMeans(n_clusters = 3) # 3 clusters
kmeans.fit(base)

#Visualização
centroides = kmeans.cluster_centers_  # local dos centroides
rotulos = kmeans.labels_ # rotulo que o algoritmo definiu para cada ponto

cores = ["g.", "r.", "b."]

for i in range(len(x)):
    plt.plot(base[i][0], base[i][1], cores[rotulos[i]], markersize = 15) # plota o grafico com a cor de cada rotulo
plt.scatter(centroides[:, 0],centroides[:, 1], marker = 'x') # plota os centroides

