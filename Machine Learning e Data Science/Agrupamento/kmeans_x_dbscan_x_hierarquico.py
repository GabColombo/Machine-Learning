import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn import datasets
import numpy as np

pontos, rotulos = datasets.make_moons(n_samples = 1500, noise = 0.09)
plt.scatter(pontos[:, 0], pontos[:, 1], s = 5)

cores = np.array(['red', 'green'])

# K-means
kmeans = KMeans(n_clusters = 2)
previsoes = kmeans.fit_predict(pontos)
plt.scatter(pontos[:, 0], pontos[:, 1], s = 5, color = cores[previsoes])

# Hierarquico
hc = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage = 'ward')
previsoes = hc.fit_predict(pontos)
plt.scatter(pontos[:, 0], pontos[:, 1], s = 5, color = cores[previsoes])

# DBSCAN
dbscan = DBSCAN(eps = 0.1)
previsoes = dbscan.fit_predict(pontos)
plt.scatter(pontos[:, 0], pontos[:, 1], s = 5, color = cores[previsoes])