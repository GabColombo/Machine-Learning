import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs

#Cria modelos de base de dados
pontos, rotulos = make_blobs(n_samples = 200, centers = 4) # A variavel 'ponto' possui os pontos x e y no grafico e a variavel 'rotulo' apresenta os rótulos de cada ponto
plt.scatter(pontos[:, 0], pontos[:, 1])

# Aprendizagem
kmeans = KMeans(n_clusters = 4)
kmeans.fit(pontos)

previsoes = kmeans.predict(pontos) # Mostra os rotulos
plt.scatter(pontos[:, 0], pontos[:, 1], c = previsoes) # Plota o gráfico, c = cores (cada rotulo possuirá uma cor)