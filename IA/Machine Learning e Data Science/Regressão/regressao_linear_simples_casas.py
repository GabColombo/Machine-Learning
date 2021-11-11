import pandas as pd

base = pd.read_csv('house-prices.csv')

X = base.iloc[:, 5:6].values #Utilizando 5:6 ele pega somente o atributo 5 mas já fica no modo matriz, necessário para utilizar o scikit-learn
Y = base.iloc[:, 2].values

from sklearn.model_selection import train_test_split
X_treinamento, X_teste, Y_treinamento, Y_teste = train_test_split(X, Y, test_size=0.3, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_treinamento, Y_treinamento)

score = regressor.score(X_treinamento, Y_treinamento) #Score de apenas 49% utilizando apenas o tamanho da casa para estimar o preço

import matplotlib.pyplot as plt
plt.scatter(X_treinamento, Y_treinamento)
plt.plot(X_treinamento, regressor.predict(X_treinamento), color = 'red')

previsoes = regressor.predict(X_teste)

resultado = abs(Y_teste - previsoes) #Diferença entre os preços reais e estimados
resultado.mean() #Média de erro dos resultados

from sklearn.metrics import mean_absolute_error, mean_squared_error #É recomendavel utilizar o mse para treinamento e mae para teste
mae = mean_absolute_error(Y_teste, previsoes)
mse = mean_squared_error(Y_teste, previsoes)

plt.scatter(X_teste, Y_teste)
plt.plot(X_teste, regressor.predict(X_teste), color = 'red')

regressor.score(X_teste, Y_teste)