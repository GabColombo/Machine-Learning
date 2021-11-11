import pandas as pd

base = pd.read_csv('plano-saude.csv')

X = base.iloc[:, 0].values
Y = base.iloc[:, 1].values

import numpy as np
correlacao = np.corrcoef(X, Y) #Correlação entre X e Y. É recomendado ter uma correlação alta para utilizar regressão

X = X.reshape(-1, 1) #Converte para matriz

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, Y)

#b0
regressor.intercept_

#b1
regressor.coef_

import matplotlib.pyplot as plt
plt.scatter(X, Y)
plt.plot(X, regressor.predict(X), color = 'red') #Plota o valor real e o previsto
plt.title('Regressão Linear Simples')
plt.xlabel('Idade')
plt.ylabel('Custo')

previsao1 = regressor.predict([[40]])
previsao2 = regressor.intercept_ + regressor.coef_*40 #Regressão utilizando a formula

scores = regressor.score(X, Y) #Semelhante à precisão dos algoritmos de classificação

from yellowbrick.regressor import ResidualsPlot
visualizador = ResidualsPlot(regressor)
visualizador.fit(X, Y)
visualizador.poof()