import pandas as pd

base = pd.read_csv('plano-saude2.csv')

X = base.iloc[:, 0:1].values
Y = base.iloc[:, 1].values

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10) # Utiliza mse como padrão
regressor.fit(X, Y)
score = regressor.score(X, Y)

import numpy as np
X_teste = np.arange(min(X), max(X), 0.1)
X_teste = X_teste.reshape(-1, 1)

import matplotlib.pyplot as plt
plt.scatter(X, Y)
plt.plot(X_teste, regressor.predict(X_teste), color = 'red')
plt.title('Regressão Random Forest')
plt.xlabel('Idade')
plt.ylabel('Custo')

previsao = regressor.predict([[40]])