import pandas as pd

base = pd.read_csv('plano-saude2.csv')

X = base.iloc[:, 0:1]
Y = base.iloc[:, 1:2]

from sklearn.preprocessing import StandardScaler

scaler_x = StandardScaler()
X = scaler_x.fit_transform(X)
scaler_y = StandardScaler()
Y = scaler_y.fit_transform(Y)

from sklearn.neural_network import MLPRegressor
regressor = MLPRegressor()
regressor.fit(X, Y)

score = regressor.score(X, Y)

import matplotlib.pyplot as plt
plt.scatter(X, Y)
plt.plot(X, regressor.predict(X), color = 'red')
plt.title('Regressão com RNA')
plt.xlabel('Idade')
plt.ylabel('Custo')

valor = scaler_x.transform([[40]])
previsao = scaler_y.inverse_transform(regressor.predict(valor))