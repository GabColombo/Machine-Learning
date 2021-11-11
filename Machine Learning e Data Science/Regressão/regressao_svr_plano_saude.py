import pandas as pd

base = pd.read_csv('plano-saude2.csv')

X = base.iloc[:, 0:1]
Y = base.iloc[:, 1:2]

# Escalonamento
from sklearn.preprocessing import StandardScaler # Ao utilizar o rbf, é necessário fazer escalonamento
scaler_x = StandardScaler()
X = scaler_x.fit_transform(X)
scaler_y = StandardScaler()
Y = scaler_y.fit_transform(Y)

# Kernel Linear
from sklearn.svm import SVR
regressor_linear = SVR(kernel= 'linear')
regressor_linear.fit(X, Y)

import matplotlib.pyplot as plt
plt.scatter(X, Y)
plt.plot(X, regressor_linear.predict(X), color = 'red')

regressor_linear.score(X, Y)

# Kernel Poly
regressor_poly = SVR(kernel = 'poly', degree = 3)
regressor_poly.fit(X, Y)

import matplotlib.pyplot as plt
plt.scatter(X, Y)
plt.plot(X, regressor_poly.predict(X), color = 'cyan')

regressor_poly.score(X, Y)

# Kernel rbf
regressor_rbf = SVR(kernel = 'rbf')
regressor_rbf.fit(X, Y)

import matplotlib.pyplot as plt
plt.scatter(X, Y)
plt.plot(X, regressor_rbf.predict(X), color = 'purple')
regressor_rbf.score(X, Y)

# Previsões
valor = scaler_x.transform([[40]])
previsao_linear = scaler_y.inverse_transform(regressor_linear.predict(valor))
previsao_poly = scaler_y.inverse_transform(regressor_poly.predict(valor))
previsao_rbf = scaler_y.inverse_transform(regressor_rbf.predict(valor))