import pandas as pd

base = pd.read_csv('house-prices.csv')

X = base.iloc[:, 3:19]
Y = base.iloc[:, 2:3]

from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()
X = scaler_x.fit_transform(X)
scaler_y = StandardScaler()
Y = scaler_y.fit_transform(Y)

from sklearn.model_selection import train_test_split
X_treinamento, X_teste, Y_treinamento, Y_teste = train_test_split(X, Y, test_size = 0.3, random_state = 0)

from sklearn.neural_network import MLPRegressor
regressor = MLPRegressor(hidden_layer_sizes = (9, 9))
regressor.fit(X_treinamento, Y_treinamento)

score = regressor.score(X_treinamento, Y_treinamento)
score_teste = regressor.score(X_teste, Y_teste)

previsoes = regressor.predict(X_teste)

Y_teste = scaler_y.inverse_transform(Y_teste)
previsoes = scaler_y.inverse_transform(previsoes)

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(previsoes, Y_teste)