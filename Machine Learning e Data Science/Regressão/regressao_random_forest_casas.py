import pandas as pd

base = pd.read_csv('house-prices.csv')

X = base.iloc[:, 3:19].values
Y = base.iloc[:, 2].values

from sklearn.model_selection import train_test_split
X_treinamento, X_teste, Y_treinamento, Y_teste = train_test_split(X, Y, test_size = 0.3, random_state = 0)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100)
regressor.fit(X_treinamento, Y_treinamento)
score = regressor.score(X_treinamento, Y_treinamento)

previsoes = regressor.predict(X_teste)

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(previsoes, Y_teste)

regressor.score(X_teste, Y_teste)