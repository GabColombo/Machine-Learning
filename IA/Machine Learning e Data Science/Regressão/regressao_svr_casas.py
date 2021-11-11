import pandas as pd

base = pd.read_csv('house-prices.csv')

X = base.iloc[:, 3:19].values
Y = base.iloc[:, 2:3].values

# Escalonamento
from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()
X = scaler_x.fit_transform(X)
scaler_y = StandardScaler()
Y = scaler_y.fit_transform(Y)

# Divisão das bases de treinamento e teste
from sklearn.model_selection import train_test_split
X_treinamento, X_teste, Y_treinamento, Y_teste = train_test_split(X, Y, test_size = 0.3, random_state = 0)

# SVR
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_treinamento, Y_treinamento)
score = regressor.score(X_treinamento, Y_treinamento)

score_teste = regressor.score(X_teste, Y_teste)

# Desescalonamento
previsoes = regressor.predict(X_teste)
Y_teste = scaler_y.inverse_transform(Y_teste)
previsoes = scaler_y.inverse_transform(previsoes)

# Erro
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(previsoes, Y_teste)
