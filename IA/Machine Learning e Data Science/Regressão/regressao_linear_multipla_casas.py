import pandas as pd

base = pd.read_csv('house-prices.csv')

X = base.iloc[:, 3:19].values #Serão utilizados os atributos do 3 ao 18 para fazer o treinamento
Y = base.iloc[:, 2].values #Preços

from sklearn.model_selection import train_test_split
X_treinamento, X_teste, Y_treinamento, Y_teste = train_test_split(X, Y, test_size = 0.3, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_treinamento, Y_treinamento)
score = regressor.score(X_treinamento, Y_treinamento)

previsoes = regressor.predict(X_teste)

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(Y_teste, previsoes)

regressor.score(X_teste, Y_teste) #Compara os valores estimados com os reais da base de dados de teste

regressor.intercept_
regressor.coef_