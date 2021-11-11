import pandas as pd

base = pd.read_csv('plano-saude2.csv')

X = base.iloc[:, 0:1].values
Y = base.iloc[:, 1].values

# Regressão linear simples

from sklearn.linear_model import LinearRegression
regressor1 = LinearRegression()
regressor1.fit(X, Y)
score1 = regressor1.score(X, Y) #Precisão

regressor1.predict([[40]])

import matplotlib.pyplot as plt
plt.scatter(X, Y)
plt.plot(X, regressor1.predict(X), color = 'cyan')
plt.title('Regressão Linear')
plt.xlabel('Idade')
plt.ylabel('Custo')

# Regressão Polinomial

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 2) # Irá criar um polinomio de ordem 2 | Valores de ordem mais altos tendem a dar um resultado melhor, porém, valores muito altos, podem resultar em overfitting
X_poly = poly.fit_transform(X) # Eleva todos os atributos ao quadrado (degree = 2)

regressor2 = LinearRegression() # Utiliza a mesma lógica da regressão linear, porém elevando a ordem dos polinômios
regressor2.fit(X_poly, Y)
score2 = regressor2.score(X_poly, Y)

regressor2.predict(poly.transform([[40]]))

plt.scatter(X, Y)
plt.plot(X, regressor2.predict(X_poly), color = 'red')
plt.title('Regressão Polinomial')
plt.xlabel('Idade')
plt.ylabel('Custo')