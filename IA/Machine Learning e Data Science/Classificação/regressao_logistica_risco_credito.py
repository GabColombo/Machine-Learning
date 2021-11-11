import pandas as pd

base = pd.read_csv('risco-credito2.csv')

previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
previsores[:, 0] = labelencoder.fit_transform(previsores[:, 0])
previsores[:, 1] = labelencoder.fit_transform(previsores[:, 1])
previsores[:, 2] = labelencoder.fit_transform(previsores[:, 2])
previsores[:, 3] = labelencoder.fit_transform(previsores[:, 3])

#Importar Biblioteca
from sklearn.linear_model import LogisticRegression
#Criar Classificador
classificador = LogisticRegression()
classificador.fit(previsores, classe)

#Visualizar Coeficientes
print(classificador.intercept_) #B0
print(classificador.coef_) #Coeficientes (B1, B2, B3, B4, etc.)

#Testes
##História boa, Dívida alta, Garantia nenhuma, Renda > 35
##História ruim, Dívida alta, Garantia adequada, Renda < 15
resultado = classificador.predict([[0, 0, 1, 2], [3, 0, 0, 0]])
resultado2 = classificador.predict_proba([[0, 0, 1, 2], [3, 0, 0, 0]]) #Probabilidade
print(resultado)
print(resultado2)
