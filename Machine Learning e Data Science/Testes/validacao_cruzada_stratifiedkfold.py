import pandas as pd

base = pd.read_csv('credit-data.csv')
base.loc[base.age < 0, 'age'] = 40.92
               
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer()
imputer = imputer.fit(previsores[:, 1:4])
previsores[:, 1:4] = imputer.transform(previsores[:, 1:4])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

from sklearn.naive_bayes import GaussianNB

import numpy as np
#a = np.zeros(5)
#previsores.shape
#previsores.shape[0]
#b = np.zeros(shape=(previsores.shape[0], 1))

from sklearn.model_selection import StratifiedKFold #Garante que cada divisão seja uma boa representação do conjunto ao todo
from sklearn.metrics import accuracy_score, confusion_matrix

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0) #10 é um bom número para divisões. Se shuffle = True e random_state = 0, os resultados serão sempre os mesmos. Se shuffle = True e random_state diferente de 0, os resultados vão variar
resultados = []
matrizes = []
for indice_treinamento, indice_teste in kfold.split(previsores, np.zeros(shape=(previsores.shape[0], 1))):
    #print('Indice treinamento: ', indice_treinamento, 'Indice teste: ', indice_teste)
    classificador = GaussianNB()
    classificador.fit(previsores[indice_treinamento], classe[indice_treinamento])
    previsoes = classificador.predict(previsores[indice_teste])
    precisao = accuracy_score(classe[indice_teste], previsoes)
    matrizes.append(confusion_matrix(classe[indice_teste], previsoes))
    resultados.append(precisao)

matriz_final = np.mean(matrizes, axis=0)
resultados = np.asarray(resultados) #converte lista para tuplas
resultados.mean()
resultados.std()
