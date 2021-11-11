# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 23:02:26 2019

@author: gabri
"""

import pandas as pd
base = pd.read_csv('credit-data.csv')
base.describe()
base.loc[base['age'] < 0]

#Apagar a coluna (não interessante)
base.drop('age', 1, inplace=True)

#Apagar somente os registros com problema
base.drop(base[base.age < 0].index, inplace=True)

#Preencher os valores manualmente

#Preencher os valores com a média
base.mean()
base['age'].mean()
base['age'][base.age > 0].mean()
base.loc[base.age < 0, 'age'] = 40.92

#Encontrar valores nulos
pd.isnull(base['age'])
base.loc[pd.isnull(base['age'])]

#Separar atributos PREVISORES do atributo CLASSE
previsores = base.iloc[:, 1:4].values #Linhas, Colunas -> Deixamos o id de fora pois não é interessante para análise
classe = base.iloc[:, 4].values #Linhas, Coluna da classe

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy='mean', axis=0)
imputer = imputer.fit(previsores[:, 0:3])
previsores[:, 0:3] = imputer.transform(previsores[:,0:3])

#Escalonamento dos previsores utilizando padronização
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)


