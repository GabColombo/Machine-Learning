import pandas as pd

base = pd.read_csv('credit-data.csv')

base.loc[base.age < 0, 'age'] = 40.92 #Corrigir valores negativos
               
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

from sklearn.impute import SimpleImputer  #Corrigir valores vazios
imputer = SimpleImputer(strategy='mean')
imputer = imputer.fit(previsores[:, 1:4])
previsores[:, 1:4] = imputer.transform(previsores[:, 1:4])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25, random_state=0)

#importação da biblioteca
from sklearn.linear_model import LogisticRegression

#criação do classificador
classificador = LogisticRegression(random_state=1)
classificador.fit(previsores_treinamento, classe_treinamento)

previsoes = classificador.predict(previsores_teste)

from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)

import collections
collections.Counter(classe_teste) #Baseline de 0.872