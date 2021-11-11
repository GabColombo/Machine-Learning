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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

import numpy as np

from sklearn.model_selection import StratifiedKFold #Garante que cada divisão seja uma boa representação do conjunto ao todo
from sklearn.metrics import accuracy_score, confusion_matrix

resultados30 = []

for i in range(30):
    kfold = StratifiedKFold(n_splits = 10, shuffle = True, random_state = i) #10 é um bom número para divisões. Se shuffle = True e random_state = 0, os resultados serão sempre os mesmos. Se shuffle = True e random_state diferente de 0, os resultados vão variar
    resultados_rodada = []
    
    for indice_treinamento, indice_teste in kfold.split(previsores, np.zeros(shape=(previsores.shape[0], 1))):
        
        #classificador = GaussianNB()
        #classificador = DecisionTreeClassifier()
        #classificador = RandomForestClassifier(n_estimators = 40, criterion = 'entropy', random_state = 0)
        #classificador = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
        #classificador = LogisticRegression(random_state = 1)
        #classificador = SVC(kernel = 'rbf', random_state = 1, C = 2)
        classificador = MLPClassifier(verbose = True, max_iter = 1000, tol = 0.000010, solver = 'adam', 
                                      hidden_layer_sizes = (100), activation = 'relu')
        
        classificador.fit(previsores[indice_treinamento], classe[indice_treinamento])
        previsoes = classificador.predict(previsores[indice_teste])
        precisao = accuracy_score(classe[indice_teste], previsoes)
        resultados_rodada.append(precisao)
    resultados_rodada = np.asarray(resultados_rodada)
    media = resultados_rodada.mean()
    resultados30.append(media)
    print(i)
    
resultados30 = np.asarray(resultados30)

for i in range(resultados30.size):
    print(str(resultados30[i]).replace('.', ','))

