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

from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25, random_state=0)

#importação da biblioteca
from sklearn.neural_network import MLPClassifier #MultLayerPerceptron
#criação do classificador
classificador = MLPClassifier(verbose = True, max_iter=1000, tol=0.000010, solver='adam', 
                              hidden_layer_sizes=(100), activation='relu')  #verbose = mostrar resultados, max_iter=número máximo de iterações, tol=tolerância  (caso o valor do erro diminua menos que a tolerância, as iterações serão abortadas), solver = gradiente, activation = função de ativação, hidden_layer_size = quantidade de neurônios na camada oculta específica
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)

from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)