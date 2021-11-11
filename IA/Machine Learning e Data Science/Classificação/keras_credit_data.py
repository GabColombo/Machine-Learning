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
from keras.models import Sequential
from keras.layers import Dense #Cada neurônio se conecta com todos os outros da próxima camada

#criação do classificador
classificador = Sequential()
classificador.add(Dense(units=2, activation='relu', input_dim=3)) #1ª Camada Oculta: units = Qtd de neurônis na camada oculta, imput_dim = Qtd de neurônios na camada de entrada
classificador.add(Dense(units=2, activation='relu')) #2ª Camada Oculta
classificador.add(Dense(units=1, activation='sigmoid')) #Camada de Saída
classificador.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) #optimizer = grdiente, loss = calculo para o erro

classificador.fit(previsores_treinamento, classe_treinamento, batch_size=10, epochs=100) #Batch size = atributos calculados até que o peso seja atualizado, nb_epoch = número de iterações 
previsoes = classificador.predict(previsores_teste)
previsoes = (previsoes>0.5) #Caso previsões seja maior que 0.5, recebe True. Caso contrário recebe False

from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)