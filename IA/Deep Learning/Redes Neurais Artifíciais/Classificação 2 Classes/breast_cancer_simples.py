import pandas as pd
    
previsores = pd.read_csv('entradas-breast.csv')
classe = pd.read_csv('saidas-breast.csv')
    
from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size = 0.25)

import keras
from keras.models import Sequential
from keras.layers import Dense
    
# Criação da rede neural
    
classificador = Sequential()
classificador.add(Dense(units = 16, activation = 'relu', # 16 Neurônios na camada oculta
                        kernel_initializer = 'random_uniform', input_dim = 30)) # 30 neurônios na entrada

classificador.add(Dense(units = 16, activation = 'relu', # 16 Neurônios na segunda camada oculta
                        kernel_initializer = 'random_uniform'))

classificador.add(Dense(units = 1, activation = 'sigmoid')) # 1 neurônio na camada de saída
    
otimizador = keras.optimizers.Adam(lr = 0.001, decay = 0.0001, # Decay diminui o valor do learning rate a cada época
                                   clipvalue = 0.5) # Clipvalue congela o valor caso ele pare de se aproximar do mínimo global

classificador.compile(optimizer = otimizador, loss = 'binary_crossentropy', # optimizer é a descida do gradiente
                      metrics = ['binary_accuracy']) # loss é utilizado para o erro, nesse caso é o binário pois nosso problema é binário
    

#classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy', # optimizer é a descida do gradiente
#                      metrics = ['binary_accuracy']) # loss é utilizado para o erro, nesse caso é o binário pois nosso problema é binário
    
# Treinamento

classificador.fit(previsores_treinamento, classe_treinamento, batch_size = 10, epochs = 100)

pesos0 = classificador.layers[0].get_weights()  # Pesos da camada de entrada com a primeira camada oculta
print(pesos0)                                   # Possui duas posições pois cria a unidade de Bias automaticamente
print(len(pesos0))

pesos1 = classificador.layers[1].get_weights() # Pesos entre a primeira e a segunda camada oculta + unidade de Bias

pesos2 = classificador.layers[2].get_weights() # Pesos entre a segunda camada oculta e a camda de saída + unidade de Bias

# Accuracy

previsoes = classificador.predict(previsores_teste)
previsoes = (previsoes > 0.5) # Se a saída > 0.5 rece True, senão rece False

from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)

resultado = classificador.evaluate(previsores_teste, classe_teste) # Retorna um valor de erro e o valor da precisão