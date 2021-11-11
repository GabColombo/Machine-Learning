import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

previsores = pd.read_csv('entradas-breast.csv')
classe = pd.read_csv('saidas-breast.csv')

def criarRede():
    classificador = Sequential()
    classificador.add(Dense(units = 16, activation = 'relu', # 16 Neurônios na camada oculta
                            kernel_initializer = 'random_uniform', input_dim = 30)) # 30 neurônios na entrada
    
    classificador.add(Dropout(0.2)) # Irá zerar 20% dos neurônios da camada de entrada (ajuda a evitar o overfitting)
    
    classificador.add(Dense(units = 16, activation = 'relu', # 16 Neurônios na segunda camada oculta
                            kernel_initializer = 'random_uniform'))
    
    classificador.add(Dropout(0.2)) # Camada de Dropout com 20%
    
    classificador.add(Dense(units = 1, activation = 'sigmoid')) # 1 neurônio na camada de saída
        
    otimizador = keras.optimizers.Adam(lr = 0.001, decay = 0.0001, # Decay diminui o valor do learning rate a cada época
                                       clipvalue = 0.5) # Clipvalue congela o valor caso ele pare de se aproximar do mínimo global
    
    classificador.compile(optimizer = otimizador, loss = 'binary_crossentropy', # optimizer é a descida do gradiente
                          metrics = ['binary_accuracy']) # loss é utilizado para o erro, nesse caso é o binário pois nosso problema é binário
    return classificador

classificador = KerasClassifier(build_fn = criarRede, 
                                epochs = 100,
                                batch_size = 10)

resultados = cross_val_score(estimator = classificador,
                             X = previsores, y = classe,
                             cv = 10, scoring = 'accuracy') # CV = Número de divisões para a Cross Validation

media = resultados.mean()
desvio_padrao = resultados.std()