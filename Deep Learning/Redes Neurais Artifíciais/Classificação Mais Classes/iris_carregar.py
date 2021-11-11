import numpy as np
import pandas as pd
from keras.models import model_from_json

arquivo = open('classificador-iris.json', 'r')
estrutura_rede = arquivo.read()
arquivo.close()

classificador = model_from_json(estrutura_rede)
classificador.load_weights('classificador-iris.h5')

novo = np.array([[4.5, 3.9, 2.5, 0.3]])

previsao = classificador.predict(novo)
previsao = (previsao > 0.5)

if previsao[0][0] == True and previsao[0][1] == False and previsao[0][2] == False:
    print('Iris setosa')
elif previsao[0][0] == False and previsao[0][1] == True and previsao[0][2] == False:
    print('Iris virginica')
elif previsao[0][0] == False and previsao[0][1] == False and previsao[0][2] == True:
    print('Iris versicolor')