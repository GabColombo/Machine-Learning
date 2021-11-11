import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasRegressor

base = pd.read_csv('autos.csv', encoding = 'ISO-8859-1')

# Apagar atributos que não serão úteis na aprendizagem
base = base.drop('dateCrawled', axis = 1) 
base = base.drop('dateCreated', axis = 1)
base = base.drop('nrOfPictures', axis = 1)
base = base.drop('postalCode', axis = 1)
base = base.drop('lastSeen', axis = 1)
base = base.drop('name', axis = 1)
base = base.drop('seller', axis = 1)
base = base.drop('offerType', axis = 1)

# Valores inconsistentes
base = base[base.price > 10]
base = base[base.price < 350000]

# Trocar o valores nulos pelos que mais aparecem
valores = {'vehicleType': 'limousine', 'gearbox': 'manuell', 'model': 'golf', 'fuelType': 'benzin',
         'notRepairedDamage': 'nein'}
base = base.fillna(value = valores)

# Label Encoder - Transformar os dados categóricos em numéricos

previsores = base.iloc[:, 1:13].values
preco_real = base.iloc[:, 0].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_previsores = LabelEncoder()
previsores[:, 0] = labelencoder_previsores.fit_transform(previsores[:, 0]) # Encode da posição 0
previsores[:, 1] = labelencoder_previsores.fit_transform(previsores[:, 1])
previsores[:, 3] = labelencoder_previsores.fit_transform(previsores[:, 3])
previsores[:, 5] = labelencoder_previsores.fit_transform(previsores[:, 5])
previsores[:, 8] = labelencoder_previsores.fit_transform(previsores[:, 8])
previsores[:, 9] = labelencoder_previsores.fit_transform(previsores[:, 9])
previsores[:, 10] = labelencoder_previsores.fit_transform(previsores[:, 10])

# OneHotEncoder - Necessário criar Dummies pois um atributos não é mais importante que outro (categórico)
onehotencoder = OneHotEncoder(categorical_features = [0, 1, 3, 5, 8, 9, 10])
previsores = onehotencoder.fit_transform(previsores).toarray()

def criar_rede():
    regressor = Sequential()
    regressor.add(Dense(units = 158, activation = 'relu', input_dim = 316))
    regressor.add(Dense(units = 158, activation = 'relu'))
    regressor.add(Dense(units = 1, activation = 'linear'))
    regressor.compile(loss = 'mean_absolute_error', optimizer = 'adam', 
                      metrics = ['mean_absolute_error'])
    return regressor

regressor = KerasRegressor(build_fn = criar_rede,
                           epochs = 100, batch_size = 300)

resultado = cross_val_score(estimator = regressor,
                            X = previsores, y = preco_real,
                            cv = 10, scoring = 'neg_mean_absolute_error')
media = resultado.mean()
desvio_padrao = resultado.std()