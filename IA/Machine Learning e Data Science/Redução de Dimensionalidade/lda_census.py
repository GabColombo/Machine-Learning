import pandas as pd

base = pd.read_csv('census.csv')

previsores = base.iloc[:, 0:14].values
classe = base.iloc[:, 14].values

from sklearn.preprocessing import LabelEncoder
label_encoder_previsores = LabelEncoder()
previsores[:, 1] = label_encoder_previsores.fit_transform(previsores[:, 1])
previsores[:, 3] = label_encoder_previsores.fit_transform(previsores[:, 3])
previsores[:, 5] = label_encoder_previsores.fit_transform(previsores[:, 5])
previsores[:, 6] = label_encoder_previsores.fit_transform(previsores[:, 6])
previsores[:, 7] = label_encoder_previsores.fit_transform(previsores[:, 7])
previsores[:, 8] = label_encoder_previsores.fit_transform(previsores[:, 8])
previsores[:, 9] = label_encoder_previsores.fit_transform(previsores[:, 9])
previsores[:, 13] = label_encoder_previsores.fit_transform(previsores[:, 13])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size = 0.15, random_state = 0)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components = 1) # O LDA é baseado na classe, como nesse caso existem apenas 2 classes, o maior valor de atributo será igual a 1
previsores_treinamento = lda.fit_transform(previsores_treinamento, classe_treinamento) # Como a aprendizagem é supervisionada, é necessário passar também a classe treinamento
previsores_teste = lda.transform(previsores_teste)

from sklearn.ensemble import RandomForestClassifier
classificador = RandomForestClassifier(n_estimators = 40, criterion = 'entropy', random_state = 0)
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)

from sklearn.metrics import accuracy_score
precisao = accuracy_score(classe_teste, previsoes)