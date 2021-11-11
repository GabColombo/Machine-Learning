import Orange

base = Orange.data.Table('credit-data.csv')
base.domain

base_dividida = Orange.evaluation.testing.sample(base, n=0.25)
base_treinamento = base_dividida[1]
base_teste = base_dividida[0]
len(base_treinamento)
len(base_teste)

classificador = Orange.classification.MajorityLearner() #Pega a classe que mais aparece e repete para todas as novas entradas
resultado = Orange.evaluation.testing.TestOnTestData(base_treinamento, base_teste, [classificador])
print(Orange.evaluation.CA(resultado))

from collections import Counter
print(Counter(str(d.get_class()) for d in base_teste))

#Base line classifier -> Define uma linha base
##Caso um algoritmo mais elaborado não possua um rendimento superior à linha base, é mais simples e eficiente utilizar o majoritylearner 