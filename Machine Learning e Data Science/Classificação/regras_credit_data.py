import Orange

base = Orange.data.Table('credit-data.csv')
base.domain

base_dividida = Orange.evaluation.testing.sample(base, n=0.25) #Divide a base em 2 tabelas, com 75% e 25% dos dados
base_treinamento = base_dividida[1]
base_teste = base_dividida[0]

len(base_treinamento)
len(base_teste)

cn2_learner = Orange.classification.rules.CN2Learner()
classificador = cn2_learner(base_treinamento)

for regras in classificador.rule_list:
    print(regras)

resultado = Orange.evaluation.testing.TestOnTestData(base_treinamento, base_teste, [classificador]) #Recebe a base de dados de treinamento, a base de dados de teste e as regras e realiza os testes
print(Orange.evaluation.CA(resultado))