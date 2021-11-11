import Orange

base = Orange.data.Table('risco-credito.csv')
base.domain

cn2_learner = Orange.classification.rules.CN2Learner() #Variável que pega a base de dados e gera as regras
classificador = cn2_learner(base) #São as próprias regras em si

for regras in classificador.rule_list:
    print (regras)

#TESTE
resultado = classificador([['boa', 'alta', 'nenhuma', 'acima_35'], ['ruim', 'alta', 'adequada', '0_15']])
for i in resultado:
    print(base.domain.class_var.values[i])