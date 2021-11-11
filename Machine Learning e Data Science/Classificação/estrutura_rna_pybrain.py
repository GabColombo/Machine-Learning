from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer, BiasUnit
from pybrain.structure import FullConnection #Classe que serve para fazer as ligações entre as camadas

rede = FeedForwardNetwork()

camadaEntrada = LinearLayer(2) #2 neurônios. Os valores da cama de entrada são lineares pois não serão alterados
camadaOculta = SigmoidLayer(3) #3 neurônios na camada oculta, será utilizada a função Sigmoid para ativação
camadaSaida = SigmoidLayer(1)  #1 neurônio na camada de saída, também será utilizada a função sigmoid
bias1 = BiasUnit() 
bias2 = BiasUnit()

#Adiciona todas as camadas à rede
rede.addModule(camadaEntrada) 
rede.addModule(camadaOculta)
rede.addModule(camadaSaida)
rede.addModule(bias1)
rede.addModule(bias2)

#Faz as conexões entre as camadas
entradaOculta = FullConnection(camadaEntrada, camadaOculta) #Conexão Entrada-Oculta
ocultaSaida = FullConnection(camadaOculta, camadaSaida) #Conexão Oculta-Saída
biasOculta = FullConnection(bias1, camadaOculta) #Conexão Bias2-Oculta
biasSaida = FullConnection(bias2, camadaSaida) #Conexão Bias1-Saída

rede.sortModules() #Constrói a rede neural

print(rede)
print(entradaOculta.params)
print(ocultaSaida.params)
print(biasOculta.params)
print(biasSaida.params)