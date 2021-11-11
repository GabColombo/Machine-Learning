import numpy as np


def StepFunction(soma): # Retorna 1 para valores >= 1 e 0 para valores < 1 (Utilizada somente para problemas linearmente separáveis)
    if (soma >= 1):
        return 1
    return 0

def SigmoidFunction(soma): # Retorna valores entre 0 e 1 (0% e 100% - Classificação com 2 classes)
    return 1 / (1 + np.exp(-soma))

def TanFunction(soma): # Retorna valores negativos e positivos (Classificação)
    return (np.exp(soma) - np.exp(-soma)) / (np.exp(soma) + np.exp(-soma))

def ReluFunction(soma): # Retorna somente os valores positivos (Convulacionais e redes com várias camadas)
    if (soma >= 0):
        return soma
    return 0

def LinearFunction(soma): # Retorna o próprio valor (Utilizada para regressão)
    return soma

def SoftmaxFunction(x): # Retorna uma probabilidade para cada uma das classes (Classificação com várias classes)
    ex = np.exp(x)
    return ex / ex.sum()

teste = TanFunction(2.1)

valores = [5.0, 2.0, 1.3]
print(SoftmaxFunction(valores))