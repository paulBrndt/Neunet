import numpy as np

#output layer often different one

def binaryStep(x: float, thershold: float = 0.5) -> float:
    return 1 if x > thershold else 0

def linearActivation(x: float) -> float:
    return x

def sigmoid(x: float) -> float:
    #every number as propability from 0 to 1
    return 1 / (1 + np.exp(-x))

def tanh(x: float) -> float:
    #every number as propability from -1 to 1
    ex  = np.exp(x)
    emx = np.exp(-x)
    return (ex - emx) / (ex + emx)


def relu(x: float) -> float:
    #output x as long as positive, no backpropagation when negative as danger
    max(0, x)

def leakyRelu(x: float,) -> float:
    max(0.1*x, x)

def parametricRelu(x: float, a: float) -> float:
    max(a*x, x)

def elu(x: float, a: float) -> float:
    return x if (x >= 0) else (a*(np.exp(x) -1))

