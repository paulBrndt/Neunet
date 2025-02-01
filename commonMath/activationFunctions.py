import numpy as np

#output layer often different one

def binaryStep(x: np.ndarray, thershold: float = 0.5) -> np.ndarray:
    return 1 if x > thershold else 0

def linearActivation(x: np.ndarray) -> np.ndarray:
    return x

def sigmoid(x: np.ndarray) -> np.ndarray:
    #every number as propability from 0 to 1
    return 1 / (1 + np.exp(-x))

def tanh(x: np.ndarray) -> np.ndarray:
    #every number as propability from -1 to 1
    return np.tanh(x)

def tanhPrime(x: np.ndarray) -> np.ndarray:
    return 1-tanh(x)**2


def relu(x: np.ndarray) -> np.ndarray:
    #output x as long as positive, no backpropagation when negative as danger
    max(0, x)

def leakyRelu(x: np.ndarray) -> np.ndarray:
    max(0.1*x, x)

def parametricRelu(x: np.ndarray, a: float) -> np.ndarray:
    max(a*x, x)

#FIXME
def elu(x: np.ndarray, a: float=0.1) -> np.ndarray:
    return x if (x >= 0) else (a*(np.exp(x) -1))

