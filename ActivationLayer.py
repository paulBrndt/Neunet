# non linear learning layer

import numpy as np
from typing import Callable

from Layer import Layer

class ActivationLayer(Layer):

    def __init__(self, activation: Callable, activationPrime: Callable) -> None:
        self.activation     = activation
        self.activationPrime = activationPrime

    def forwardPropagation(self, input: np.ndarray):
        self.input          = input
        self.output         = self.activation(self.input)
        return                self.output
    
    def backwardPropagation(self, outputError, learningRate):
        return self.activationPrime(self.input) * outputError

