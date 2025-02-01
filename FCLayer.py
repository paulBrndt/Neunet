#Fully connected layer

import numpy as np

from Layer import Layer

class FCLayer(Layer):
    
    def __init__(self, inputSize: int, ouputSize: int) -> None:
        """Create a new fully connected layer with random weights and bias

        Args:
            inputSize (int): number of neurons as input
            ouputSize (int): number of neurons as output
        """
        self.weights: np.ndarray= np.random.rand(inputSize, ouputSize)
        self.bias: np.ndarray   = np.random.rand(1, ouputSize)



    def forwardPropagation(self, input: np.ndarray) -> np.ndarray:
        self.input      = input
        self.output     = np.dot(self.input, self.weights) + self.bias
        return self.output
    

    def backwardPropagation(self, outputError, learningRate: float):
        inputError      = np.dot(outputError, self.weights.T)
        weightsError    = np.dot(self.input.T, outputError)

        self.weights   -= learningRate * weightsError
        self.bias       = self.bias - (learningRate * outputError)
        return inputError