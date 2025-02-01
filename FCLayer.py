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
        self.weights    = np.random.rand(inputSize, ouputSize)
        self.bias       = np.random.rand(1, ouputSize)