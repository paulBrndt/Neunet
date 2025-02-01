import numpy as np


#TODO: rename to base layer class

class Layer:

    def __init__(self) -> None:
        self.input: np.ndarray  = None
        self.output: np.ndarray = None

    def forwardPropagation(self, input: np.ndarray):
        raise NotImplementedError
    
    def backwardPropagation(self, outputError, learningRate):
        raise NotImplementedError