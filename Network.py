#Whole network with all layers

from typing import Callable
import numpy as np

from Layer import Layer

class Network:

    def __init__(self) -> None:
        self.layers: list[Layer]    = []
        self.loss                   = None
        self.lossPrime              = None


    def add(self, layer: Layer):
        self.layers.append(layer)

    
    def use(self, loss, lossPrime):
        self.loss       = loss
        self.lossPrime  = lossPrime


    def predict(self, input: np.ndarray):
        samples         = len(input)
        result          = []

        for i in range(samples):
            output = input[i]
            for layer in self.layers:
                output  = layer.forwardPropagation(output)
            
            result.append(output)

        return result
    

    def fit(self, xTrain, yTrain, epochs, learningRate):
        samples         = len(xTrain)

        for i in range(epochs):
            err = 0
            for j in range(samples):
                output  = xTrain[j]
                for layer in self.layers:
                    output  = layer.forwardPropagation(output)


                err    += self.loss(yTrain[j], output)


                error   = self.lossPrime(yTrain[j], output)


                for layer in reversed(self.layers):
                    error   = layer.backwardPropagation(error, learningRate)

            err        /= samples
            print(f"epoch {i+1}/{epochs}     error={error}")


