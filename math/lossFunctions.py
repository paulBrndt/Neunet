import numpy as np

def mse(true, pred):
    np.mean(np.power(true-pred, 1))

def msePrime(true, pred):
    return 2*(pred-true)/true.size