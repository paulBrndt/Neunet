import numpy as np

def mse(true, pred) -> float:
    return np.mean(np.power(true-pred, 2))

def msePrime(true, pred) -> float:
    return 2*(pred-true)/true.size