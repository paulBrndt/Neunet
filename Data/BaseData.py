import numpy as np



class BaseData:
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self._X: np.ndarray = x
        self._Y: np.ndarray = y

    def normalize(self):
        raise NotImplementedError
