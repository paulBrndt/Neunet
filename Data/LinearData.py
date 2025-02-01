# Linear Data

import numpy as np

from Data.BaseData import BaseData



class LinearData(BaseData):

    X_STEP_SIZE     = 0.1

    def __init__(self, m: float = 1, t: float = 0, minX: float = 0, maxX: float = 10, forceUpdate: bool = False) -> None:
        """Create a new set of Linear Data, for 1 Linear Function

        Args:
            m (float): angle. Defaults to 1.
            t (float): shift upwards. Defaults to 0.
            minX (float): start of x. Defaults to 0.
            maxX (float): end of x. Defaults to 10.
            forceUpdate (bool): re-calculate Y always or just when fetching?. Defaults to False.
        """
        self._M: float      = m
        self._T: float      = t

        self._X: np.ndarray = np.arange(minX, maxX, self.X_STEP_SIZE)
        self._Y: np.ndarray = self._X

        self.forceUpdate    = forceUpdate

        self.state: int     = 0b0000         # 0 xxxx, 1 T changed, 2 M changed, 3 X changed

        self._generate()


#   ----- Public -----

#   ..... Setters .....

    def setM(self, m: float):
        if self._M == m: return #check if something changed
        self._M     = m
        self.state |= 0b0110

        if self.forceUpdate:
            self.state = self._updateM()
    
    def setT(self, t: float):
        if self._T == t: return #check if something changed
        oldT        = self._T
        self._T     = t
        self.state |= 0b0100

        if self.forceUpdate:
            self.state = self._updateT(oldT)

#   ..... Getters .....

    def getM(self) -> float:
        return self._M
    
    def getT(self) -> float:
        return self._T
    
    def getX(self) -> np.ndarray:
        return self._X
    
    def getY(self) -> np.ndarray:
        if self.state == 0: #y is up to date
            return self._Y
        
        if self.state & 0b0001: #x changed
            self._generate()
        if self.state & 0b0010: #m changed
            self._updateM()
        if self.state & 0b0100: #t changed
            self._updateT()

        return self._Y

    
#   ------ Private -----

#   ...... Update ......

    def _updateT(self, oldT=0):
        dif             = self._T - oldT
        self._Y         = self._Y + dif
        self.state     ^= 0b0100


    def _updateM(self):
        self._Y         = self._M * self._X
        if self._T     != 0:
            self._updateT()
        self.state     ^= 0b0010

    def _generate(self):
        self._updateM()


        

