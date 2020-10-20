import numpy as np
from .base import BaseLR
from .value import LRvalue

class OnedimLR(BaseLR):
    def __init__(self, scale: float, gamma: float, alpha: float, c: float):
        self.__scale = scale
        self.__gamma = gamma
        self.__alpha = alpha
        self.__c = c
        self.__v = LRvalue(0, 1)

    def __call__(self, t: int, grad_t: np.ndarray) -> LRvalue:
        self.__v.lr = self.__scale * self.__gamma * np.power(1 + self.__alpha * self.__gamma * t, -self.__c)
        return self.__v
