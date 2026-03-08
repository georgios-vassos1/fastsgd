import numpy as np
from .base import BaseLR
from .value import LRvalue

__all__ = ['DDimLR', 'ddimLR']


class DDimLR(BaseLR):
    """D-dimensional learning rate generalising AdaGrad, RMSProp, and d-dim.

    The diagonal accumulator updates as:
        I_diag = a * I_diag + b * grad_t^2
    then the learning rate is:
        lr = eta / (I_diag + eps)^c

    Specialisations (set parameters accordingly):
        AdaGrad : a=1,     b=1,       c=0.5, eta=1.0
        RMSProp : a=gamma, b=1-gamma, c=0.5, eta=1.0
        d-dim   : a=0,     b=1,       c=1.0, eta=1.0
    """
    def __init__(self, d: int, eta: float, a: float, b: float, c: float, eps: float):
        self.__d = d
        self.__Idiag = np.ones(d)
        self.__eta = eta
        self.__a = a
        self.__b = b
        self.__c = c
        self.__eps = eps
        self.__v = LRvalue(1, d)

    def __call__(self, t: int, grad_t: np.ndarray) -> LRvalue:
        self.__Idiag = self.__a * self.__Idiag + self.__b * np.power(grad_t, 2)
        mask = np.abs(self.__Idiag) > 1e-8
        self.__v.lr[mask] = self.__eta / np.power(self.__Idiag[mask] + self.__eps, self.__c)
        if np.any(~mask):
            self.__v.lr[~mask] = self.__Idiag[~mask]
        return self.__v


# Backward-compatible alias
ddimLR = DDimLR
