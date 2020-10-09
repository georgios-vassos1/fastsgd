import numpy as np
from .base import BaseLR
from .value import LRvalue

class ddimLR(BaseLR):
    # d-dimensional learning rate, which includes as special cases polular learning rates:
    # adagrad: a, b, c, eta, eps = 1, 1, 1/2, 1, 1e-6
    # d-dim: a, b, c, eta, eps = 0, 1, 1, 1, 1e-6
    # rmsprop: a, b, c, eta, eps = gamma, 1-gamma, 1/2, 1, 1e-6
    #
    # d     dimension of learning rate
    # eta   scale factor in numberator
    # a     weights of perivous gradient
    # b     weight of new gradient
    # c     exponential power
    # eps   small value to prevent division by zero
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
        self.__v.lr[mask] = self.__eta / np.power(self.__Idiag + self.__eps, self.__c)
        if len(self.__v.lr[~mask]): self.__v.lr[~mask] = self.__Idiag[~mask]
        return self.__v
