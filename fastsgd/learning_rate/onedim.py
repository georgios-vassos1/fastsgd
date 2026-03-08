import numpy as np
from .base import BaseLR
from .value import LRvalue

__all__ = ['OnedimLR']


class OnedimLR(BaseLR):
    """Scalar polynomial-decay learning rate.

    Computes: a_t = scale * gamma * (1 + alpha * gamma * t)^{-c}

    Parameters
    ----------
    scale : float
        Multiplicative scale applied to every step size.
    gamma : float
        Initial learning rate (step size at t=0 is scale * gamma).
    alpha : float
        Controls how quickly the rate decays; larger alpha → faster decay.
    c : float
        Decay exponent. Must satisfy 0.5 < c <= 1 for Robbins-Monro
        convergence guarantees (sum a_t = ∞, sum a_t² < ∞).
    """
    def __init__(self, scale: float, gamma: float, alpha: float, c: float):
        self.__scale = scale
        self.__gamma = gamma
        self.__alpha = alpha
        self.__c = c
        self.__v = LRvalue(0, 1)

    def __call__(self, t: int, grad_t: np.ndarray) -> LRvalue:
        self.__v.lr = self.__scale * self.__gamma * np.power(1 + self.__alpha * self.__gamma * t, -self.__c)
        return self.__v
