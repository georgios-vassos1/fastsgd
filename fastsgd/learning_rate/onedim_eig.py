import numpy as np
from .base import BaseLR
from .value import LRvalue

__all__ = ['OnedimEigLR']


class OnedimEigLR(BaseLR):
    """Eigenvalue-adaptive scalar learning rate.

    Approximates the minimum eigenvalue of the Fisher information matrix via
    the bound min_eigen <= d / trace(Fisher) and uses the gradient norm as a
    proxy for the trace. Reverts to 1/t when the gradient is zero to avoid
    division by zero.
    """
    def __init__(self, d: int):
        self.__d = d
        self.__v = LRvalue(0, 1)

    def __call__(self, t: int, grad_t: np.ndarray) -> LRvalue:
        sum_eigen = np.sum(np.power(grad_t, 2))
        if sum_eigen == 0.0:
            self.__v.lr = 1.0 / t
        else:
            self.__v.lr = 1.0 / ((sum_eigen / self.__d) * t)
        return self.__v
