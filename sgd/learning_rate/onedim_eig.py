import numpy as np
from .base import BaseLR
from .value import LRvalue

class OnedimEigLR(BaseLR):
    def __init__(self, d: int):
        self.__d = d
        self.__v = LRvalue(0, 1)

    def __call__(self, t: int, grad_t: np.ndarray) -> LRvalue:
        sum_eigen = np.sum(np.power(grad_t, 2))
        ## based on the bound of min_eigen <= d / trace(Fisher_matrix)
        self.__v.lr = 1.0 / ((sum_eigen / self.__d) * t)
        return self.__v
