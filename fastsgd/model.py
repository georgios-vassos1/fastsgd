from abc import ABC, abstractmethod
import numpy as np
from .utils import *

__all__ = ['Model', 'model']


class Model(ABC):
    def __init__(self, name: str, lambda1: float, lambda2: float):
        self._name = name
        self._lambda1 = lambda1
        self._lambda2 = lambda2

    def gradient(self, t: int, theta_old: np.ndarray, data: DataSet) -> np.ndarray:
        datum = data.get_data_point(t)
        return self._gradient_at_point(datum, theta_old)

    @abstractmethod
    def _gradient_at_point(self, datum: DataPoint, theta_old: np.ndarray) -> np.ndarray:
        pass

    def gradient_penalty(self, theta: np.ndarray) -> np.ndarray:
        return self._lambda1 * np.sign(theta) + self._lambda2 * theta

    ## Functions for the implicit update
    ## ell'(x^T theta + at x^T grad(penalty) + ksi ||x||^2)
    @abstractmethod
    def scale_factor(self, ksi: float, at: float, datum: DataPoint, theta_old: np.ndarray, normx: float) -> float:
        pass

    ## d/d(ksi) ell'
    def scale_factor_first_deriv(self, ksi: float, at: float, datum: DataPoint, theta_old: np.ndarray, normx: float) -> float:
        pass

    ## d^2/d(ksi)^2 ell'
    def scale_factor_second_deriv(self, ksi: float, at: float, datum: DataPoint, theta_old: np.ndarray, normx: float) -> float:
        pass


# Backward-compatible alias
model = Model
