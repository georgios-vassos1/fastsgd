import numpy as np
from .utils import DataPoint, DataSet
from .model import *
from .family import *
from .transfer import *

__all__ = ['GLM', 'glm']


class GLM(Model):
    def __init__(self, family: str, transfer: str, lambda1: float = 0.0, lambda2: float = 0.0):
        super().__init__("Generalized Linear Model", lambda1, lambda2)
        self._family = family
        self._transfer = transfer
        try:
            self._family_instance = {
                'gaussian':  Gaussian(),
                'poisson':   Poisson(),
                'gamma':     Gamma(),
                'binomial':  Binomial(),
            }[self._family]
        except KeyError:
            raise ValueError(f"Unknown family '{family}'. Choose from: gaussian, poisson, gamma, binomial.")
        try:
            self._transfer_instance = {
                'identity':    Identity(),
                'exponential': Exponential(),
                'inverse':     Inverse(),
                'logistic':    Logistic(),
            }[self._transfer]
        except KeyError:
            raise ValueError(f"Unknown transfer '{transfer}'. Choose from: identity, exponential, inverse, logistic.")

    def _gradient_at_point(self, datum: DataPoint, theta_old: np.ndarray) -> np.ndarray:
        return ((datum._y - self._transfer_instance.h(datum._x @ theta_old)) * datum._x) - self.gradient_penalty(theta_old)

    def scale_factor(self, ksi: float, at: float, datum: DataPoint, theta_old: np.ndarray, normx: float) -> float:
        return datum._y - self._transfer_instance.h(
            (theta_old @ datum._x) - at * (self.gradient_penalty(theta_old) @ datum._x) + ksi * normx
        )

    def score_matrix(self, data: DataSet, theta: np.ndarray) -> np.ndarray:
        eta = data._X @ theta
        resid = data._Y - self._transfer_instance.h_vec(eta)
        return resid[:, None] * data._X

    def hessian_weights(self, data: DataSet, theta: np.ndarray) -> np.ndarray:
        eta = data._X @ theta
        return np.asarray(self._transfer_instance.first_deriv_vec(eta), dtype=float).ravel()


# Backward-compatible alias
glm = GLM
