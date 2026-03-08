import numpy as np
from .utils import DataPoint, DataSet
from .model import Model
from .family import Gaussian, Poisson, Gamma, Binomial
from .transfer import Identity, Exponential, Inverse, Logistic

__all__ = ['GLM']


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

    def __repr__(self) -> str:
        return f"GLM(family='{self._family}', transfer='{self._transfer}')"

    def _gradient_at_point(self, datum: DataPoint, theta_old: np.ndarray) -> np.ndarray:
        eta = datum._x @ theta_old
        mu = self._transfer_instance.h(eta)
        h_prime = self._transfer_instance.first_deriv(eta)
        v_mu = self._family_instance.variance(mu)
        return (datum._y - mu) / v_mu * h_prime * datum._x - self.gradient_penalty(theta_old)

    def scale_factor(self, ksi: float, at: float, datum: DataPoint, theta_old: np.ndarray, normx: float) -> float:
        # Fixed-point equation for the quasi-likelihood estimating equations
        # (y - h(eta)) * x = 0.  Valid for canonical links; for non-canonical
        # links the proper MLE scale_factor would also include V(mu) and h'(eta).
        return datum._y - self._transfer_instance.h(
            (theta_old @ datum._x) - at * (self.gradient_penalty(theta_old) @ datum._x) + ksi * normx
        )

    def score_matrix(self, data: DataSet, theta: np.ndarray) -> np.ndarray:
        eta = data._X @ theta
        mu = self._transfer_instance.h_vec(eta)
        h_prime = np.asarray(self._transfer_instance.first_deriv_vec(eta), dtype=float).ravel()
        v_mu = np.asarray(self._family_instance.variance(mu), dtype=float).ravel()
        return ((data._Y - mu) / v_mu * h_prime)[:, None] * data._X

    def hessian_weights(self, data: DataSet, theta: np.ndarray) -> np.ndarray:
        eta = data._X @ theta
        mu = self._transfer_instance.h_vec(eta)
        h_prime = np.asarray(self._transfer_instance.first_deriv_vec(eta), dtype=float).ravel()
        v_mu = np.asarray(self._family_instance.variance(mu), dtype=float).ravel()
        return h_prime ** 2 / v_mu

