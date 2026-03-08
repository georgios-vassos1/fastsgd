from abc import ABC, abstractmethod
import numpy as np
from .utils import DataPoint, DataSet

__all__ = ['Model']


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

    ## Fixed-point equation for the implicit update:
    ## ell'(x^T theta + at x^T grad(penalty) + ksi ||x||^2)
    @abstractmethod
    def scale_factor(self, ksi: float, at: float, datum: DataPoint, theta_old: np.ndarray, normx: float) -> float:
        pass

    ## Covariance estimation
    @abstractmethod
    def score_matrix(self, data: DataSet, theta: np.ndarray) -> np.ndarray:
        """Return the (n, p) matrix of per-observation score contributions.

        The score for observation i is the unpenalised gradient contribution:
        the influence function evaluated at (xᵢ, yᵢ, θ). Regularisation
        penalties are excluded because covariance estimation targets the
        population-level estimating equations.
        """
        pass

    @abstractmethod
    def hessian_weights(self, data: DataSet, theta: np.ndarray) -> np.ndarray:
        """Return the (n,) array of per-observation Hessian weights.

        For a GLM this is h′(xᵢᵀθ); for an M-estimator it is ψ′(rᵢ).
        The expected Hessian is then A = (1/n) Xᵀ diag(w) X.
        """
        pass

