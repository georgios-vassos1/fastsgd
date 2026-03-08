from abc import ABC, abstractmethod
import numpy as np

__all__ = ['BaseLoss', 'HuberLoss']


class BaseLoss(ABC):
    """Abstract base for M-estimator loss functions.

    All methods take a residual ``u = y - x^T theta`` and a robustness
    ``threshold`` which controls the transition between the quadratic and
    linear regimes. It is not a regularisation penalty.

    Scalar methods (``loss``, ``first_deriv``, etc.) operate on a single float.
    Vectorised ``_vec`` variants operate on numpy arrays and are used for
    batch covariance estimation.
    """
    @abstractmethod
    def loss(self, u: float, threshold: float = 0.0) -> float:
        pass

    @abstractmethod
    def first_deriv(self, u: float, threshold: float = 0.0) -> float:
        pass

    @abstractmethod
    def second_deriv(self, u: float, threshold: float = 0.0) -> float:
        pass

    @abstractmethod
    def third_deriv(self, u: float, threshold: float = 0.0) -> float:
        pass

    def loss_vec(self, u: np.ndarray, threshold: float) -> np.ndarray:
        return np.where(np.abs(u) <= threshold, u ** 2 / 2, threshold * np.abs(u) - threshold ** 2 / 2)

    def first_deriv_vec(self, u: np.ndarray, threshold: float) -> np.ndarray:
        return np.where(np.abs(u) <= threshold, u, threshold * np.sign(u))

    def second_deriv_vec(self, u: np.ndarray, threshold: float) -> np.ndarray:
        return np.where(np.abs(u) <= threshold, 1.0, 0.0)


class HuberLoss(BaseLoss):
    """Huber loss: quadratic for |u| <= threshold, linear beyond.

        rho(u) = u²/2                            if |u| <= threshold
               = threshold|u| - threshold²/2     otherwise

    The influence function psi = rho' is bounded by threshold, giving
    robustness to outliers. threshold=∞ recovers ordinary least squares.
    """
    def loss(self, u: float, threshold: float) -> float:
        if np.abs(u) <= threshold:
            return u ** 2 / 2
        return threshold * np.abs(u) - threshold ** 2 / 2

    def first_deriv(self, u: float, threshold: float) -> float:
        if np.abs(u) <= threshold:
            return u
        return threshold * np.sign(u)

    def second_deriv(self, u: float, threshold: float) -> float:
        return 1.0 if np.abs(u) <= threshold else 0.0

    def third_deriv(self, u: float, threshold: float) -> float:
        return 0.0
