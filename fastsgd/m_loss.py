from abc import ABC, abstractmethod
import numpy as np

__all__ = ['BaseLoss', 'HuberLoss']


class BaseLoss(ABC):
    """Abstract base for M-estimator loss functions.

    All methods take a residual ``u = y - x^T theta`` and a threshold ``l``
    which controls the transition between the quadratic and linear regimes.
    ``l`` is a robustness threshold, not a regularisation penalty.

    Scalar methods (``loss``, ``first_deriv``, etc.) operate on a single float.
    Vectorised ``_vec`` variants operate on numpy arrays and are used for
    batch covariance estimation.
    """
    @abstractmethod
    def loss(self, u: float, l: float = 0.0) -> float:
        pass

    @abstractmethod
    def first_deriv(self, u: float, l: float = 0.0) -> float:
        pass

    @abstractmethod
    def second_deriv(self, u: float, l: float = 0.0) -> float:
        pass

    @abstractmethod
    def third_deriv(self, u: float, l: float = 0.0) -> float:
        pass

    def loss_vec(self, u: np.ndarray, l: float) -> np.ndarray:
        return np.where(np.abs(u) <= l, u ** 2 / 2, l * np.abs(u) - l ** 2 / 2)

    def first_deriv_vec(self, u: np.ndarray, l: float) -> np.ndarray:
        return np.where(np.abs(u) <= l, u, l * np.sign(u))

    def second_deriv_vec(self, u: np.ndarray, l: float) -> np.ndarray:
        return np.where(np.abs(u) <= l, 1.0, 0.0)


class HuberLoss(BaseLoss):
    """Huber loss: quadratic for |u| <= l, linear beyond.

        rho(u) = u²/2            if |u| <= l
               = l|u| - l²/2     otherwise

    The influence function psi = rho' is bounded by l, giving robustness to
    outliers. l=∞ recovers ordinary least squares.
    """
    def loss(self, u: float, l: float) -> float:
        if np.abs(u) <= l:
            return u ** 2 / 2
        return l * np.abs(u) - l ** 2 / 2

    def first_deriv(self, u: float, l: float) -> float:
        if np.abs(u) <= l:
            return u
        return l * np.sign(u)

    def second_deriv(self, u: float, l: float) -> float:
        return 1.0 if np.abs(u) <= l else 0.0

    def third_deriv(self, u: float, l: float) -> float:
        return 0.0
