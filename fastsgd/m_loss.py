from abc import ABC, abstractmethod
import numpy as np

__all__ = ['BaseLoss', 'HuberLoss']


class BaseLoss(ABC):
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
