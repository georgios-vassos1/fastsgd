from abc import ABC, abstractmethod
import numpy as np

__all__ = ['Transfer', 'Identity', 'Exponential', 'Inverse', 'Logistic']


class Transfer(ABC):
    """Abstract base class for GLM transfer (inverse-link) functions.

    Subclasses must implement ``h``, ``first_deriv``, and ``second_deriv``
    as NumPy-compatible callables that accept both scalars and arrays.
    The ``_vec`` variants delegate to these methods; subclasses only need
    to override them if the base implementation is insufficient.
    """

    @abstractmethod
    def h(self, eta: np.ndarray) -> np.ndarray:
        """Apply the transfer function: mu = h(eta)."""
        pass

    @abstractmethod
    def first_deriv(self, eta: np.ndarray) -> np.ndarray:
        """First derivative h'(eta)."""
        pass

    @abstractmethod
    def second_deriv(self, eta: np.ndarray) -> np.ndarray:
        """Second derivative h''(eta)."""
        pass

    def h_vec(self, u: np.ndarray) -> np.ndarray:
        return self.h(u)

    def first_deriv_vec(self, u: np.ndarray) -> np.ndarray:
        return self.first_deriv(u)

    def second_deriv_vec(self, u: np.ndarray) -> np.ndarray:
        return self.second_deriv(u)


class Identity(Transfer):
    def h(self, eta):
        return eta

    def first_deriv(self, eta):
        return np.ones_like(np.asarray(eta, dtype=float))

    def second_deriv(self, eta):
        return np.zeros_like(np.asarray(eta, dtype=float))


class Inverse(Transfer):
    """Inverse (reciprocal) transfer function: h(eta) = -1/eta.

    The negative sign follows the convention in McCullagh & Nelder (1989)
    for the Gamma GLM canonical link, where the linear predictor eta is
    negative so that mu = -1/eta > 0.
    """

    def h(self, eta):
        return -1.0 / eta

    def first_deriv(self, eta):
        eta = np.asarray(eta, dtype=float)
        safe = np.where(eta != 0.0, eta, 1.0)
        return np.where(eta != 0.0, 1.0 / (safe ** 2), 0.0)

    def second_deriv(self, eta):
        eta = np.asarray(eta, dtype=float)
        safe = np.where(eta != 0.0, eta, 1.0)
        return np.where(eta != 0.0, -2.0 / (safe ** 3), 0.0)


class Exponential(Transfer):
    def h(self, eta):
        return np.exp(eta)

    def first_deriv(self, eta):
        return np.exp(eta)

    def second_deriv(self, eta):
        return np.exp(eta)


class Logistic(Transfer):
    def h(self, eta):
        return self._sigmoid(eta)

    def first_deriv(self, eta):
        s = self._sigmoid(eta)
        return s * (1.0 - s)

    def second_deriv(self, eta):
        s = self._sigmoid(eta)
        return 2 * (s ** 3) - 3 * (s ** 2) + s

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))
