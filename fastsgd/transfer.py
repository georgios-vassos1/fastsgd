import numpy as np

__all__ = ['Transfer', 'Identity', 'Exponential', 'Inverse', 'Logistic']


class Transfer:
    """Base class for GLM link (transfer) functions.

    Subclasses set self.h, self.g, self.first_deriv, self.second_deriv as
    NumPy-compatible callables. The _vec variants call these directly; subclasses
    may override them when the scalar form requires conditional logic.
    """
    def __init__(self, h):
        self.h = h

    def h_vec(self, u: np.ndarray) -> np.ndarray:
        return self.h(u)

    def first_deriv_vec(self, u: np.ndarray) -> np.ndarray:
        return self.first_deriv(u)

    def second_deriv_vec(self, u: np.ndarray) -> np.ndarray:
        return self.second_deriv(u)


class Identity(Transfer):
    def __init__(self):
        super().__init__(lambda x: x)
        self.first_deriv = lambda x: np.ones_like(np.asarray(x, dtype=float))
        self.second_deriv = lambda x: np.zeros_like(np.asarray(x, dtype=float))

    def __str__(self):
        return "This is an instance of the Identity class."


class Inverse(Transfer):
    def __init__(self):
        super().__init__(lambda x: -1.0 / x)
        self.first_deriv = lambda x: 1.0 / (x ** 2) if x != 0.0 else 0.0
        self.second_deriv = lambda x: -2.0 / (x ** 3) if x != 0.0 else 0.0

    def first_deriv_vec(self, u: np.ndarray) -> np.ndarray:
        safe = np.where(u != 0.0, u, 1.0)
        return np.where(u != 0.0, 1.0 / (safe ** 2), 0.0)

    def second_deriv_vec(self, u: np.ndarray) -> np.ndarray:
        safe = np.where(u != 0.0, u, 1.0)
        return np.where(u != 0.0, -2.0 / (safe ** 3), 0.0)


class Exponential(Transfer):
    def __init__(self):
        super().__init__(np.exp)
        self.first_deriv = np.exp
        self.second_deriv = np.exp


class Logistic(Transfer):
    def __init__(self):
        super().__init__(self._sigmoid)
        self.first_deriv = lambda x: self._sigmoid(x) * (1.0 - self._sigmoid(x))
        self.second_deriv = lambda x: 2 * (self._sigmoid(x) ** 3) - 3 * (self._sigmoid(x) ** 2) + self._sigmoid(x)

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))
