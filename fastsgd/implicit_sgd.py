import numpy as np
from scipy.optimize import brentq
from .utils import DataPoint, DataSet
from .sgd import *
from .model import *

__all__ = ['ImplicitSGD', 'ImplicitFn']


# Root finding function for the fixed-point equation:
#   ksi - at * scale_factor(ksi) = 0
class ImplicitFn:
    def __init__(self, model: Model, a: float, d: DataPoint, theta_old: np.ndarray, n: float):
        self._model = model
        self._at = a
        self._datum = d
        self._theta_old = theta_old
        self._normx = n

    def __call__(self, ksi: float) -> float:
        return ksi - self._at * self._model.scale_factor(ksi, self._at, self._datum, self._theta_old, self._normx)


class ImplicitSGD(SGD):
    def __init__(self, n: int, p: int, timer, **details):
        super().__init__(n, p, timer, method="Implicit SGD", **details)

    def update(self, t: int, theta_old: np.ndarray, data: DataSet, model: Model, good_gradient: bool) -> np.ndarray:
        datum = data.get_data_point(t)
        grad_t = model._gradient_at_point(datum, theta_old)
        if not np.all(np.isfinite(grad_t)):
            self._good_gradient = False
        at = self._learning_rate(t, grad_t).mean()

        normx = np.linalg.norm(datum._x)

        r = at * model.scale_factor(0, at, datum, theta_old, normx)
        lower, upper = (r, 0) if r < 0 else (0, r)

        if lower != upper:
            ksi = brentq(ImplicitFn(model, at, datum, theta_old, normx), lower, upper, maxiter=100)
        else:
            ksi = lower

        return theta_old + ksi * datum._x - at * model.gradient_penalty(theta_old)
