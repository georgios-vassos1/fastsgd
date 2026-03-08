import numpy as np
from .utils import DataSet
from .sgd import *
from .model import *

__all__ = ['ExplicitSGD']


class ExplicitSGD(SGD):
    def __init__(self, n: int, p: int, timer, **kwargs):
        super().__init__(n, p, timer, method="Explicit SGD", **kwargs)

    def update(self, t: int, theta_old: np.ndarray, data: DataSet, model: Model, good_gradient: bool) -> np.ndarray:
        grad_t = model.gradient(t, theta_old, data)
        if not np.all(np.isfinite(grad_t)):
            self._good_gradient = False
        at = self._learning_rate(t, grad_t).mean()
        return theta_old + at * grad_t
