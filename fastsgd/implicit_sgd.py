import numpy as np
from scipy.optimize import brentq
from .utils import DataPoint, DataSet
from .sgd import SGD
from .model import Model

__all__ = ['ImplicitSGD', 'ImplicitFn']


# Root finding function for the fixed-point equation:
#   ksi - at * scale_factor(ksi) = 0
class ImplicitFn:
    """Callable for the implicit fixed-point equation f(ksi) = 0, where

        f(ksi) = ksi - a_t * ell'(x_t^T theta + ksi * ||x_t||²)

    Passed to scipy.optimize.brentq to find the scalar ksi that satisfies the
    implicit SGD update equation at iteration t.
    """
    def __init__(self, model: Model, a: float, d: DataPoint, theta_old: np.ndarray, n: float):
        self._model = model
        self._at = a
        self._datum = d
        self._theta_old = theta_old
        self._normx = n

    def __call__(self, ksi: float) -> float:
        return ksi - self._at * self._model.scale_factor(ksi, self._at, self._datum, self._theta_old, self._normx)


class ImplicitSGD(SGD):
    """Implicit (proximal) SGD from Toulis & Airoldi (2017).

    Achieves the Cramér-Rao efficiency bound for GLMs; see the paper for
    details. Each update solves the scalar fixed-point equation

        ksi - a_t * ell'(x_t^T theta_old + ksi * ||x_t||²) = 0

    via Brent's method, bracketed on [0, a_t * ell'(x_t^T theta_old)].

    Note: the bracket construction assumes scale_factor(0) and the root have
    opposite signs, which holds for GLMs. For M-estimators with saturating
    influence functions (e.g. Huber) the bracket may be invalid; use
    ExplicitSGD in that case.
    """
    def __init__(self, n: int, p: int, timer, **details):
        super().__init__(n, p, timer, method="Implicit SGD", **details)

    def update(self, t: int, theta_old: np.ndarray, data: DataSet, model: Model, good_gradient: bool) -> np.ndarray:
        """Compute one implicit SGD step and return the updated parameter vector.

        The bracket [lower, upper] is constructed from the value of the
        fixed-point function at ksi=0: r = a_t * ell'(x_t^T theta_old).
        If r < 0 the root lies in [r, 0]; if r > 0 it lies in [0, r].
        When r == 0 the update is a no-op (ksi = 0).
        """
        datum = data.get_data_point(t)
        grad_t = model._gradient_at_point(datum, theta_old)
        if not np.all(np.isfinite(grad_t)):
            self._good_gradient = False
        at = self._learning_rate(t, grad_t).mean()

        normx = np.linalg.norm(datum._x)

        r = at * model.scale_factor(0, at, datum, theta_old, normx)
        lower, upper = (r, 0) if r < 0 else (0, r)

        if lower != upper:
            try:
                ksi = brentq(ImplicitFn(model, at, datum, theta_old, normx), lower, upper, maxiter=100)
            except ValueError as exc:
                raise ValueError(
                    f"ImplicitSGD: brentq could not bracket a root at iteration {t} "
                    f"(bracket=[{lower:.4g}, {upper:.4g}]). "
                    "This typically occurs for M-estimators with saturating influence "
                    "functions (e.g. Huber loss). Use ExplicitSGD for such models."
                ) from exc
        else:
            ksi = lower

        return theta_old + ksi * datum._x - at * model.gradient_penalty(theta_old)
