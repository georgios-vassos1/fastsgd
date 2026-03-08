import numpy as np
from .utils import DataPoint, DataSet
from .model import Model
from .loss import HuberLoss

__all__ = ['MModel']


class MModel(Model):
    def __init__(self, loss: str = "huber", threshold: float = 3.0, lambda1: float = 0.0, lambda2: float = 0.0):
        super().__init__("M estimator", lambda1, lambda2)
        self._loss_name = loss
        if loss.lower() == 'huber':
            self._threshold = threshold
            self._lossobj = HuberLoss()
        else:
            raise ValueError(f"Unsupported loss '{loss}'. Currently only 'huber' is supported.")

    @property
    def loss(self):
        return self._loss_name

    @property
    def threshold(self):
        return self._threshold

    def __repr__(self) -> str:
        return f"MModel(loss='{self._loss_name}', threshold={self._threshold})"

    def _gradient_at_point(self, datum: DataPoint, theta_old: np.ndarray) -> np.ndarray:
        return (
            self._lossobj.first_deriv(datum._y - (datum._x @ theta_old), self._threshold) * datum._x
            - self.gradient_penalty(theta_old)
        )

    def score_matrix(self, data: DataSet, theta: np.ndarray) -> np.ndarray:
        resid = data._Y - data._X @ theta
        psi = self._lossobj.first_deriv_vec(resid, self._threshold)
        return psi[:, None] * data._X

    def hessian_weights(self, data: DataSet, theta: np.ndarray) -> np.ndarray:
        resid = data._Y - data._X @ theta
        return self._lossobj.second_deriv_vec(resid, self._threshold)

    def scale_factor(self, ksi: float, at: float, datum: DataPoint, theta_old: np.ndarray, normx: float) -> float:
        return self._lossobj.first_deriv(
            datum._y - (datum._x @ theta_old) - at * (self.gradient_penalty(theta_old) @ datum._x) + ksi * normx,
            self._threshold,
        )
