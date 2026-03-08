import numpy as np
from .utils import DataPoint, DataSet
from .model import *
from .m_loss import HuberLoss

__all__ = ['MModel', 'm_model']


class MModel(Model):
    def __init__(self, loss: str = "huber", l: float = 3.0, lambda1: float = 0.0, lambda2: float = 0.0):
        super().__init__("M estimator", lambda1, lambda2)
        self._loss_name = loss
        if loss.lower() == 'huber':
            self._l = l
            self._lossobj = HuberLoss()
        else:
            raise ValueError(f"Unsupported loss '{loss}'. Currently only 'huber' is supported.")

    @property
    def loss(self):
        return self._loss_name

    @property
    def l(self):
        return self._l

    def _gradient_at_point(self, datum: DataPoint, theta_old: np.ndarray) -> np.ndarray:
        return (
            self._lossobj.first_deriv(datum._y - (datum._x @ theta_old), self._l) * datum._x
            - self.gradient_penalty(theta_old)
        )

    def scale_factor(self, ksi: float, at: float, datum: DataPoint, theta_old: np.ndarray, normx: float) -> float:
        return self._lossobj.first_deriv(
            datum._y - (datum._x @ theta_old) - at * (self.gradient_penalty(theta_old) @ datum._x) + ksi * normx,
            self._l,
        )

    def scale_factor_first_deriv(self, ksi: float, at: float, datum: DataPoint, theta_old: np.ndarray, normx: float) -> float:
        return self._lossobj.second_deriv(
            datum._y - (datum._x @ theta_old) - at * (self.gradient_penalty(theta_old) @ datum._x) + ksi * normx,
            self._l,
        ) * normx

    def scale_factor_second_deriv(self, ksi: float, at: float, datum: DataPoint, theta_old: np.ndarray, normx: float) -> float:
        return self._lossobj.third_deriv(
            datum._y - (datum._x @ theta_old) - at * (self.gradient_penalty(theta_old) @ datum._x) + ksi * normx,
            self._l,
        ) * normx * normx


# Backward-compatible alias
m_model = MModel
