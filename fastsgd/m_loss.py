import numpy as np
from functools import partial

class BaseLoss:
    def __init__(self):
        pass

    def loss(self, u: float, l: float=0.0) -> float: return

    def first_deriv(self, u: float, l: float=0.0) -> float: return

    def second_deriv(self, u: float, l: float=0.0) -> float: return

    def third_deriv(self, u: float, l: float=0.0) -> float: return

    def loss_vec(self, u: np.ndarray, l: float) -> np.ndarray:
        return np.vectorize(partial(self.loss, l=l))(u)

    def first_deriv_vec(self, u: np.ndarray, l: float) -> np.ndarray:
        return np.vectorize(partial(self.first_deriv, l=l))(u)

    def second_deriv_vec(self, u: np.ndarray, l: float) -> np.ndarray:
        return np.vectorize(partial(self.second_deriv, l=l))(u)


class HuberLoss(BaseLoss):
    def __init__(self):
        pass

    def loss(self, u: float, l: float) -> float:
        if np.abs(u) <= l: return np.power(u, 2) / 2
        else: return (l * np.abs(u)) - (np.power(u, 2) / 2)

    def first_deriv(self, u: float, l: float) -> float:
        if np.abs(u) <= l: return u
        else: return l * np.sign(u)

    def second_deriv(self, u: float, l: float) -> float:
        if np.abs(u) <= l: return 1.0
        else: return 0.0

    def third_deriv(self, u: float, l: float) -> float:
        return 0.0

# if __name__=='__main__':
#     hl = HuberLoss()
#     print(hl.loss(10.0, 5.0), hl.first_deriv(10.0, 5.0))
#     print(hl.loss_vec(np.arange(1,11), 5.0), hl.first_deriv_vec(np.arange(1,11), 5.0))
