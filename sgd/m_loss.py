import numpy as np

class BaseLoss:
    def __init__(self):
        pass

    def loss(self, u: float, l: float=None) -> float:
        return 0.0

    def first_deriv(self, u: float, l: float=None) -> float:
        return 0.0

    def second_deriv(self, u: float, l: float=None) -> float:
        return 0.0

    def third_deriv(self, u: float, l: float=None) -> float:
        return 0.0

    def loss_vec(self, u: pd.ndarray, l: float=None) -> np.ndarray:
        return np.vectorize(self.loss)(u, l)

    def first_deriv_vec(self, u: np.ndarray, l: float=None) -> np.ndarray:
        return np.vectorize(self.first_deriv)(u, l)

    def second_deriv_vec(self, u: np.ndarray, l: float=None) -> np.ndarray:
        return np.vectorize(self.second_deriv)(u, l)


class HuberLoss(BaseLoss):
    def __init__(self):
        pass

    def loss(self, u: float, l: float) -> float:
        if np.abs(u) <= l:
            return np.power(u, 2) >> 1
        else:
            return (l * np.abs(u)) - (np.power(u, 2) >> 1)

    def first_deriv(self, u: float, l: float) -> float:
        if np.abs(u) <= l:
            return u
        else:
            return l * np.sign(u)

    def second_deriv(self, u: float, l: float) -> float:
        if np.abs(u) <= l:
            return 1.0
        else:
            return 0.0

    def third_deriv(self, u: float, l: float) -> float:
        return 0.0
