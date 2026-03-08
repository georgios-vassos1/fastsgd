import numpy as np

__all__ = ['DataPoint', 'DataSet', 'data_point', 'data_set']


class DataPoint:
    def __init__(self, x: np.ndarray, y: float, i: int):
        self._x, self._y, self._i = x, y, i


class DataSet:
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be a 2-D array, got shape {X.shape}")
        if Y.ndim != 1:
            raise ValueError(f"Y must be a 1-D array, got shape {Y.shape}")
        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"X and Y must have the same number of observations, "
                f"got X.shape[0]={X.shape[0]} and len(Y)={Y.shape[0]}"
            )
        self._X, self._Y, self._n, self._p = X, Y, X.shape[0], X.shape[1]

    def get_data_point(self, t: int) -> DataPoint:
        t = (t - 1) % self._n
        return DataPoint(self._X[t], self._Y[t], t)


# Backward-compatible aliases
data_point = DataPoint
data_set = DataSet
