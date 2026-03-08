import numpy as np

__all__ = ['DataLoader', 'DataPoint', 'DataSet', 'data_point', 'data_set']


class DataLoader:
    def __init__(self, dbcon=None, data_path=None, data_uri=None):
        self._dbcon = dbcon
        self._data_path = data_path
        self._data_uri = data_uri

    def load_from_file(self, path: str):
        pass

    def load_from_query(self, query_str: str):
        pass

    def load_from_uri(self, uri: str):
        pass


class DataPoint:
    def __init__(self, x: np.ndarray, y: float, i: int):
        self._x, self._y, self._i = x, y, i


class DataSet:
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self._X, self._Y, self._n, self._p = X, Y, X.shape[0], X.shape[1]

    def get_data_point(self, t: int) -> DataPoint:
        t = (t - 1) % self._n
        return DataPoint(self._X[t], self._Y[t], t)


# Backward-compatible aliases
data_point = DataPoint
data_set = DataSet
