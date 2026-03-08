import numpy as np

__all__ = ['LRvalue']


class LRvalue:
    def __init__(self, seq_type: int, d: int = None):
        self.__type = seq_type
        if self.__type == 0:
            self.__lr = np.array([1.0])
        elif self.__type == 1:
            self.__lr = np.ones(d)
        elif self.__type == 2:
            self.__lr = np.eye(d)
        else:
            raise ValueError("seq_type must be 0 (scalar), 1 (vector), or 2 (matrix).")

    @property
    def type(self) -> int:
        return self.__type

    @property
    def lr(self) -> np.ndarray:
        return self.__lr

    @lr.setter
    def lr(self, seq: np.ndarray):
        if isinstance(seq, (float, int)):
            seq = np.array([seq])
        self.__lr = seq
        self.__update_type()

    def __update_type(self):
        if len(self.__lr) == 1:
            self.__type = 0
        else:
            self.__type = len(self.__lr.shape)

    def at(self, i: int = None, j: int = None) -> float:
        return {0: self.__lr[0], 1: self.__lr[i], 2: self.__lr[i, j]}[self.__type]

    def mean(self) -> float:
        return np.mean(self.__lr)

    def __mul__(self, rhs: np.ndarray):
        if self.__type != 2:
            return self.__lr * rhs
        return self.__lr @ rhs

    def __lt__(self, threshold: float) -> bool:
        if isinstance(self.__lr[0], np.ndarray):
            return np.all(np.diagonal(self.__lr) < threshold)
        return np.all(self.__lr < threshold)

    def __gt__(self, threshold: float) -> bool:
        if isinstance(self.__lr[0], np.ndarray):
            return np.all(np.diagonal(self.__lr) > threshold)
        return np.all(self.__lr > threshold)
