import numpy as np

__all__ = ['LRvalue']


class LRvalue:
    """Thin wrapper around a numpy learning-rate array.

    seq_type 0 — scalar (shape (1,))
    seq_type 1 — per-parameter vector (shape (d,))
    """

    def __init__(self, seq_type: int, d: int = None):
        self.__type = seq_type
        if self.__type == 0:
            self.__lr = np.array([1.0])
        elif self.__type == 1:
            self.__lr = np.ones(d)
        else:
            raise ValueError("seq_type must be 0 (scalar) or 1 (vector).")

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
        self.__type = 0 if len(self.__lr) == 1 else 1

    def at(self, i: int) -> float:
        return self.__lr[0] if self.__type == 0 else self.__lr[i]

    def mean(self) -> float:
        return np.mean(self.__lr)

    def __mul__(self, rhs: np.ndarray):
        return self.__lr * rhs

    def __lt__(self, threshold: float) -> bool:
        return np.all(self.__lr < threshold)

    def __gt__(self, threshold: float) -> bool:
        return np.all(self.__lr > threshold)
