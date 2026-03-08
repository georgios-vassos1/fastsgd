from abc import ABC, abstractmethod
import numpy as np
from .value import LRvalue

__all__ = ['BaseLR']


class BaseLR(ABC):
    @abstractmethod
    def __call__(self, t: int, grad_t: np.ndarray) -> LRvalue:
        pass
