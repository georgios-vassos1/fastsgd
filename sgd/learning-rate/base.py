from value import LRvalue
import numpy as np

class BaseLR:
    def __init__(self):
        pass

    def __call__(self, t: int, grad_t: np.ndarray) -> LRvalue:
        return t

