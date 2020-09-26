import numpy as np
from .value import LRvalue

class BaseLR:
    def __init__(self):
        pass

    def __call__(self, t: int, grad_t: np.ndarray) -> LRvalue:
        pass

