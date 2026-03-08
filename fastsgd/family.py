from abc import ABC, abstractmethod
import numpy as np

__all__ = [
    'Family', 'Gaussian', 'Poisson', 'Gamma', 'Binomial',
    'Inverse_Gaussian', 'QuasiPoisson', 'QuasiBinomial', 'Quasi',
]


class Family(ABC):
    @abstractmethod
    def deviance(self, y: np.ndarray, mu: np.ndarray, wt: np.ndarray) -> float:
        pass


class Gaussian(Family):
    def __init__(self):
        self.variance = lambda x: 1.0

    def deviance(self, y, mu, wt):
        return np.sum(wt * ((y - mu) ** 2))

    def __str__(self):
        return "This is an instance of the Gaussian class."


class Poisson(Family):
    def __init__(self):
        self.variance = lambda x: x

    def deviance(self, y, mu, wt):
        r, flag = np.zeros(len(y)), y > 0.0
        r[~flag] = mu[~flag] * wt[~flag]
        r[flag] = wt[flag] * ((y[flag] * np.log(y[flag] / mu[flag])) - (y[flag] - mu[flag]))
        return np.sum(2.0 * r)


class Gamma(Family):
    def __init__(self):
        self.variance = lambda x: x ** 2

    def deviance(self, y, mu, wt):
        r, flag = np.zeros(len(y)), y != 0.0
        r[~flag] = mu[~flag] * wt[~flag]
        r[flag] = wt[flag] * (np.log(y[flag] / mu[flag]) - (y[flag] - mu[flag]) / mu[flag])
        return np.sum(-2.0 * r)


class Binomial(Family):
    def __init__(self):
        self.variance = lambda x: x * (1.0 - x)

    def deviance(self, y: np.ndarray, mu: np.ndarray, wt: np.ndarray) -> float:
        r = np.zeros(len(y))
        r[y == 0] = wt[y == 0] * np.log(1.0 - mu[y == 0])
        r[y == 1] = wt[y == 1] * np.log(mu[y == 1])
        return np.sum(-2.0 * r)


class Inverse_Gaussian(Family):
    def deviance(self, y, mu, wt):
        pass


class QuasiPoisson(Family):
    def deviance(self, y, mu, wt):
        pass


class QuasiBinomial(Family):
    def deviance(self, y, mu, wt):
        pass


class Quasi(Family):
    def deviance(self, y, mu, wt):
        pass
