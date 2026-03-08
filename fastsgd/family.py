from abc import ABC, abstractmethod
import numpy as np

__all__ = ['Family', 'Gaussian', 'Poisson', 'Gamma', 'Binomial']


class Family(ABC):
    """Abstract base for GLM response distributions.

    Subclasses define the variance function (stored as ``self.variance``) and
    the deviance, which measures goodness-of-fit on the response scale.
    """
    @abstractmethod
    def deviance(self, y: np.ndarray, mu: np.ndarray, wt: np.ndarray) -> float:
        """Return the total deviance D(y, mu) = 2 * sum( wt * d_i ).

        Parameters
        ----------
        y : np.ndarray
            Observed responses.
        mu : np.ndarray
            Fitted means (on the response scale, after applying the transfer).
        wt : np.ndarray
            Per-observation prior weights (e.g. binomial denominators).
        """
        pass


class Gaussian(Family):
    def __init__(self):
        self.variance = lambda x: np.ones_like(np.asarray(x, dtype=float))

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
        raise NotImplementedError("Inverse_Gaussian.deviance is not yet implemented.")


class QuasiPoisson(Family):
    def deviance(self, y, mu, wt):
        raise NotImplementedError("QuasiPoisson.deviance is not yet implemented.")


class QuasiBinomial(Family):
    def deviance(self, y, mu, wt):
        raise NotImplementedError("QuasiBinomial.deviance is not yet implemented.")


class Quasi(Family):
    def deviance(self, y, mu, wt):
        raise NotImplementedError("Quasi.deviance is not yet implemented.")
