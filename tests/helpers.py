"""Shared helpers for the test suite."""
import numpy as np
from fastsgd import ImplicitSGD, DataSet


def simulate_gaussian(n, p, seed=0):
    """Return a (DataSet, truth) pair for a Gaussian/identity GLM."""
    np.random.seed(seed)
    theta = np.arange(1, p + 1, dtype=float) / p
    X = np.random.randn(n, p)
    Y = X @ theta + np.random.randn(n) * 0.5
    return DataSet(X, Y), theta


def fit_implicit(D, model, truth, npasses=30, reltol=1e-3):
    """Run ImplicitSGD with AdaGrad until convergence and return the estimate."""
    n, p = D._n, D._p
    sgd = ImplicitSGD(n, p,
                      lr='adagrad', lr_controls={'eta': 1.0, 'eps': 1e-6},
                      npasses=npasses, check=True, truth=truth, reltol=reltol)
    theta = np.zeros(p)
    for t in range(1, n * npasses + 1):
        theta_new = sgd.update(t, theta, D, model, True)
        sgd.sync_members(theta_new)
        if sgd.convergence(theta_new, theta):
            break
        theta = theta_new
    return theta_new
