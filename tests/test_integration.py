"""
End-to-end integration tests: verify that ImplicitSGD and ExplicitSGD converge
to near-true parameters on simulated datasets for each supported model type.
"""
import time
import numpy as np
import pytest
from fastsgd import GLM, MModel, ImplicitSGD, ExplicitSGD, DataSet


def run_implicit(D, model, n, p, truth, npasses=30, reltol=1e-3):
    sgd = ImplicitSGD(n, p, time,
                      lr='adagrad', lr_controls={'eta': 1.0, 'eps': 1e-6},
                      npasses=npasses, check=True, truth=truth, reltol=reltol)
    theta = np.zeros(p)
    for t in range(1, n * npasses + 1):
        theta_new = sgd.update(t, theta, D, model, True)
        sgd.sync_members(theta_new)
        if sgd.convergence(theta_new, theta): break
        theta = theta_new
    return theta_new


def run_explicit(D, model, n, p, truth, npasses=30, reltol=1e-3):
    sgd = ExplicitSGD(n, p, time,
                      lr='adagrad', lr_controls={'eta': 1.0, 'eps': 1e-6},
                      npasses=npasses, check=True, truth=truth, reltol=reltol)
    theta = np.zeros(p)
    for t in range(1, n * npasses + 1):
        theta_new = sgd.update(t, theta, D, model, True)
        sgd.sync_members(theta_new)
        if sgd.convergence(theta_new, theta): break
        theta = theta_new
    return theta_new


class TestGaussianIdentity:
    def setup_method(self):
        np.random.seed(1)
        self.p = 5
        self.truth = np.array([1.0, -0.5, 0.25, -0.25, 0.5])
        n = 500
        X = np.random.randn(n, self.p)
        Y = X @ self.truth + np.random.randn(n) * 0.1
        self.D = DataSet(X, Y)
        self.n = n

    def test_implicit_sgd_converges(self):
        m = GLM(family='gaussian', transfer='identity')
        theta = run_implicit(self.D, m, self.n, self.p, self.truth)
        assert np.mean((theta - self.truth) ** 2) < 0.01

    def test_explicit_sgd_converges(self):
        m = GLM(family='gaussian', transfer='identity')
        theta = run_explicit(self.D, m, self.n, self.p, self.truth)
        assert np.mean((theta - self.truth) ** 2) < 0.01


class TestLogisticRegression:
    def setup_method(self):
        np.random.seed(2)
        self.p = 4
        self.truth = np.array([1.0, -1.0, 0.5, -0.5])
        n = 600
        X = np.random.randn(n, self.p)
        logits = X @ self.truth
        probs = 1.0 / (1.0 + np.exp(-logits))
        Y = (np.random.rand(n) < probs).astype(float)
        self.D = DataSet(X, Y)
        self.n = n

    def test_implicit_sgd_sign_correct(self):
        # Even if exact values differ, the signs should match the truth
        m = GLM(family='binomial', transfer='logistic')
        theta = run_implicit(self.D, m, self.n, self.p, self.truth, npasses=50, reltol=5e-3)
        assert np.all(np.sign(theta) == np.sign(self.truth))


class TestMEstimationHuber:
    def setup_method(self):
        np.random.seed(3)
        self.p = 3
        self.truth = np.array([2.0, -1.0, 0.5])
        n = 400
        X = np.random.randn(n, self.p)
        # Add occasional outliers
        noise = np.random.randn(n)
        outlier_mask = np.random.rand(n) < 0.05
        noise[outlier_mask] *= 10.0
        Y = X @ self.truth + noise
        self.D = DataSet(X, Y)
        self.n = n

    def test_explicit_sgd_huber_converges(self):
        # ImplicitSGD's brentq bracket is only guaranteed for GLMs;
        # ExplicitSGD is the correct choice for M-estimators.
        m = MModel(loss='huber', l=3.0)
        theta = run_explicit(self.D, m, self.n, self.p, self.truth, npasses=40)
        assert np.mean((theta - self.truth) ** 2) < 0.2


class TestLearningRateVariants:
    """Smoke test: all LR schedules complete without error on a simple problem."""

    def setup_method(self):
        np.random.seed(4)
        p = 3
        n = 100
        X = np.random.randn(n, p)
        Y = X @ np.ones(p) + np.random.randn(n) * 0.1
        self.D = DataSet(X, Y)
        self.n, self.p = n, p
        self.m = GLM(family='gaussian', transfer='identity')

    @pytest.mark.parametrize("lr,controls", [
        ("one-dim",       {"scale": 1.0, "alpha": 1.0, "gamma": 0.6, "c": 0.5}),
        ("one-dim-eigen", None),
        ("d-dim",         {"eps": 1e-6}),
        ("adagrad",       {"eta": 1.0, "eps": 1e-6}),
        ("rmsprop",       {"eta": 1.0, "gamma": 0.9, "eps": 1e-6}),
    ])
    def test_lr_does_not_crash(self, lr, controls):
        kwargs = {"lr": lr, "npasses": 2}
        if controls:
            kwargs["lr_controls"] = controls
        sgd = ImplicitSGD(self.n, self.p, time, **kwargs)
        theta = np.zeros(self.p)
        for t in range(1, self.n * 2 + 1):
            theta_new = sgd.update(t, theta, self.D, self.m, True)
            sgd.sync_members(theta_new)
            theta = theta_new
        assert np.all(np.isfinite(theta_new))
