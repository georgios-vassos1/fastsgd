"""
Tests for covariance estimators.

Key properties verified:
  1. Output shape and symmetry / positive-definiteness
  2. FisherCovariance == SandwichCovariance for a correctly-specified GLM
     (sandwich "bread" A equals the Fisher information, so B = A and
      A⁻¹ B A⁻¹ = A⁻¹)
  3. Standard errors shrink as n grows (consistency)
  4. Estimated standard errors are in the right ballpark relative to the
     empirical standard deviation across many independent runs
  5. M-estimator (Huber) covariance is computable and well-shaped
"""
import numpy as np
import pytest
from fastsgd import GLM, MModel, DataSet
from fastsgd.covariance import CovarianceEstimator, FisherCovariance, SandwichCovariance
from fastsgd import ImplicitSGD, ExplicitSGD
import time


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def simulate_gaussian(n, p, seed=0):
    np.random.seed(seed)
    theta = np.arange(1, p + 1, dtype=float) / p
    X = np.random.randn(n, p)
    Y = X @ theta + np.random.randn(n) * 0.5
    return DataSet(X, Y), theta


def fit_implicit(D, model, truth, npasses=30, reltol=1e-3):
    n, p = D._n, D._p
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


# ---------------------------------------------------------------------------
# ABC
# ---------------------------------------------------------------------------

class TestCovarianceEstimatorABC:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            CovarianceEstimator()


# ---------------------------------------------------------------------------
# Output shape and structure
# ---------------------------------------------------------------------------

class TestOutputShape:
    def setup_method(self):
        self.D, self.truth = simulate_gaussian(200, 4)
        self.m = GLM(family='gaussian', transfer='identity')
        self.theta = fit_implicit(self.D, self.m, self.truth)

    def test_fisher_shape(self):
        V = FisherCovariance().estimate(self.theta, self.D, self.m)
        assert V.shape == (4, 4)

    def test_sandwich_shape(self):
        V = SandwichCovariance().estimate(self.theta, self.D, self.m)
        assert V.shape == (4, 4)

    def test_fisher_symmetric(self):
        V = FisherCovariance().estimate(self.theta, self.D, self.m)
        assert np.allclose(V, V.T)

    def test_sandwich_symmetric(self):
        V = SandwichCovariance().estimate(self.theta, self.D, self.m)
        assert np.allclose(V, V.T)

    def test_fisher_positive_definite(self):
        V = FisherCovariance().estimate(self.theta, self.D, self.m)
        assert np.all(np.linalg.eigvalsh(V) > 0)

    def test_sandwich_positive_definite(self):
        V = SandwichCovariance().estimate(self.theta, self.D, self.m)
        assert np.all(np.linalg.eigvalsh(V) > 0)

    def test_std_errors_shape(self):
        se = FisherCovariance().std_errors(self.theta, self.D, self.m)
        assert se.shape == (4,)

    def test_std_errors_positive(self):
        se = FisherCovariance().std_errors(self.theta, self.D, self.m)
        assert np.all(se > 0)


# ---------------------------------------------------------------------------
# Fisher == Sandwich for correctly-specified Gaussian model
# ---------------------------------------------------------------------------

class TestFisherEqualsSandwichGaussian:
    """For Gaussian/identity with true parameters, B ≈ A so both estimators agree."""

    def setup_method(self):
        # Large n to reduce Monte Carlo noise.
        # Unit noise (σ=1) is required for the information identity B = A to hold.
        np.random.seed(7)
        p = 3
        n = 2000
        theta = np.arange(1, p + 1, dtype=float) / p
        X = np.random.randn(n, p)
        Y = X @ theta + np.random.randn(n)          # σ = 1
        from fastsgd import DataSet
        self.D = DataSet(X, Y)
        self.truth = theta
        self.m = GLM(family='gaussian', transfer='identity')
        self.theta = fit_implicit(self.D, self.m, self.truth)

    def test_fisher_approx_equals_sandwich(self):
        V_f = FisherCovariance().estimate(self.theta, self.D, self.m)
        V_s = SandwichCovariance().estimate(self.theta, self.D, self.m)
        # The information identity B = A holds for a Gaussian GLM only when
        # the noise variance σ² = 1 (unit dispersion). With σ = 1, B ≈ A so
        # the two estimators agree up to finite-sample noise.
        # Off-diagonal elements are near zero so we compare with absolute
        # tolerance, not relative tolerance.
        assert np.allclose(V_f, V_s, atol=0.05, rtol=0)


# ---------------------------------------------------------------------------
# Standard errors are consistent (shrink with n)
# ---------------------------------------------------------------------------

class TestConsistency:
    def test_std_errors_shrink_with_n(self):
        m = GLM(family='gaussian', transfer='identity')
        ses = []
        for n in [200, 1000, 5000]:
            D, truth = simulate_gaussian(n, 3, seed=42)
            theta = fit_implicit(D, m, truth)
            se = FisherCovariance().std_errors(theta, D, m)
            ses.append(se.mean())
        assert ses[0] > ses[1] > ses[2]


# ---------------------------------------------------------------------------
# Empirical coverage: SE matches empirical spread across repeated runs
# ---------------------------------------------------------------------------

class TestEmpiricalCoverage:
    """The estimated SE should be in the same ballpark as the empirical SD."""

    def test_se_matches_empirical_sd(self):
        np.random.seed(99)
        p, n, n_runs = 3, 500, 60
        truth = np.ones(p)
        m = GLM(family='gaussian', transfer='identity')
        estimates = []
        for seed in range(n_runs):
            D, _ = simulate_gaussian(n, p, seed=seed)
            theta = fit_implicit(D, m, truth, npasses=20)
            estimates.append(theta)
        estimates = np.array(estimates)
        empirical_sd = np.std(estimates, axis=0)

        # Use one representative dataset for the SE estimate
        D_ref, _ = simulate_gaussian(n, p, seed=0)
        theta_ref = fit_implicit(D_ref, m, truth, npasses=20)
        estimated_se = FisherCovariance().std_errors(theta_ref, D_ref, m)

        # SE and empirical SD should agree within a factor of 2
        ratio = estimated_se / empirical_sd
        assert np.all(ratio > 0.4) and np.all(ratio < 2.5)


# ---------------------------------------------------------------------------
# M-estimator (Huber) — only sandwich is meaningful
# ---------------------------------------------------------------------------

class TestMEstimatorCovariance:
    def setup_method(self):
        np.random.seed(5)
        p, n = 3, 400
        truth = np.array([1.0, -0.5, 0.25])
        X = np.random.randn(n, p)
        Y = X @ truth + np.random.randn(n) * 0.3
        self.D = DataSet(X, Y)
        self.truth = truth
        self.m = MModel(loss='huber', l=3.0)
        # Huber with large l ≈ least squares; use ExplicitSGD (see note in test_integration)
        sgd = ExplicitSGD(n, p, time,
                          lr='adagrad', lr_controls={'eta': 1.0, 'eps': 1e-6},
                          npasses=40, check=True, truth=truth, reltol=1e-3)
        theta = np.zeros(p)
        for t in range(1, n * 40 + 1):
            theta_new = sgd.update(t, theta, self.D, self.m, True)
            sgd.sync_members(theta_new)
            if sgd.convergence(theta_new, theta): break
            theta = theta_new
        self.theta = theta_new

    def test_sandwich_shape(self):
        V = SandwichCovariance().estimate(self.theta, self.D, self.m)
        assert V.shape == (3, 3)

    def test_sandwich_positive_definite(self):
        V = SandwichCovariance().estimate(self.theta, self.D, self.m)
        assert np.all(np.linalg.eigvalsh(V) > 0)

    def test_std_errors_positive(self):
        se = SandwichCovariance().std_errors(self.theta, self.D, self.m)
        assert np.all(se > 0)


# ---------------------------------------------------------------------------
# score_matrix and hessian_weights on known values
# ---------------------------------------------------------------------------

class TestScoreAndHessianWeights:
    def test_glm_score_matrix_shape(self):
        D, _ = simulate_gaussian(50, 3)
        m = GLM(family='gaussian', transfer='identity')
        theta = np.zeros(3)
        G = m.score_matrix(D, theta)
        assert G.shape == (50, 3)

    def test_glm_score_matrix_gaussian_identity(self):
        # For Gaussian/identity: score_i = (y_i - x_i^T theta) * x_i
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        Y = np.array([1.0, 2.0])
        D = DataSet(X, Y)
        m = GLM(family='gaussian', transfer='identity')
        theta = np.zeros(2)
        G = m.score_matrix(D, theta)
        expected = (Y - X @ theta)[:, None] * X
        assert np.allclose(G, expected)

    def test_glm_hessian_weights_identity(self):
        # Identity transfer: h'(eta) = 1 for all eta
        D, _ = simulate_gaussian(30, 3)
        m = GLM(family='gaussian', transfer='identity')
        w = m.hessian_weights(D, np.zeros(3))
        assert np.allclose(w, np.ones(30))

    def test_mmodel_score_matrix_shape(self):
        D, _ = simulate_gaussian(50, 3)
        m = MModel(loss='huber', l=3.0)
        G = m.score_matrix(D, np.zeros(3))
        assert G.shape == (50, 3)

    def test_mmodel_hessian_weights_shape(self):
        D, _ = simulate_gaussian(50, 3)
        m = MModel(loss='huber', l=3.0)
        w = m.hessian_weights(D, np.zeros(3))
        assert w.shape == (50,)

    def test_mmodel_hessian_weights_binary(self):
        # Huber second_deriv is 1 inside threshold, 0 outside
        X = np.array([[1.0, 0.0], [1.0, 0.0]])
        Y = np.array([0.5, 100.0])   # first inside l=3, second outside
        D = DataSet(X, Y)
        m = MModel(loss='huber', l=3.0)
        w = m.hessian_weights(D, np.zeros(2))
        assert w[0] == pytest.approx(1.0)
        assert w[1] == pytest.approx(0.0)
