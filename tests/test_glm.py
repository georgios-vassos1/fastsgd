import numpy as np
import pytest
from fastsgd.glm import GLM, glm
from fastsgd.utils import DataPoint, DataSet


class TestGLMConstruction:
    def test_valid_gaussian_identity(self):
        GLM(family='gaussian', transfer='identity')

    def test_valid_poisson_exponential(self):
        GLM(family='poisson', transfer='exponential')

    def test_valid_gamma_inverse(self):
        GLM(family='gamma', transfer='inverse')

    def test_valid_binomial_logistic(self):
        GLM(family='binomial', transfer='logistic')

    def test_invalid_family_raises(self):
        with pytest.raises(ValueError, match="Unknown family"):
            GLM(family='invalid', transfer='identity')

    def test_invalid_transfer_raises(self):
        with pytest.raises(ValueError, match="Unknown transfer"):
            GLM(family='gaussian', transfer='invalid')

    def test_glm_alias(self):
        assert glm is GLM

    def test_lambda_defaults_to_zero(self):
        m = GLM(family='gaussian', transfer='identity')
        assert m._lambda1 == 0.0
        assert m._lambda2 == 0.0


class TestGLMGradient:
    def setup_method(self):
        # Gaussian/identity: gradient = (y - x^T theta) * x
        self.m = GLM(family='gaussian', transfer='identity')
        self.x = np.array([1.0, 2.0, 3.0])
        self.y = 6.0
        self.theta = np.zeros(3)
        X = self.x.reshape(1, 3)
        Y = np.array([self.y])
        self.D = DataSet(X, Y)

    def test_gradient_at_zero_theta(self):
        # gradient = (y - 0) * x = y * x
        expected = self.y * self.x
        grad = self.m.gradient(1, self.theta, self.D)
        assert np.allclose(grad, expected)

    def test_gradient_at_nonzero_theta(self):
        theta = np.array([1.0, 1.0, 1.0])
        residual = self.y - self.x @ theta  # 6 - 6 = 0
        expected = residual * self.x
        grad = self.m.gradient(1, theta, self.D)
        assert np.allclose(grad, expected)

    def test_gradient_with_l2_penalty(self):
        m = GLM(family='gaussian', transfer='identity', lambda2=1.0)
        theta = np.array([1.0, 0.0, 0.0])
        residual = self.y - self.x @ theta
        expected = residual * self.x - theta  # L2 penalty subtracts theta
        grad = m.gradient(1, theta, self.D)
        assert np.allclose(grad, expected)


class TestGLMScaleFactor:
    def setup_method(self):
        self.m = GLM(family='gaussian', transfer='identity')
        self.x = np.array([1.0, 2.0])
        self.y = 5.0
        self.theta = np.array([1.0, 1.0])
        self.datum = DataPoint(self.x, self.y, 0)
        self.normx = np.linalg.norm(self.x)

    def test_scale_factor_gaussian_identity(self):
        # scale_factor(ksi=0) = y - (theta @ x) = 5 - 3 = 2
        sf = self.m.scale_factor(0.0, 0.1, self.datum, self.theta, self.normx)
        assert sf == pytest.approx(self.y - self.x @ self.theta)

    def test_scale_factor_first_deriv_gaussian_identity(self):
        # first_deriv of identity is 1, so result = 1 * normx
        sfd = self.m.scale_factor_first_deriv(0.0, 0.1, self.datum, self.theta, self.normx)
        assert sfd == pytest.approx(self.normx)

    def test_scale_factor_second_deriv_gaussian_identity(self):
        # second_deriv of identity is 0
        sfdd = self.m.scale_factor_second_deriv(0.0, 0.1, self.datum, self.theta, self.normx)
        assert sfdd == pytest.approx(0.0)
