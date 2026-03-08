import numpy as np
import pytest
from fastsgd.model import Model, model
from fastsgd.utils import DataPoint, DataSet


class TestModelABC:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            Model("test", 0.0, 0.0)

    def test_alias(self):
        assert model is Model


class TestGradientPenalty:
    """Use a concrete subclass to access gradient_penalty."""

    class ConcreteModel(Model):
        def _gradient_at_point(self, datum, theta_old):
            return np.zeros_like(theta_old)

        def scale_factor(self, ksi, at, datum, theta_old, normx):
            return 0.0

        def score_matrix(self, data, theta):
            return np.zeros((data._n, data._p))

        def hessian_weights(self, data, theta):
            return np.ones(data._n)

    def setup_method(self):
        self.theta = np.array([1.0, -2.0, 0.0, 3.0])

    def test_no_penalty(self):
        m = self.ConcreteModel("test", 0.0, 0.0)
        assert np.allclose(m.gradient_penalty(self.theta), np.zeros(4))

    def test_l1_penalty(self):
        m = self.ConcreteModel("test", 1.0, 0.0)
        expected = np.sign(self.theta)
        assert np.allclose(m.gradient_penalty(self.theta), expected)

    def test_l2_penalty(self):
        m = self.ConcreteModel("test", 0.0, 1.0)
        assert np.allclose(m.gradient_penalty(self.theta), self.theta)

    def test_elastic_net_penalty(self):
        l1, l2 = 0.5, 0.3
        m = self.ConcreteModel("test", l1, l2)
        expected = l1 * np.sign(self.theta) + l2 * self.theta
        assert np.allclose(m.gradient_penalty(self.theta), expected)

    def test_gradient_delegates_to_gradient_at_point(self):
        m = self.ConcreteModel("test", 0.0, 0.0)
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        Y = np.array([0.1, 0.2])
        D = DataSet(X, Y)
        theta = np.zeros(2)
        # gradient() should call _gradient_at_point() and return zeros (our stub)
        result = m.gradient(1, theta, D)
        assert np.allclose(result, np.zeros(2))
