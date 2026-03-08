import numpy as np
import pytest
from fastsgd.m_model import MModel
from fastsgd.utils import DataPoint, DataSet


class TestMModelConstruction:
    def test_default_threshold(self):
        m = MModel()
        assert m.threshold == 3.0

    def test_custom_threshold(self):
        m = MModel(loss='huber', threshold=1.5)
        assert m.threshold == 1.5

    def test_invalid_loss_raises(self):
        with pytest.raises(ValueError, match="Unsupported loss"):
            MModel(loss='squared')

    def test_repr(self):
        m = MModel(loss='huber', threshold=1.5)
        assert repr(m) == "MModel(loss='huber', threshold=1.5)"

    def test_loss_property(self):
        m = MModel(loss='huber')
        assert m.loss == 'huber'


class TestMModelGradient:
    def setup_method(self):
        self.threshold = 3.0
        self.m = MModel(loss='huber', threshold=self.threshold)
        self.x = np.array([1.0, 2.0])
        self.y = 1.0
        self.theta = np.zeros(2)
        self.datum = DataPoint(self.x, self.y, 0)
        X = self.x.reshape(1, 2)
        Y = np.array([self.y])
        self.D = DataSet(X, Y)

    def test_gradient_huber_quadratic_branch(self):
        # residual = y - x^T theta = 1.0, which is < l=3 → first_deriv = residual
        grad = self.m._gradient_at_point(self.datum, self.theta)
        expected = self.y * self.x  # first_deriv(1.0, 3.0) = 1.0
        assert np.allclose(grad, expected)

    def test_gradient_via_data_set(self):
        grad_direct = self.m._gradient_at_point(self.datum, self.theta)
        grad_via_data = self.m.gradient(1, self.theta, self.D)
        assert np.allclose(grad_direct, grad_via_data)

    def test_gradient_huber_linear_branch(self):
        # Make residual >> l so we're in the linear branch
        m = MModel(loss='huber', threshold=0.1)
        theta = np.zeros(2)
        datum = DataPoint(np.array([1.0, 0.0]), 5.0, 0)
        # residual = 5.0 > l=0.1 → first_deriv = 0.1 * sign(5.0) = 0.1
        grad = m._gradient_at_point(datum, theta)
        assert np.allclose(grad, np.array([0.1, 0.0]))


class TestMModelScaleFactor:
    def setup_method(self):
        self.m = MModel(loss='huber', threshold=3.0)
        self.x = np.array([1.0, 2.0])
        self.y = 1.0
        self.theta = np.zeros(2)
        self.datum = DataPoint(self.x, self.y, 0)
        self.normx = np.linalg.norm(self.x)

    def test_scale_factor_quadratic_branch(self):
        # ksi=0: residual = y - x^T theta = 1.0, first_deriv(1.0, 3.0) = 1.0
        sf = self.m.scale_factor(0.0, 0.1, self.datum, self.theta, self.normx)
        assert sf == pytest.approx(1.0)
