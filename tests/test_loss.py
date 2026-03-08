import numpy as np
import pytest
from fastsgd.loss import BaseLoss, HuberLoss


class TestBaseLossABC:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            BaseLoss()


class TestHuberLoss:
    def setup_method(self):
        self.h = HuberLoss()
        self.l = 1.0

    # --- loss ---

    def test_loss_quadratic_branch(self):
        # |u| <= l: u^2 / 2
        assert self.h.loss(0.5, self.l) == pytest.approx(0.125)

    def test_loss_linear_branch(self):
        # |u| > l: l*|u| - l^2/2
        assert self.h.loss(2.0, self.l) == pytest.approx(2.0 * 1.0 - 1.0 / 2)

    def test_loss_continuous_at_boundary(self):
        # Both branches agree at |u| == l
        assert self.h.loss(self.l, self.l) == pytest.approx(self.h.loss(self.l + 1e-9, self.l), rel=1e-4)

    def test_loss_symmetric(self):
        assert self.h.loss(0.5, self.l) == pytest.approx(self.h.loss(-0.5, self.l))
        assert self.h.loss(2.0, self.l) == pytest.approx(self.h.loss(-2.0, self.l))

    def test_loss_vec(self):
        u = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
        expected = np.where(np.abs(u) <= self.l, u ** 2 / 2, self.l * np.abs(u) - self.l ** 2 / 2)
        assert np.allclose(self.h.loss_vec(u, self.l), expected)

    # --- first_deriv ---

    def test_first_deriv_quadratic_branch(self):
        assert self.h.first_deriv(0.5, self.l) == pytest.approx(0.5)

    def test_first_deriv_linear_branch_positive(self):
        assert self.h.first_deriv(2.0, self.l) == pytest.approx(1.0)

    def test_first_deriv_linear_branch_negative(self):
        assert self.h.first_deriv(-2.0, self.l) == pytest.approx(-1.0)

    def test_first_deriv_vec(self):
        u = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
        expected = np.where(np.abs(u) <= self.l, u, self.l * np.sign(u))
        assert np.allclose(self.h.first_deriv_vec(u, self.l), expected)

    # --- second_deriv ---

    def test_second_deriv_quadratic_branch(self):
        assert self.h.second_deriv(0.5, self.l) == pytest.approx(1.0)

    def test_second_deriv_linear_branch(self):
        assert self.h.second_deriv(2.0, self.l) == pytest.approx(0.0)

    def test_second_deriv_vec(self):
        u = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
        expected = np.where(np.abs(u) <= self.l, 1.0, 0.0)
        assert np.allclose(self.h.second_deriv_vec(u, self.l), expected)

    # --- third_deriv ---

    def test_third_deriv_always_zero(self):
        for u in [-2.0, -0.5, 0.0, 0.5, 2.0]:
            assert self.h.third_deriv(u, self.l) == pytest.approx(0.0)
