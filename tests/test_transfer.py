import numpy as np
import pytest
from fastsgd.transfer import Identity, Exponential, Inverse, Logistic


U_SCALAR = 2.0
U_ARRAY = np.array([-2.0, -1.0, 0.5, 1.0, 2.0])


class TestIdentity:
    def setup_method(self):
        self.t = Identity()

    def test_h_scalar(self):
        assert self.t.h(U_SCALAR) == U_SCALAR

    def test_h_vec(self):
        assert np.allclose(self.t.h_vec(U_ARRAY), U_ARRAY)

    def test_first_deriv_scalar(self):
        assert np.allclose(self.t.first_deriv(U_SCALAR), 1.0)

    def test_first_deriv_vec(self):
        assert np.allclose(self.t.first_deriv_vec(U_ARRAY), np.ones_like(U_ARRAY))

    def test_second_deriv_scalar(self):
        assert np.allclose(self.t.second_deriv(U_SCALAR), 0.0)

    def test_second_deriv_vec(self):
        assert np.allclose(self.t.second_deriv_vec(U_ARRAY), np.zeros_like(U_ARRAY))


class TestExponential:
    def setup_method(self):
        self.t = Exponential()

    def test_h_scalar(self):
        assert self.t.h(U_SCALAR) == pytest.approx(np.exp(U_SCALAR))

    def test_h_vec(self):
        assert np.allclose(self.t.h_vec(U_ARRAY), np.exp(U_ARRAY))

    def test_first_deriv_equals_h(self):
        assert np.allclose(self.t.first_deriv_vec(U_ARRAY), np.exp(U_ARRAY))

    def test_second_deriv_equals_h(self):
        assert np.allclose(self.t.second_deriv_vec(U_ARRAY), np.exp(U_ARRAY))


class TestInverse:
    def setup_method(self):
        self.t = Inverse()

    def test_h_scalar(self):
        assert self.t.h(2.0) == pytest.approx(-0.5)

    def test_h_vec(self):
        assert np.allclose(self.t.h_vec(U_ARRAY), -1.0 / U_ARRAY)

    def test_first_deriv_scalar(self):
        assert self.t.first_deriv(2.0) == pytest.approx(0.25)

    def test_first_deriv_vec(self):
        assert np.allclose(self.t.first_deriv_vec(U_ARRAY), 1.0 / (U_ARRAY ** 2))

    def test_second_deriv_scalar(self):
        assert self.t.second_deriv(2.0) == pytest.approx(-2.0 / 8.0)

    def test_second_deriv_vec(self):
        assert np.allclose(self.t.second_deriv_vec(U_ARRAY), -2.0 / (U_ARRAY ** 3))

    def test_first_deriv_vec_zero_input(self):
        result = self.t.first_deriv_vec(np.array([0.0]))
        assert result[0] == 0.0

    def test_second_deriv_vec_zero_input(self):
        result = self.t.second_deriv_vec(np.array([0.0]))
        assert result[0] == 0.0

    def test_no_warnings_on_zero_input(self, recwarn):
        self.t.first_deriv_vec(np.array([0.0, 1.0]))
        self.t.second_deriv_vec(np.array([0.0, 1.0]))
        runtime_warnings = [w for w in recwarn.list if issubclass(w.category, RuntimeWarning)]
        assert len(runtime_warnings) == 0


class TestLogistic:
    def setup_method(self):
        self.t = Logistic()
        self.sig = lambda x: 1.0 / (1.0 + np.exp(-x))

    def test_h_scalar(self):
        assert self.t.h(0.0) == pytest.approx(0.5)

    def test_h_vec(self):
        assert np.allclose(self.t.h_vec(U_ARRAY), self.sig(U_ARRAY))

    def test_first_deriv_at_zero(self):
        # σ'(0) = 0.5 * 0.5 = 0.25
        assert self.t.first_deriv(0.0) == pytest.approx(0.25)

    def test_first_deriv_vec(self):
        s = self.sig(U_ARRAY)
        assert np.allclose(self.t.first_deriv_vec(U_ARRAY), s * (1 - s))

    def test_second_deriv_at_zero(self):
        # σ''(0) = 2*(0.5)^3 - 3*(0.5)^2 + 0.5 = 0.25 - 0.75 + 0.5 = 0.0
        assert self.t.second_deriv(0.0) == pytest.approx(0.0)

    def test_second_deriv_vec_formula(self):
        s = self.sig(U_ARRAY)
        expected = 2 * s ** 3 - 3 * s ** 2 + s
        assert np.allclose(self.t.second_deriv_vec(U_ARRAY), expected)

    def test_second_deriv_not_using_wrong_formula(self):
        # Guard against regression to the old bug (coefficient 2 instead of 1)
        s = self.sig(U_ARRAY)
        wrong = 2 * s ** 3 - 3 * s ** 2 + 2 * s
        assert not np.allclose(self.t.second_deriv_vec(U_ARRAY), wrong)
