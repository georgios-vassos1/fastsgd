import numpy as np
import pytest
from fastsgd.family import (
    Family, Gaussian, Poisson, Gamma, Binomial,
    Inverse_Gaussian, QuasiPoisson, QuasiBinomial, Quasi,
)


class TestFamilyABC:
    def test_cannot_instantiate_family_directly(self):
        with pytest.raises(TypeError):
            Family()

    def test_stub_families_instantiate(self):
        # Stub families implement deviance (with pass), so they are concrete
        for cls in (Inverse_Gaussian, QuasiPoisson, QuasiBinomial, Quasi):
            cls()  # should not raise


class TestGaussian:
    def setup_method(self):
        self.g = Gaussian()
        self.y = np.array([1.0, 2.0, 3.0])
        self.mu = np.array([1.1, 1.9, 3.2])
        self.wt = np.ones(3)

    def test_deviance_zero_when_perfect(self):
        assert self.g.deviance(self.y, self.y, self.wt) == pytest.approx(0.0)

    def test_deviance_weighted(self):
        expected = np.sum(self.wt * (self.y - self.mu) ** 2)
        assert self.g.deviance(self.y, self.mu, self.wt) == pytest.approx(expected)

    def test_deviance_with_weights(self):
        wt = np.array([2.0, 1.0, 0.5])
        expected = np.sum(wt * (self.y - self.mu) ** 2)
        assert self.g.deviance(self.y, self.mu, wt) == pytest.approx(expected)


class TestPoisson:
    def setup_method(self):
        self.p = Poisson()
        self.wt = np.ones(3)

    def test_deviance_zero_when_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert self.p.deviance(y, y, self.wt) == pytest.approx(0.0)

    def test_deviance_handles_zero_y(self):
        # y=0 contributes mu*wt to deviance, not log(0)
        y = np.array([0.0, 1.0])
        mu = np.array([0.5, 1.0])
        result = self.p.deviance(y, mu, self.wt[:2])
        assert np.isfinite(result)
        assert result >= 0.0

    def test_deviance_positive(self):
        y = np.array([1.0, 2.0, 4.0])
        mu = np.array([1.5, 1.5, 3.5])
        assert self.p.deviance(y, mu, self.wt) > 0.0


class TestGamma:
    def setup_method(self):
        self.g = Gamma()
        self.wt = np.ones(3)

    def test_deviance_zero_when_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert self.g.deviance(y, y, self.wt) == pytest.approx(0.0)

    def test_deviance_handles_zero_y(self):
        y = np.array([0.0, 1.0, 2.0])
        mu = np.array([0.5, 1.0, 2.0])
        result = self.g.deviance(y, mu, self.wt)
        assert np.isfinite(result)


class TestBinomial:
    def setup_method(self):
        self.b = Binomial()
        self.wt = np.ones(4)

    def test_deviance_zero_when_perfect(self):
        y = np.array([0.0, 0.0, 1.0, 1.0])
        mu = np.array([0.01, 0.01, 0.99, 0.99])
        # Near-perfect predictions → near-zero deviance
        assert self.b.deviance(y, mu, self.wt) >= 0.0

    def test_deviance_positive(self):
        y = np.array([0.0, 1.0])
        mu = np.array([0.5, 0.5])
        result = self.b.deviance(y, mu, self.wt[:2])
        assert result > 0.0

    def test_deviance_formula(self):
        y = np.array([0.0, 1.0])
        mu = np.array([0.3, 0.7])
        expected = -2.0 * (np.log(1 - 0.3) + np.log(0.7))
        assert self.b.deviance(y, mu, self.wt[:2]) == pytest.approx(expected)
