import numpy as np
import pytest
from fastsgd.learning_rate.value import LRvalue
from fastsgd.learning_rate.base import BaseLR
from fastsgd.learning_rate.onedim import OnedimLR
from fastsgd.learning_rate.onedim_eig import OnedimEigLR
from fastsgd.learning_rate.ddim import DDimLR, ddimLR


class TestLRvalue:
    def test_scalar_type(self):
        v = LRvalue(0, 1)
        assert v.type == 0
        assert v.mean() == pytest.approx(1.0)

    def test_vector_type(self):
        v = LRvalue(1, 4)
        assert v.type == 1
        assert v.lr.shape == (4,)

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError):
            LRvalue(5)

    def test_matrix_type_raises(self):
        # Matrix type (2) was removed; seq_type=2 is now invalid
        with pytest.raises(ValueError):
            LRvalue(2, 3)

    def test_setter_scalar(self):
        v = LRvalue(0, 1)
        v.lr = 0.5
        assert v.mean() == pytest.approx(0.5)
        assert v.type == 0

    def test_setter_vector_updates_type(self):
        v = LRvalue(0, 1)
        v.lr = np.array([0.1, 0.2, 0.3])
        assert v.type == 1

    def test_mul_scalar(self):
        v = LRvalue(0, 1)
        v.lr = 2.0
        result = v * np.array([1.0, 2.0])
        assert np.allclose(result, [2.0, 4.0])

    def test_mul_vector(self):
        v = LRvalue(1, 3)
        v.lr = np.array([1.0, 2.0, 3.0])
        result = v * np.ones(3)
        assert np.allclose(result, [1.0, 2.0, 3.0])

    def test_lt(self):
        v = LRvalue(1, 3)
        v.lr = np.array([0.1, 0.2, 0.3])
        assert v < 1.0
        assert not (v < 0.1)

    def test_gt(self):
        v = LRvalue(1, 3)
        v.lr = np.array([2.0, 3.0, 4.0])
        assert v > 1.0
        assert not (v > 5.0)

    def test_gt_not_same_as_not_lt(self):
        # Guard against regression to the old bug: not(a < x) is >= not >
        v = LRvalue(1, 2)
        v.lr = np.array([1.0, 2.0])  # one element equals threshold
        assert not (v > 1.0)   # not ALL elements are strictly > 1.0
        assert not (v < 1.0)   # not ALL elements are < 1.0 either


class TestBaseLRABC:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            BaseLR()


class TestOnedimLR:
    def test_returns_lrvalue(self):
        lr = OnedimLR(scale=1.0, gamma=0.6, alpha=1.0, c=0.5)
        result = lr(1, np.zeros(3))
        assert isinstance(result, LRvalue)

    def test_decay_formula(self):
        scale, gamma, alpha, c = 1.0, 0.6, 1.0, 0.5
        lr = OnedimLR(scale, gamma, alpha, c)
        t = 5
        expected = scale * gamma * (1 + alpha * gamma * t) ** (-c)
        result = lr(t, np.zeros(3))
        assert result.mean() == pytest.approx(expected)

    def test_decreases_over_time(self):
        lr = OnedimLR(1.0, 0.6, 1.0, 0.5)
        r1 = lr(1, np.zeros(3)).mean()
        r10 = lr(10, np.zeros(3)).mean()
        assert r10 < r1


class TestOnedimEigLR:
    def test_returns_lrvalue(self):
        lr = OnedimEigLR(d=3)
        result = lr(1, np.ones(3))
        assert isinstance(result, LRvalue)

    def test_normal_formula(self):
        d = 4
        lr = OnedimEigLR(d)
        grad = np.array([1.0, 2.0, 3.0, 4.0])
        t = 2
        sum_eigen = np.sum(grad ** 2)
        expected = 1.0 / ((sum_eigen / d) * t)
        assert lr(t, grad).mean() == pytest.approx(expected)

    def test_zero_gradient_returns_1_over_t(self):
        lr = OnedimEigLR(d=3)
        t = 7
        result = lr(t, np.zeros(3)).mean()
        assert result == pytest.approx(1.0 / t)

    def test_decreases_over_time(self):
        lr = OnedimEigLR(d=3)
        grad = np.ones(3)
        r1 = lr(1, grad).mean()
        r10 = lr(10, grad).mean()
        assert r10 < r1


class TestDDimLR:
    def test_returns_lrvalue(self):
        lr = DDimLR(d=3, eta=1.0, a=1.0, b=1.0, c=0.5, eps=1e-6)
        result = lr(1, np.ones(3))
        assert isinstance(result, LRvalue)

    def test_adagrad_specialisation(self):
        # AdaGrad: a=1, b=1, c=0.5, eta=1.0
        d = 3
        lr = DDimLR(d=d, eta=1.0, a=1.0, b=1.0, c=0.5, eps=1e-6)
        grad = np.array([1.0, 2.0, 3.0])
        result = lr(1, grad)
        expected = 1.0 / np.sqrt(1.0 * np.ones(d) + 1.0 * grad ** 2 + 1e-6)
        assert np.allclose(result.lr, expected)

    def test_ddim_alias(self):
        assert ddimLR is DDimLR

    def test_vector_output_shape(self):
        d = 5
        lr = DDimLR(d=d, eta=1.0, a=1.0, b=1.0, c=0.5, eps=1e-6)
        result = lr(1, np.ones(d))
        assert result.lr.shape == (d,)
