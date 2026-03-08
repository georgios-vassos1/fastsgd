import time
import numpy as np
import pytest
from fastsgd.utils import DataSet
from fastsgd.glm import GLM
from fastsgd.explicit_sgd import ExplicitSGD
from fastsgd.implicit_sgd import ImplicitSGD


def make_gaussian_problem(n=100, p=3, seed=42):
    np.random.seed(seed)
    theta_true = np.ones(p)
    X = np.random.randn(n, p)
    Y = X @ theta_true + np.random.randn(n) * 0.1
    return DataSet(X, Y), theta_true


class TestSGDInit:
    def test_default_attributes(self):
        sgd = ExplicitSGD(100, 3, time, npasses=5)
        assert sgd._n_params == 3
        assert sgd._n_passes == 5
        assert sgd._good_gradient is True
        assert sgd._t == 0

    def test_get_value_of(self):
        sgd = ExplicitSGD(100, 3, time, npasses=7)
        assert sgd.get_value_of("n_passes") == 7

    def test_get_value_of_unknown_raises(self):
        sgd = ExplicitSGD(100, 3, time)
        with pytest.raises(AttributeError):
            sgd.get_value_of("nonexistent")

    def test_all_lr_choices_initialise(self):
        n, p = 100, 3
        for lr, controls in [
            ("one-dim",      {"scale": 1.0, "alpha": 1.0, "gamma": 0.6, "c": 0.5}),
            ("one-dim-eigen", None),
            ("d-dim",        {"eps": 1e-6}),
            ("adagrad",      {"eta": 1.0, "eps": 1e-6}),
            ("rmsprop",      {"eta": 1.0, "gamma": 0.9, "eps": 1e-6}),
        ]:
            kwargs = {"lr": lr}
            if controls:
                kwargs["lr_controls"] = controls
            ExplicitSGD(n, p, time, **kwargs)  # should not raise


class TestConvergence:
    def setup_method(self):
        self.p = 3
        self.truth = np.ones(self.p)

    def test_convergence_with_truth_triggers_when_close(self):
        sgd = ExplicitSGD(100, self.p, time, check=True, truth=self.truth, reltol=1.0)
        theta_new = self.truth + 0.01
        theta_old = self.truth
        assert sgd.convergence(theta_new, theta_old) is True

    def test_convergence_with_truth_false_when_far(self):
        sgd = ExplicitSGD(100, self.p, time, check=True, truth=self.truth, reltol=1e-5)
        theta_new = self.truth + 10.0
        theta_old = self.truth
        assert sgd.convergence(theta_new, theta_old) is False

    def test_convergence_relative_triggers_when_close(self):
        sgd = ExplicitSGD(100, self.p, time, check=False, reltol=1.0)
        theta_old = np.ones(self.p)
        theta_new = theta_old * 1.001
        assert sgd.convergence(theta_new, theta_old) is True

    def test_convergence_relative_false_when_far(self):
        sgd = ExplicitSGD(100, self.p, time, check=False, reltol=1e-10)
        theta_old = np.ones(self.p)
        theta_new = theta_old * 2.0
        assert sgd.convergence(theta_new, theta_old) is False


class TestSyncMembers:
    def test_t_increments(self):
        sgd = ExplicitSGD(10, 2, time, npasses=1, size=3)
        theta = np.ones(2)
        sgd.sync_members(theta)
        assert sgd._t == 1

    def test_last_estimate_updated(self):
        sgd = ExplicitSGD(10, 2, time, npasses=1)
        theta = np.array([1.5, 2.5])
        sgd.sync_members(theta)
        assert np.allclose(sgd._last_estimate, theta)


class TestAveragedEstimate:
    def test_returns_last_estimate_before_any_snapshot(self):
        sgd = ExplicitSGD(100, 3, time, npasses=10, size=5)
        result = sgd.averaged_estimate()
        assert result.shape == (3,)
        assert np.allclose(result, np.zeros(3))

    def test_shape_after_snapshots(self):
        D, truth = make_gaussian_problem(n=100, p=3)
        m = GLM(family='gaussian', transfer='identity')
        sgd = ExplicitSGD(100, 3, time, lr='adagrad', lr_controls={'eta': 1.0, 'eps': 1e-6},
                          npasses=5)
        theta = np.zeros(3)
        for t in range(1, 100 * 5 + 1):
            theta = sgd.update(t, theta, D, m, True)
            sgd.sync_members(theta)
        result = sgd.averaged_estimate()
        assert result.shape == (3,)

    def test_average_is_finite(self):
        D, truth = make_gaussian_problem(n=100, p=3)
        m = GLM(family='gaussian', transfer='identity')
        sgd = ExplicitSGD(100, 3, time, lr='adagrad', lr_controls={'eta': 1.0, 'eps': 1e-6},
                          npasses=5)
        theta = np.zeros(3)
        for t in range(1, 100 * 5 + 1):
            theta = sgd.update(t, theta, D, m, True)
            sgd.sync_members(theta)
        assert np.all(np.isfinite(sgd.averaged_estimate()))


class TestGoodGradient:
    def test_good_gradient_set_false_on_inf(self):
        D, _ = make_gaussian_problem()
        m = GLM(family='gaussian', transfer='identity')
        sgd = ExplicitSGD(100, 3, time)
        # Corrupt the model output by passing infinite theta
        theta_inf = np.array([np.inf, np.inf, np.inf])
        sgd.update(1, theta_inf, D, m, True)
        assert sgd._good_gradient is False

    def test_good_gradient_stays_true_on_finite(self):
        D, _ = make_gaussian_problem()
        m = GLM(family='gaussian', transfer='identity')
        sgd = ExplicitSGD(100, 3, time)
        sgd.update(1, np.zeros(3), D, m, True)
        assert sgd._good_gradient is True


class TestExplicitSGDUpdate:
    def test_update_returns_array_of_correct_shape(self):
        D, _ = make_gaussian_problem(p=3)
        m = GLM(family='gaussian', transfer='identity')
        sgd = ExplicitSGD(100, 3, time, lr='adagrad', lr_controls={'eta': 1.0, 'eps': 1e-6})
        theta_new = sgd.update(1, np.zeros(3), D, m, True)
        assert theta_new.shape == (3,)

    def test_update_moves_toward_truth(self):
        # After many steps, theta should be closer to truth than zeros
        D, truth = make_gaussian_problem(n=200, p=3)
        m = GLM(family='gaussian', transfer='identity')
        sgd = ExplicitSGD(200, 3, time, lr='adagrad', lr_controls={'eta': 1.0, 'eps': 1e-6},
                          npasses=20)
        theta = np.zeros(3)
        for t in range(1, 200 * 20 + 1):
            theta_new = sgd.update(t, theta, D, m, True)
            sgd.sync_members(theta_new)
            if sgd.convergence(theta_new, theta): break
            theta = theta_new
        assert np.mean((theta_new - truth) ** 2) < 0.1


class TestImplicitSGDUpdate:
    def test_update_returns_array_of_correct_shape(self):
        D, _ = make_gaussian_problem(p=3)
        m = GLM(family='gaussian', transfer='identity')
        sgd = ImplicitSGD(100, 3, time, lr='adagrad', lr_controls={'eta': 1.0, 'eps': 1e-6})
        theta_new = sgd.update(1, np.zeros(3), D, m, True)
        assert theta_new.shape == (3,)

    def test_update_moves_toward_truth(self):
        D, truth = make_gaussian_problem(n=200, p=3)
        m = GLM(family='gaussian', transfer='identity')
        sgd = ImplicitSGD(200, 3, time, lr='adagrad', lr_controls={'eta': 1.0, 'eps': 1e-6},
                          npasses=20, check=True, truth=truth)
        theta = np.zeros(3)
        for t in range(1, 200 * 20 + 1):
            theta_new = sgd.update(t, theta, D, m, True)
            sgd.sync_members(theta_new)
            if sgd.convergence(theta_new, theta): break
            theta = theta_new
        assert np.mean((theta_new - truth) ** 2) < 0.05
