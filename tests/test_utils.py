import numpy as np
import pytest
from fastsgd.utils import DataPoint, DataSet, data_point, data_set


class TestDataPoint:
    def test_attributes(self):
        x = np.array([1.0, 2.0])
        dp = DataPoint(x, 3.5, 7)
        assert np.array_equal(dp._x, x)
        assert dp._y == 3.5
        assert dp._i == 7


class TestDataSet:
    def setup_method(self):
        self.X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        self.Y = np.array([0.1, 0.2, 0.3])
        self.D = DataSet(self.X, self.Y)

    def test_shape_attributes(self):
        assert self.D._n == 3
        assert self.D._p == 2
        assert np.array_equal(self.D._X, self.X)
        assert np.array_equal(self.D._Y, self.Y)

    def test_get_data_point_first(self):
        dp = self.D.get_data_point(1)
        assert np.array_equal(dp._x, self.X[0])
        assert dp._y == self.Y[0]

    def test_get_data_point_last(self):
        dp = self.D.get_data_point(3)
        assert np.array_equal(dp._x, self.X[2])
        assert dp._y == self.Y[2]

    def test_get_data_point_cyclic(self):
        # t = n+1 should wrap back to index 0
        dp_wrap = self.D.get_data_point(self.D._n + 1)
        dp_first = self.D.get_data_point(1)
        assert np.array_equal(dp_wrap._x, dp_first._x)
        assert dp_wrap._y == dp_first._y

    def test_get_data_point_returns_datapoint(self):
        dp = self.D.get_data_point(1)
        assert isinstance(dp, DataPoint)


class TestDataSetValidation:
    def test_1d_X_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            DataSet(np.array([1.0, 2.0]), np.array([1.0, 2.0]))

    def test_2d_Y_raises(self):
        with pytest.raises(ValueError, match="1-D"):
            DataSet(np.array([[1.0, 2.0]]), np.array([[1.0]]))

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same number"):
            DataSet(np.ones((3, 2)), np.ones(4))

    def test_integer_arrays_coerced_to_float(self):
        X = np.array([[1, 2], [3, 4]])
        Y = np.array([1, 2])
        D = DataSet(X, Y)
        assert D._X.dtype == np.float64
        assert D._Y.dtype == np.float64


class TestBackwardAliases:
    def test_data_point_alias(self):
        assert data_point is DataPoint

    def test_data_set_alias(self):
        assert data_set is DataSet
