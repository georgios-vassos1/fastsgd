"""
Asymptotic covariance estimators for SGD-based parameter estimates.

For an estimating equation (1/n) Σ ψ(xᵢ, yᵢ, θ) = 0, the central limit
theorem gives √n (θ̂ − θ₀) → N(0, V) where

    V  = A⁻¹ B A⁻¹         (sandwich)
    A  = E[∂ψ/∂θ]           (expected Hessian, estimated by (1/n) Xᵀ diag(w) X)
    B  = E[ψ ψᵀ]            (outer product of scores, estimated by (1/n) GᵀG)

The asymptotic variance of θ̂ itself is therefore V/n, and standard errors are
    se = sqrt(diag(V / n))

A key result of Toulis & Airoldi (2017) is that Implicit SGD achieves the
Cramér-Rao bound, so its asymptotic variance equals I(θ)⁻¹/n (the Fisher
information bound). Explicit SGD does not achieve this bound in general.

Two estimators are provided:

    FisherCovariance   — V = A⁻¹ = I(θ̂)⁻¹
                         Assumes the model is correctly specified.
                         Appropriate for GLMs; not meaningful for M-estimators.

    SandwichCovariance — V = A⁻¹ B A⁻¹
                         Robust to model misspecification.
                         Valid for both GLMs and M-estimators (e.g. Huber).
"""
from abc import ABC, abstractmethod
import numpy as np
from .utils import DataSet
from .model import Model

__all__ = ['CovarianceEstimator', 'FisherCovariance', 'SandwichCovariance']


class CovarianceEstimator(ABC):
    @abstractmethod
    def estimate(self, theta: np.ndarray, data: DataSet, model: Model) -> np.ndarray:
        """Return the asymptotic covariance matrix V (p × p).

        √n (θ̂ − θ₀) → N(0, V), so Var(θ̂) ≈ V / n.
        """
        pass

    def std_errors(self, theta: np.ndarray, data: DataSet, model: Model) -> np.ndarray:
        """Return asymptotic standard errors: sqrt(diag(V / n))."""
        V = self.estimate(theta, data, model)
        return np.sqrt(np.diag(V) / data._n)


class FisherCovariance(CovarianceEstimator):
    """V = I(θ̂)⁻¹ — inverse of the observed Fisher information.

    Valid only when the model is correctly specified. For GLMs this gives the
    same asymptotic variance as maximum likelihood estimation. Not meaningful
    for M-estimators where no likelihood is assumed.
    """

    def estimate(self, theta: np.ndarray, data: DataSet, model: Model) -> np.ndarray:
        w = model.hessian_weights(data, theta)          # (n,)
        A = (data._X.T * w) @ data._X / data._n        # (p, p)
        return np.linalg.inv(A)


class SandwichCovariance(CovarianceEstimator):
    """V = A⁻¹ B A⁻¹ — the sandwich (Huber-White) estimator.

    Robust to model misspecification. Valid for both GLMs and M-estimators.
    Reduces to FisherCovariance when the model is correctly specified (B = A).
    """

    def estimate(self, theta: np.ndarray, data: DataSet, model: Model) -> np.ndarray:
        n = data._n
        G = model.score_matrix(data, theta)             # (n, p)
        w = model.hessian_weights(data, theta)          # (n,)
        A = (data._X.T * w) @ data._X / n              # (p, p)
        B = G.T @ G / n                                 # (p, p)
        A_inv = np.linalg.inv(A)
        return A_inv @ B @ A_inv
