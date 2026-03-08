import time
from abc import ABC, abstractmethod
import numpy as np
from .learning_rate import *

__all__ = ['SGD']


class SGD(ABC):
    """Base class for SGD optimisers. Use ImplicitSGD or ExplicitSGD directly.

    Parameters
    ----------
    n : int
        Number of observations in the dataset.
    p : int
        Number of parameters.
    timer : time module
        Kept for API compatibility; not currently used internally.

    Keyword arguments
    -----------------
    npasses : int, default 10
        Maximum number of full passes over the data.
    reltol : float, default 1e-5
        Convergence tolerance. Interpretation depends on ``check``:
        - check=False: mean absolute relative change in theta.
        - check=True:  mean squared error against ``truth``.
    size : int, default 10
        Number of parameter snapshots stored, spaced log-uniformly over all
        iterations. Accessible via ``_estimates`` (shape p × size).
    lr : str, default 'one-dim'
        Learning rate schedule. One of:
        - 'one-dim'       — polynomial decay: scale * gamma * (1 + alpha*gamma*t)^{-c}
        - 'one-dim-eigen' — eigenvalue-adaptive scalar rate
        - 'd-dim'         — per-parameter accumulator (AdaGrad-like, c=1)
        - 'adagrad'       — AdaGrad diagonal: eta / (sum_grad² + eps)^0.5
        - 'rmsprop'       — RMSProp diagonal: eta / (ewma_grad² + eps)^0.5
    lr_controls : dict
        Hyperparameters for the chosen schedule:
        - 'one-dim':  scale (float), gamma (float), alpha (float), c (float)
        - 'd-dim':    eps (float)
        - 'adagrad':  eta (float), eps (float)
        - 'rmsprop':  eta (float), gamma (float, decay), eps (float)
    check : bool, default False
        If True, ``convergence()`` measures MSE against ``truth`` instead of
        the relative parameter change. Intended for simulations only.
    truth : np.ndarray, optional
        True parameter vector used when ``check=True``.
    """
    def __init__(self, n: int, p: int, timer: time, **kwargs):
        self._name = kwargs.get("method", None)
        self._n_params = p
        self._reltol = kwargs.get("reltol", 1e-5)   # relative tolerance for convergence
        self._n_passes = kwargs.get("npasses", 10)  # number of passes over data
        self._size = kwargs.get("size", 10)          # number of estimates to be recorded (log-uniformly)
        self._estimates = np.zeros((self._n_params, self._size))
        self._last_estimate = np.zeros(self._n_params)
        self._timer = timer
        self._t = 0
        self._n_recorded = 0              # number of coefs that have been recorded
        self._pos = np.zeros(self._size)  # the iteration of recorded coefs
        self._pass = kwargs.get("pass", True)  # force running for n_passes on data
        self._good_gradient = True
        self._check = kwargs.get("check", False)
        if self._check:
            self._truth = kwargs.get("truth", None)

        ## Select the iterations to store estimates
        n_iters = n * self._n_passes
        self._pos = (10.0 ** (np.arange(self._size) * np.log10(float(n_iters)) / (self._size - 1))).astype(int)
        if self._pos[-1] != n_iters:
            self._pos[-1] = n_iters

        ## Set learning rate
        self._lr_choice = kwargs.get("lr", "one-dim")
        controls = kwargs.get("lr_controls", {"scale": 1.0, "alpha": 1.0, "gamma": 0.6, "c": 0.5})
        if self._lr_choice == "one-dim":
            self._lr_obj = OnedimLR(controls["scale"], controls["gamma"], controls["alpha"], controls["c"])
        elif self._lr_choice == "one-dim-eigen":
            self._lr_obj = OnedimEigLR(self._n_params)
        elif self._lr_choice == "d-dim":
            self._lr_obj = DDimLR(self._n_params, 1.0, 0.0, 1.0, 1.0, controls["eps"])
        elif self._lr_choice == "adagrad":
            self._lr_obj = DDimLR(self._n_params, controls["eta"], 1.0, 1.0, 0.5, controls["eps"])
        elif self._lr_choice == "rmsprop":
            self._lr_obj = DDimLR(self._n_params, controls["eta"], controls["gamma"], 1.0 - controls["gamma"], 0.5, controls["eps"])
        else:
            raise ValueError(
                f"Unknown learning rate '{self._lr_choice}'. "
                "Choose from: 'one-dim', 'one-dim-eigen', 'd-dim', 'adagrad', 'rmsprop'."
            )

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(p={self._n_params}, "
                f"lr='{self._lr_choice}', npasses={self._n_passes})")

    @abstractmethod
    def update(self, t: int, theta_old: np.ndarray, data, model, good_gradient: bool) -> np.ndarray:
        """Compute one SGD step and return the updated parameter vector."""
        pass

    def get_value_of(self, attribute: str):
        try:
            return self.__dict__["_" + attribute]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{attribute}'")

    def convergence(self, theta_new: np.ndarray, theta_old: np.ndarray) -> bool:
        """Return True if the stopping criterion is satisfied.

        When ``check=True`` (simulation mode): MSE(theta_new, truth) < reltol.
        Otherwise: mean |theta_new - theta_old| / mean |theta_old| < reltol.
        """
        if self._check:
            if np.mean((theta_new - self._truth) ** 2) < self._reltol:
                return True
        else:
            denom = np.mean(np.abs(theta_old))
            if denom == 0.0:
                return False
            if np.mean(np.abs(theta_new - theta_old)) / denom < self._reltol:
                return True
        return False

    def sync_members(self, theta_new: np.ndarray):
        """Record theta_new and advance the iteration counter.

        Snapshots are stored in ``_estimates`` at log-uniformly spaced
        iterations so that convergence can be inspected post-hoc.
        """
        self._last_estimate = theta_new
        self._t += 1
        if self._t == self._pos[self._n_recorded]:
            self._estimates[:, self._n_recorded] = theta_new
            self._n_recorded += 1
            while (self._n_recorded < self._size) and (self._pos[self._n_recorded - 1] == self._pos[self._n_recorded]):
                self._estimates[:, self._n_recorded] = theta_new
                self._n_recorded += 1

    def averaged_estimate(self) -> np.ndarray:
        """Return the Polyak-Ruppert average of stored parameter snapshots.

        Averages the log-uniformly spaced snapshots recorded by
        ``sync_members``. This can have lower asymptotic variance than the
        final iterate when using a non-optimal (e.g. polynomial-decay) learning
        rate. See Polyak & Juditsky (1992).

        Returns ``_last_estimate`` (the most recent iterate) if no snapshots
        have been recorded yet.
        """
        if self._n_recorded == 0:
            return self._last_estimate.copy()
        return self._estimates[:, :self._n_recorded].mean(axis=1)

    def _learning_rate(self, t: int, grad_t: np.ndarray) -> LRvalue:
        return self._lr_obj(t, grad_t)
