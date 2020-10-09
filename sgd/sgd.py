import os, time
from functools import partial
from scipy.optimize import brentq
import pandas as pd
import numpy as np
from .learning_rate.__init__ import *


class SGD:
    def __init__(self, n_samples: int, timer: time, **details):
        self._name = details["method"]
        self._n_params = details["nparams"]
        self._reltol = details["reltol"]    # relative tolerance for convergence
        self._n_passes = details["npasses"] # number of passes over data
        self._size = details["size"]        # number of estimates to be recorded (log-uniformly)
        self._estimates = np.zeros((self._n_params, self._size))
        self._last_estimate = np.zeros(self._n_params)
        self._times = np.zeros(self._size)
        self._timer = timer
        self._t = 0
        self._n_recorded = 0              # number of coefs that have been recorded
        self._pos = np.zeros(self._size)  # the iteration of recorded coefs
        self._pass = details["pass"]      # force running for n_passes on data
        self._check = details["check"]

        if self._check: self._truth = details["truth"]

        ## Select the iterations to store estimates
        n_iters = n_samples * self._n_passes
        self._pos = (10.0 ** (np.arange(self._size) * np.log10(float(n_iters)) / (self._size-1))).astype(int)
        if self._pos[-1] != n_iters: self._pos[-1] = n_iters

        ## Set learning rate
        self._lr_choice = details["lr"]     # type of learning rate: 'one-dim', 'one-dim-eigen', 'd-dim', 'adagrad', 'rmsprop'
        controls = details["lr_controls"]
        if self._lr_choice == "one-dim":
            self._lr_obj = OnedimLR(controls["scale"], controls["gamma"], controls["alpha"], controls["c"])
        elif self._lr_choice == "one-dim-eigen":
            self._lr_obj = OnedimEigLR(self._n_params)
        elif self._lr_choice == "d-dim":
            self._lr_obj = ddimLR(self._n_params, 1.0, 0.0, 1.0, 1.0, controls["eps"])
        elif self._lr_choice == "adagrad":
            self._lr_obj = ddimLR(self._n_params, controls["eta"], 1.0, 1.0, 0.5, controls["eps"])
        elif self._lr_choice == "rmsprop":
            self._lr_obj = ddimLR(self._n_params, controls["eta"], controls["a"], 1.0 - controls["a"], 0.5, controls["eps"])


    def get_value_of(self, attribute: str):
        try: return self.__dict__["_" + attribute]
        except KeyError as e: print(attribute + " is not an attribute of the caller.")

    def convergence(self, theta_new: np.ndarray, theta_old: np.ndarray) -> bool:
        if self._check:
            qe = np.mean((theta_new - self._truth) ** 2)
            # print(qe)
            if qe < self._reltol: return True
        elif not self._pass:
            qe = np.mean(np.mean(np.abs(theta_new - theta_old))) / np.mean(np.mean(np.abs(theta_old)))
            if qe < self._reltol: return True
        return False

    def sync_members(self, theta_new: np.ndarray):
        self._last_estimate = theta_new
        self._t += 1
        if self._t == self._pos[self._n_recorded]:
            self._estimates[:, self._n_recorded] = theta_new
            ## TODO record elapsed time
            self._n_recorded += 1
            while (self._n_recorded < self._size) and (self._pos[self._n_recorded-1] == self._pos[self._n_recorded]):
                self._estimates[:, self._n_recorded] = theta_new
                ## TODO record elapsed time
                self._n_recorded += 1

    def early_stop(self):
        pass

    def _learning_rate(self, t: int, grad_t: np.ndarray) -> LRvalue:
        return self._lr_obj(t, grad_t)


