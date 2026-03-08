# fastsgd

A Python library implementing Stochastic Gradient Descent (SGD) for parameter
estimation with large datasets, including the statistically efficient
**Implicit SGD** algorithm of Toulis & Airoldi (2017).

## Overview

Two SGD variants are provided:

| Class | Description |
|---|---|
| `ImplicitSGD` | Proximal (implicit) SGD — achieves the Cramér-Rao efficiency bound for GLMs |
| `ExplicitSGD` | Standard (explicit) SGD — simpler update rule, required for M-estimators |

Both operate on **Generalized Linear Models** (`GLM`) and **M-estimators** (`MModel`).

## Quick start

### 1. Prepare data

```python
import numpy as np
from fastsgd import DataSet

X = np.random.randn(1000, 5)
Y = X @ np.ones(5) + np.random.randn(1000)
D = DataSet(X, Y)
```

### 2. Define a model

```python
from fastsgd import GLM, MModel

# Generalized Linear Model
# family:   "gaussian" | "poisson" | "gamma" | "binomial"
# transfer: "identity" | "exponential" | "inverse" | "logistic"
m = GLM(family="gaussian", transfer="identity")

# Robust M-estimator with Huber loss (use ExplicitSGD for M-estimators)
# m = MModel(loss="huber", threshold=1.5)
```

### 3. Fit with SGD

```python
import time
from fastsgd import ImplicitSGD

n, p = D._n, D._p
sgd = ImplicitSGD(n, p, time,
                  lr="adagrad",
                  lr_controls={"eta": 1.0, "eps": 1e-6},
                  npasses=20,
                  reltol=1e-4)

theta = np.zeros(p)
for t in range(1, n * sgd._n_passes + 1):
    theta_new = sgd.update(t, theta, D, m, True)
    sgd.sync_members(theta_new)
    if sgd.convergence(theta_new, theta):
        break
    theta = theta_new

print(theta_new)   # estimated parameters
```

### 4. Estimate standard errors

```python
from fastsgd import FisherCovariance, SandwichCovariance

# Fisher information — assumes correctly specified model
se = FisherCovariance().std_errors(theta_new, D, m)

# Sandwich / Huber-White — robust to model misspecification
se_robust = SandwichCovariance().std_errors(theta_new, D, m)
```

## Learning rate schedules

The `lr` kwarg selects the schedule; `lr_controls` provides its hyperparameters.

| `lr` | Formula | `lr_controls` keys |
|---|---|---|
| `"one-dim"` | `scale · γ · (1 + α·γ·t)^{-c}` | `scale`, `alpha`, `gamma`, `c` |
| `"one-dim-eigen"` | `d / (‖∇‖² · t)` | — |
| `"d-dim"` | per-parameter accumulator | `eps` |
| `"adagrad"` | `η / (Σ∇²ᵢ + ε)^{0.5}` | `eta`, `eps` |
| `"rmsprop"` | `η / (EWMA(∇²ᵢ) + ε)^{0.5}` | `eta`, `gamma`, `eps` |

## Covariance estimation

Two asymptotic covariance estimators are provided. Both return `V` such that
`sqrt(n) (θ̂ − θ₀) → N(0, V)`, so `Var(θ̂) ≈ V / n`.

```python
from fastsgd import FisherCovariance, SandwichCovariance

V  = FisherCovariance().estimate(theta, D, m)      # V = I(θ)⁻¹
Vs = SandwichCovariance().estimate(theta, D, m)    # V = A⁻¹ B A⁻¹
se = FisherCovariance().std_errors(theta, D, m)    # sqrt(diag(V / n))
```

**Key result** (Toulis & Airoldi, 2017): ImplicitSGD achieves the Cramér-Rao
bound, so its asymptotic variance equals `I(θ)⁻¹ / n`. ExplicitSGD does not
achieve this bound in general.

## Polyak-Ruppert averaging

`averaged_estimate()` returns the mean of the log-uniformly spaced parameter
snapshots stored during `sync_members` calls. This can reduce asymptotic
variance when using a non-optimal learning rate.

```python
theta_avg = sgd.averaged_estimate()
```

## Supported models

### GLM families and canonical links

| Family | Transfer | Use case |
|---|---|---|
| `"gaussian"` | `"identity"` | Linear regression |
| `"binomial"` | `"logistic"` | Logistic / binary regression |
| `"poisson"` | `"exponential"` | Count data |
| `"gamma"` | `"inverse"` | Positive continuous responses |

### M-estimators

| Loss | Parameter | Description |
|---|---|---|
| `"huber"` | `l` (threshold) | Robust regression; use `ExplicitSGD` |

## Installation

```bash
git clone <repo-url>
cd fastsgd
uv pip install -e ".[dev]"   # or: pip install -e ".[dev]"
```

## Development

```bash
make test         # run the test suite
make build        # build a wheel
make clear-cache  # remove __pycache__ and build artefacts
```

## Author

Written and maintained by **George Vassos**, based on work by
[Dustin Tran](http://dustintran.com/) and [Panos Toulis](https://www.ptoulis.com/)
on the efficient implementation of implicit SGD.

George Vassos
Associate Research Scientist, A.P. Moller Maersk A/S
Esplanaden 50, 1098 Copenhagen K, Denmark

## License

Apache License 2.0 — see [LICENSE](LICENSE).

## References

[1] Toulis, P. and Airoldi, E. M. (2017).
Asymptotic and finite-sample properties of estimators based on stochastic gradients.
*Annals of Statistics*, 45(4), 1694–1727.

[2] Tran, D., Toulis, P. and Airoldi, E. M. (2015).
Stochastic gradient descent methods for estimation with large data sets.
*arXiv*, preprint arXiv:1509.06459.

## Citation

```bibtex
@article{toulis2017,
  title   = {Asymptotic and finite-sample properties of estimators based on stochastic gradients},
  author  = {Toulis, Panos and Airoldi, Edoardo M.},
  journal = {Annals of Statistics},
  volume  = {45},
  number  = {4},
  pages   = {1694--1727},
  year    = {2017},
  doi     = {10.1214/16-AOS1506}
}
```
