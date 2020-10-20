# An efficient implementation of Stochastic Gradient Descent (SGD) procedures for estimation with big data
This repository hosts a fast python library implementing stochastic
approximation methods for parameter estimation with large data sets. 

The traditional SGD and Implicit SGD methods, the latter presented in [[1]](#1), 
are both included to be used in the estimation of Generalized Linear Models
(GLMs), or in M-estimation. 

## Features

### Use the data_set object
Form a design matrix, e.g. X, with your covariates and a vector of your response, e.g. Y, 
and create a data\_set object with
```python
from fastsgd import data_set
D = data_set(X, Y)
```
You can create a glm object by providing its family and transfer function names,
as well as regularization parameters if your design matrix is sparse, or you
think your variables are highly correlated. 
```python
from fastsgd import glm
# family: "gaussian", "poisson", "gamma", or "binomial"
# transfer: "identity", "exponential", "inverse", or "logistic"
m = glm(family="gaussian", transfer="identity", lambda1=0.0, lambda2=0.0)
```
Setting both lambda1 and lambda2 arguments to 0.0 entails no regularization. 

You can also create an m\_model object.
```python
from fastsgd import m_model
m = m_model(loss="huber", lambda1=0.0, lambda2=0.0)
```
Next, you can create either an ExplicitSGD (traditional SGD method), or an
ImplicitSGD object. Below you can find an example with default values for the
various arguments of the ImplicitSGD object.
```python
from fastsgd import ImplicitSGD
import time

## standard arguments
timer = time
n, p = D._X.shape

## Many additional arguments
# - learning rate (lr): "one-dim", "one-dim-eigen", "d-dim", "adagrad", "rmsprop"
# - learning rate controls (lr_controls):
#     if lr == "one-dim", then lr_controls = { "scale": 1.0, "alpha": 1.0, "gamma": 0.6, "c": 0.5 }
#     if lr == "one-dim-eigen", then lr_controls = None
#     if lr == "d-dim", then lr_controls = { "eps": 1e-6  }
#     if lr == "adagrad", then lr_controls = { "eta": 1.0, "eps": 1e-6  }
#     if lr == "rmsprop", then lr_controls = { "eta": 1.0, "gamma": 0.6, "eps": 1e-6  }
# - error tolerance (reltol): 5e-4
# - number of passes (npasses): 20
# - check: True if the true value of the parameter is know, else Fasle
# - truth: if check is True, then provide the vector of true parameters as an argument

details = {
    "lr": "adagrad",
    "lr_controls": {
        "eta": 1.0,
        "eps": 1e-6
    },
    "reltol": 5e-4,
    "npasses": 20,
    "check": True,
    "truth": theta
}

tester = ImplicitSGD(n, p, timer, **details)
```
At this point you are ready to trigger the implicit iteration. To do that you
can use a number of auxiliary variables that can keep track of the validity of
the procedure in case you are not sure whether the argument values that you are
using is proper.
```python
n_passes = tester.get_value_of("n_passes")
good_gradient = True # Will shift to False if the gradient blows up
averaging = False # Use averaging to ensure statistical efficiency
theta_old = np.ones(p) / 100.0 # Random initial values
theta_old_ave = theta_old # Do the same if you are averaging
max_iters = n * n_passes
converged = False

t = 1

while True:
    theta_new = tester.update(t, theta_old, D, m, good_gradient)
    if not averaging:
        tester.sync_members(theta_new)
        converged = tester.convergence(theta_new, theta_old)
    else:
        theta_new_ave = 0.5 * theta_old_ave + 0.5 * theta_new
        tester.sync_members(theta_new_ave)
        converged = tester.convergence(theta_new_ave, theta_old_ave)
        theta_old_ave = theta_new_ave
    if converged: break
    theta_old = theta_new
    if t == max_iters: break
    t += 1
```
If you have a well-defined problem, then the above loop should at some point
stop and you will be able to verify the final iterate by checking the values of
```python
converged, theta_new
```

## How to simulate data
Below we present a code snippet that can be used to generate a 1500x20 design matrix. 
```python
import numpy as np
from fastsgd import data_set

def simulate_normal_data(theta: np.ndarray, size: tuple=(1500, 20)) -> data_set:
    N, p = size
    # Create the covariance matrix with values between 0.5 and 5
    S = np.eye(p) * np.random.uniform(0.5, 5, p)
    # Generate the design matrix
    X = np.random.multivariate_normal(mean=np.zeros(p), cov=S, size=N)
    # Generate the response vector
    Y = np.random.normal(X @ theta, 1, size=N)
    return data_set(X, Y)
```

## Installation
At the moment the library is not uploaded on PyPI. Hence, to install the package
you have to 
* clone the repository in your `root` directory
* run `pip install -e /sgd/`


## Author
This package is written and maintained by George Vassos and is heavily based on
the work done by <a href="http://dustintran.com/">Dustin Tran</a> and <a href="https://www.ptoulis.com/">Panos Toulis</a> 
on the efficient implementaion of the implicit SGD methods [[2]](#2).

George Vassos<br/>
Associate Research Scientist at A.P. Moller Maersk A/S<br/>
Esplanaden 50, 1098 Copenhagen K, Denmark



## References
<a href="#1">[1]</a> 
Toulis, P. and Airoldi, E. M. (2017).
Asymptotic and finite-sample properties of estimators based on stochastic gradients.
*Annals of Statistics*, 45(4), 1694-1727.

<a href="#2">[2]</a> 
Tran, D., Toulis, P. and Airoldi, E. M. (2015).
Stochastic gradient descent methods for estimation with large data sets.
*arXiv*, preprint arXiv:1509.06459.

## Citation
```
@article{toulis2017a,
  title = {Asymptotic and finite-sample properties of estimators based on stochastic gradients},
  language = {eng},
  publisher = {The Institute of Mathematical Statistics},
  journal = {Annals of Statistics},
  volume = {45},
  number = {4},
  pages = {1694-1727},
  year = {2017},
  issn = {21688966, 00905364},
  doi = {10.1214/16-AOS1506},
  author = {Toulis, Panos and Airoldi, Edoardo M.}
}
```
