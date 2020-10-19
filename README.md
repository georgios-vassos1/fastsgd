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
D = data_set(X, Y)
```
You can create a glm object by providing its family and transfer function names,
as well as regularization parameters if your design matrix is sparse, or you
think your variables are highly correlated. 
```python
# family: "gaussian", "poisson", "gamma", or "binomial"
# transfer: "identity", "exponential", "inverse", or "logistic"
m = glm(family="gaussian", transfer="identity", lambda1=0.0, lambda2=0.0)
```
Setting both lambda1 and lambda2 arguments to 0.0 entails no regularization. 

You can also create an m\_model object.
```python
m = m_model(loss="huber", lambda1=0.0, lambda2=0.0)
```
Next, you can create either an ExplicitSGD (traditional SGD method), or an
ImplicitSGD object. 
```python
from sgd import *
import time

## standard arguments
timer = time
n, p = D._X.shape

## Many additional arguments
# - learning rate (lr): "one-dim", "one-dim-eigen", "d-dim", "adagrad", "rmsprop"
# - learning rate controls (lr_controls):
#       if lr == "one-dim", then lr_controls = { "scale": 1.0, "alpha": 1.0, "gamma": 0.6, "c": 0.5 }
#       if lr == "one-dim-eigen", then lr_controls = None
#       if lr == "d-dim", then lr_controls = { "eps": 1e-6  }
#       if lr == "adagrad", then lr_controls = { "eta": 1.0, "eps": 1e-6  }
#       if lr == "rmsprop", then lr_controls = { "eta": 1.0, "gamma": 0.6, "eps": 1e-6  }
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


## How to simulate data


## Installation


## Author


## References
<a href="#1">[1]</a> 
Toulis, Panos and Airoldi, Edoardo M. (2017).
Asymptotic and finite-sample properties of estimators based on stochastic gradients.
*Annals of Statistics*, 45(4), 1694-1727.

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
