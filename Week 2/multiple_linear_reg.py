# extending our linear model to support multiple features

import copy, math
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=2) # reduced display precision on numpy arrays
# what this does: when you print the arrays it rounds the numbers to 2 decimal places
#                 does not change its value; when printing/displaying alone, it shows those numbers
#                 rounded off to two decimal places

"""
a                 → scalar (non-bold)
a (bold)          → vector
A (bold, capital) → matrix

X        → training example matrix          (Python: X_train)
y        → training targets                 (Python: y_train)

x^(i)    → i-th training example            (Python: X[i])
y^(i)    → i-th target                      (Python: y[i])

m        → number of training examples
n        → number of features per example

w        → parameters (weights)             (Python: w)
b        → parameter (bias)                 (Python: b)

f_w,b(x^(i)) → model prediction for i-th example
              = w · x^(i) + b
                                            (Python: f_wb)

"""