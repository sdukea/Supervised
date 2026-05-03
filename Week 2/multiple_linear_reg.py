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

# same motivating (running) example: housing price prediction

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

# features (in order): size, no: of bedrooms, no: of floors, age of home
# target: price of house

# X_train is now a matrix/2D array consisting of 1-D arrays as its values where each 1-D array
# is a training example/row with 4 values -> hence, we have 4 features
# for each training example, we have 1 target
# 3 training examples, each with 4 values/4 features and we have 3 targets for each tr.eg in y_train

# m = 3 (number of rows) - X_train.shape[0]
# n = 4 (number of columns) - X_train.shape[1]

print(f"X shape: {X_train.shape}")
print(f"y shape: {y_train.shape}")

# w is a vector (1D array) with n elements in it, for each feature

# x^(i) is a vector, representing one traininge example, unlike our previous univariate linear 
# reg. example where x^(i) was just a sinlge value as x_train (X_train here as it is a matrix now and
# 'x' is capitalized to 'X') had only one feature

# x^(i)j is a feature value from the x^(i) vector

b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])

# loading w and b that are already near optimal for demonstration 

