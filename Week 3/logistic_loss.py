import numpy as np
import matplotlib.pyplot as plt

# see why sq. error loss function is not appropriate for logistic regression
# exploring the logistic loss function

# get data
x_train = np.array([0, 1, 2, 3, 4, 5], dtype=np.longdouble)
y_train = np.array([0,0,0,1,1,1])

# why MSE cost function won't do good for log. reg.

def sigmoid(z):

    g = 1 / (1+np.exp(-z))

    return g

