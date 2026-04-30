# implementing gradient descent

# use the same running example - predicting house prices
# (intuition remains the same for x_train, y_train and all)

import numpy as np
import matplotlib.pyplot as plt
import math, copy

x_train = np.array([1.0, 2.0]) # one feature - 2 training example/values/data points
y_train = np.array([300.0, 500.0]) # for each value, a label

def compute_cost(x, y, w, b):

    m = x_train.shape[0]

    total_cost = 0

    for i in range(m):
        f_wb = w * x[i] + b

        cost = (f_wb - y) ** 2

        total_cost = total_cost + cost

    final_cost = (1 / (2 * m)) * total_cost

    return final_cost

