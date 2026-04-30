# implementing gradient descent

# use the same running example - predicting house prices
# (intuition remains the same for x_train, y_train and all)

import numpy as np
import matplotlib.pyplot as plt
import math, copy

x_train = np.array([1.0, 2.0]) # one feature - 2 training example/values/data points
y_train = np.array([300.0, 500.0]) # for each value, a label

def compute_cost(x, y, w, b):

    m = x.shape[0]

    total_cost = 0

    for i in range(m):
        f_wb = w * x[i] + b

        cost = (f_wb - y) ** 2

        total_cost = total_cost + cost

    final_cost = (1 / (2 * m)) * total_cost

    return final_cost

# returns final cost

def compute_gradient_terms(x, y, w, b):
    
    m = x.shape[0]

    # exactly what we're computing
    dj_dw = 0 # gradient term for w update
    dj_db = 0 # gradient term for b update
    # start accumulating both these now to return it

    for i in range(m):
        
        f_wb = w * x[i] + b
        # model prediction - y-hat/estimate (as you know it)

        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = (f_wb - y[i])

        dj_dw += dj_dw_i
        dj_db += dj_db_i
    
    dj_dw = dj_dw / m
    dj_db = dj_db / m

# returns the gradient terms to implement g.d.