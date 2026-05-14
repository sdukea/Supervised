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

def compute_cost(x, y, w, b):

    m,n = x.shape

    total_cost = 0

    for i in range(m):
        f_wb_x_i_nv = w * x[i] + b

        cost = (f_wb_x_i_nv - y[i])**2

        total_cost += cost
    
    total_cost = total_cost / (2*m)

    return total_cost

def compute_grad_terms(x, y, w, b):

    dj_dw = 0
    dj_db = 0

    m, n = x.shape

    for i in range(m):

        pred = w * x[i] + b

        d_dw_j_i = (pred - y[i])**2

        d_dw_b_i = pred

        dj_dw += d_dw_j_i
        dj_db += d_dw_b_i
    
    dj_dw /= (2*m)
    dj_db /= (2*m)

    return dj_dw, dj_db

def gd(x, y, w_in, b_in, alpha, num_iters, cost_func, grad_func):

    J_history = []
    p_history = []

    w = w_in
    b = b_in