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

        cost = (f_wb - y[i]) ** 2

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
        # because
        # 1. derivative term d/dw J(w,b) gives this w/ summation from i = 1 to m (the for loop here)
        # 2. because the der. term is form i = 1 through m, you need dj/dw 1 through m
        dj_db_i = (f_wb - y[i])
        # because
        # (same as above but for b)

        dj_dw += dj_dw_i
        dj_db += dj_db_i
    
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db

# returns the gradient terms to implement g.d.

def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_func, gradient_func):

    # this performs g.d. to fit w and b such that it minimizes c.f with handled/computed
    # gradient terms

    # returns: w, b, J_history (history of cost values), p_history (history of parameters w and b)

    J_history = []
    p_history = []
    w = w_in
    b = b_in

    for i in range(num_iters): # iterate/repeat until num_iters (if it hopefully leads to convergence)

        dj_dw, dj_db = gradient_func(x, y, w, b) # get gradient terms here

        # update

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # save cost J at each iteration

        J_history.append(cost_func(x, y, w, b))
        p_history.append([w, b])

        # print cost at intervals of 10 based on number of iterations
        # if num_iters = 1000, print at every 100th interval so that you print
        # cost at 10 intervals evenly (1000/10 = 100)
       
        if i % math.ceil(num_iters / 10) == 0:
            # if the current iteration is one of the 10 intervals split and spaced from
            # the num of iterations (because we need it to print when it is only from the 10 intervals)

            print(f"Iteration {i}: Cost {J_history[-1]} ",
                  f"dj_dw: {dj_dw}, dj_db: {dj_db}  ",
                  f"w: {w}, b:{b}") 
    
    return w, b, J_history, p_history # return all these for graphing

# initialize parameters

w_init = 0
b_init = 0

# start with w and b by randomly guessing - yes
# but w and b = 0 are usually what we start with

iterations = 10000

tmp_alpha = 1.0e-2

w_final, b_final, J_hist, p_hist = gradient_descent(x_train, y_train, w_init, b_init, tmp_alpha,
                                                    iterations, compute_cost, compute_gradient_terms)

print(w_final, b_final)