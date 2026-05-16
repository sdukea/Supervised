# examine implementation and utilize c.f. for logistic regression

import numpy as np
import matplotlib.pyplot as plt

X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])  #(m,n)
y_train = np.array([0, 0, 0, 1, 1, 1])                                           #(m,)

# so two features; X_train is a 2D matrix

# plot data

pos = y_train == 1
neg = y_train == 0

fig, ax = plt.subplots(1, 1, figsize=(5, 3))

ax.scatter(X_train[:,0], X_train[:,1])

ax.set_ylabel('y', fontsize=12)
ax.set_xlabel('x', fontsize=12)

ax.axis([0, 4, 0, 4])

# compute cost for each training eg: loss

def compute_cost_logistic(x, y, w, b, sig_func):

    m = x.shape[0]
    # get rows


    cost = 0.0

    for i in range(m): # for each training example

        z_i = np.dot(x[i], w) + b
        # raw, >1/<0 prediction for each tr. eg.

        f_wb_i = sig_func(z_i)
        # convert to probabilistic value showing how the tr. eg. is equal to y = 1

        cost += -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)

    cost = cost / m

    return cost

ax.legend()

plt.show()