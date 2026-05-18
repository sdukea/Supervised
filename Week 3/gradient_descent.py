# update g.d. for logistic regression
# explore g.d. on a familiar data set

import copy, math
import numpy as np
import matplotlib.pyplot as plt

# data
X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])

# plot data

fig, ax = plt.subplots(1, 1, figsize=(4,4))

ax.scatter(X_train[:,0], X_train[:,1])
ax.axis([0, 4, 0, 3.5])
ax.set_ylabel("x1", fontsize=12)
ax.set_xlabel("x0", fontsize=12)

plt.show()

def sigmoid(z):

    g_z = 1 / (1 + np.exp(-z))

    return g_z

def compute_grad_terms_logistic(X, y, w, b, sigmoid):

    m, n = X.shape[0]

    dj_dw = np.zeros(n)

    dj_db = 0

    # dj_dw to 0 to start accumulating

    for i in range(m): # for each training example

        f_wb_x_i = sigmoid(np.dot(X[i], w) + b)

        error_i = f_wb_x_i - y[i]

        for j in range(n):

            dj_dw[i] += error_i * X[i, j]

        dj_db += error_i
    
    dj_dw = dj_dw / m
    dj_db = dj_db / m


