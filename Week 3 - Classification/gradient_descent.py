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

def sigmoid(z):

    g_z = 1 / (1 + np.exp(-z))

    return g_z

def compute_grad_terms_logistic(X, y, w, b, sigmoid):

    m, n = X.shape

    dj_dw = np.zeros(n)

    dj_db = 0

    # dj_dw to 0 to start accumulating

    for i in range(m):

        f_wb_x_i = sigmoid(np.dot(X[i], w) + b)

        error_i = f_wb_x_i - y[i]

        for j in range(n):

            dj_dw[j] += error_i * X[i, j]

        dj_db += error_i
    
    # for each tr. eg.
    # 1. compute pred.
    # 2. find error
    # 3. multiply error with each feature value in tr. eg.
    # 4. accumulate in dj_dw by position for each w (dj_dw1 for w1, dj_dw2 for w2 and so on)
    # 5. finally, accumulate dj_db for error
    

    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_db, dj_dw


# check implementation
X_tmp = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_tmp = np.array([0, 0, 0, 1, 1, 1])
w_tmp = np.array([2.,3.])
b_tmp = 1.

dj_db_tmp, dj_dw_tmp = compute_grad_terms_logistic(X_tmp, y_tmp, w_tmp, b_tmp, sigmoid=sigmoid)

print(f"dj_db: {dj_db_tmp}" )
print(f"dj_dw: {dj_dw_tmp.tolist()}" )

# do g.d.

def gradient_descent(X, y, w_initial, b_initial, alpha, num_iters):

    cost_history = []

    w = copy.deepcopy(w_initial)

    b = b_initial   

    for i in range(num_iters):

        dj_db, dj_dw = compute_grad_terms_logistic(X, y, w, b, sigmoid=sigmoid)

        w = w - alpha * dj_dw

        b = b - alpha * dj_db

    return w, b


w_tmp = np.zeros_like(X_train[0])
b_tmp = 0.

alph = 0.1
iters = 10000

w_out, b_out = gradient_descent(X_train, y_train, w_tmp, b_tmp, alph, iters)

print(f"Final parameters; w = {w_out}, b = {b_out}")

# plot results of gradient descent

fig, ax = plt.subplots(figsize=(5,4))

pos = y_train == 1
neg = y_train == 0

# class 0 points
ax.scatter(
    X_train[neg][:,0],
    X_train[neg][:,1],
    marker='o',
    label='y=0'
)

# class 1 points
ax.scatter(
    X_train[pos][:,0],
    X_train[pos][:,1],
    marker='x',
    label='y=1'
)

plt.show()