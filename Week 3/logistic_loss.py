import numpy as np
import matplotlib.pyplot as plt

# see why sq. error loss function is not appropriate for logistic regression
# exploring the logistic loss function

# get data
x_train = np.array([0, 1, 2, 3, 4, 5], dtype=np.longdouble)
y_train = np.array([0,0,0,1,1,1])

# why MSE cost function won't do good for log. reg.

import numpy as np

x_train = np.array([0,1,2,3,4,5], dtype=np.longdouble)
y_train = np.array([0,0,0,1,1,1])

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(x, y, w, b):

    m = len(x)

    total_cost = 0

    for i in range(m):

        f = sigmoid(w * x[i] + b)

        total_cost += (f - y[i])**2

    return total_cost / (2*m)

def compute_gradient(x, y, w, b):

    m = len(x)

    dj_dw = 0
    dj_db = 0

    for i in range(m):

        f = sigmoid(w * x[i] + b)

        error = (f - y[i])

        dj_dw += error * f * (1-f) * x[i]

        dj_db += error * f * (1-f)

    dj_dw /= m
    dj_db /= m

    return dj_dw, dj_db

def gradient_descent(x, y, w, b, alpha, iterations):

    J_history = []
    p_hist = []

    for i in range(iterations):

        dj_dw, dj_db = compute_gradient(x, y, w, b)

        w -= alpha * dj_dw
        b -= alpha * dj_db

        p_hist.append([w, b])

        J_history.append(compute_cost(x, y, w, b))

    return w, b, J_history, p_hist

w, b, J, p = gradient_descent(
    x_train,
    y_train,
    0,
    0,
    alpha=0.1,
    iterations=1000
)

print("Final w:", w)
print("Final b:", b)

for i in p:
    print(i)