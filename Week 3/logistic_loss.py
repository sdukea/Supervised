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

w, b, J_model, p = gradient_descent(
    x_train,
    y_train,
    0,
    0,
    alpha=0.1,
    iterations=1000
)

print("Final w:", w)
print("Final b:", b)

w_model = []
b_model = []

for w, b, in p:
    w_model.append(w)
    b_model.append(b)

# NOTE: values in J_model, w_model and b_model are only for the RIGHT BABY STEPS IN THE G.D. plot
# it represents the coordinates for the optimized path G.D. takes to reach a valley fast such that
# we have the lowest cost J for that w and b (valley)
# but in order to see G.D. plot, we shouldn't just use this BABY STEPS PATH ALONE - this optimized path
# will just be a line (shaky of course) that goes to the valley
# For G.D. plot, we need ALL possible combinations of w, b and inherent J so that we get the entire area 
# plotted AND then draw the optimized path with the subset w and b (w_model and b_model) and J_model that
# shows us the optimized path towards a valley from the whole area plot of G.D. 

# so to generate the entire G.D. plot with all possible combinations of w and b and all of the cost J
# for them, we use meshgrid


w_vals = np.linspace(-10, 10, 100)

b_vals = np.linspace(-10, 10, 100)

# meshgrid
W, B = np.meshgrid(w_vals, b_vals)

# emppty array with same size as W (ofc)
J_vals = np.zeros_like(W)

# ----------------------------------------

for i in range(W.shape[0]):

    for j in range(W.shape[1]):

        w = W[i, j]

        b = B[i, j]

        J_vals[i, j] = compute_cost(
            x_train,
            y_train,
            w,
            b
        )

# plot 3D surface plot

fig = plt.figure(figsize=(10,7))

ax = fig.add_subplot(projection='3d')

# ChatGPT stopped answering my questions from here