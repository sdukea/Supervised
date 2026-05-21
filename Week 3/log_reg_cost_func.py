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

ax.legend()


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

# check c.func. implementation

w_tmp = np.array([1, 1])
b_tmp = -3
# recall, we used w1 = 1, w2 = 1 and b = -3 prev. for d. boundary
# driving them here as well 
# these are pre-given/for demonstration

def sigmoid(z):

    g_z = 1 / (1 + np.exp(-z))

    return g_z

print(compute_cost_logistic(X_train, y_train, w_tmp, b_tmp, sig_func=sigmoid))

# let's set diff. value for w1, w2 and b now and see what the cost outputs

# say: w1 = 1, w2 = 1, b = -4

# so you have two diff. d. boundaries now:
# 1. x1 + x2 -3 = 0
#       x1 + x2 = 3

# 2. x1 + x2 - 4 = 0
#        x1 + x2 = 4

x0 = np.arange(0, 6)

x1 = 3 - x0
x1_other = 4 - x0

# plot them

fig, ax = plt.subplots(1, 1, figsize=(4, 4))

ax.plot(x0, x1, c='blue', label='b=-3')
ax.plot(x0, x1_other, c='magenta', label='b=-4')
ax.axis([0, 4, 0, 4])

# these are only d. boundary plots - only lines
# you have to now plot the binary outcomes to see how well the d. boundary seperates them - proposes a 
# boundary

ax.scatter(X_train[:,0], X_train[:,1])

ax.set_ylabel('y', fontsize=12)
ax.set_xlabel('x', fontsize=12)

# yeah, the same plot as above

# now you see that the magenta line – the new d. boundary line with b = -4 – is not a good boundary of
# the outcome
# - it misclassifies/misinterprets the outcomes
# - the points < boundary = negative class
#   but positive class points fall below the boundary and are being misclassified as 'negative'/False
#   and there is no meaning to this seperation given by the boundary

# so, b = -4 and the 2. d.boundary is not good
# so it should have higher cost

# let's see that



plt.show()

# NOTE: above, we're actually adding the loss as 'cost' and readding it to the variable
# its actually loss for a single training example that we acumulate as cost finally (average of all
# the losses for every tr. eg.)

# but why: -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)
# when true output label is 1, this ^^^^^^^^^^^^^^^^^^^^^^^ term will be made 0 as
# 1 - 1 (i.e. y[i]) = 0 and (0)*np.log(1-f_wb_i) = 0 and finally, its just the first term we
# will be looking at to see how well the prediction is when the true output label is actually 1
# (we are TRAINING here, and we TRAIN THE MODEL WITH KNOWN OUTPUT LABELS, don't forget)

