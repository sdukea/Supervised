# let's contrast classification and regression

import numpy as np
import matplotlib.pyplot as plt
import math

# examples of classification include:
# 1. identifying email – spam or not spam
# 2. determining tumor – present or not present
# these are all examples of binary classifications where there are two possible outcomes/categories/
# classes
# and these binary outcomes can be defined as: 0/1, yes/no, true/false, positive/negative

x_train = np.array([0., 1, 2, 3, 4, 5])
y_train = np.array([0,  0, 0, 1, 1, 1])

X_train2 = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
# now has two features – like x_train but x_train is a vector (1D) and X_train2 is a matrix of features
# (2D)
y_train2 = np.array([0, 0, 0, 1, 1, 1])

# some data; for each training example, we have 1 associated binary outcome/class – in our case, its 0
# or 1

pos = y_train == 1
neg = y_train == 0

print(pos)
# this is an array like: [False, False, False, True, True, True]
# all positive examples i.e. where ever there is 1s are given as 'True'

print(neg)
# [True, True, True, False, False, False]

# get all positive and negative outcomes

fig,ax = plt.subplots(1,2,figsize=(8,3))

# NOTE: In plot, the convention is:
# x -> y= 1
# o -> y = 0 


#plot 1, single variable/feature
ax[0].scatter(x_train[pos], y_train[pos], marker='x', s=80, c = 'red', label="y=1")

# x_train[pos] demonstrates boolean masking
# of course, pos should be the same size as x_train
# and where ever x_train is indexed with True from pos i.e all the values
# of x_train where it is True (from pos), will be = x_train[pos]

# so, x_train has True for the last three values (as in pos, the last three index values are True)
# so it contains only these values now, as we've indexed with boolean and these values have True in them

# so x_train[pos] = [3., 4., 5.]

# same for x_train[neg]

ax[0].scatter(x_train[neg], y_train[neg], marker='o', s=100, label="y=0", facecolors='none', 
              edgecolors='blue',lw=3)

ax[0].set_ylim(-0.08,1.1)
ax[0].set_ylabel('y', fontsize=12)
ax[0].set_xlabel('x', fontsize=12)
ax[0].set_title('one variable plot')
ax[0].legend()

#plot 2, two variables
# plot_data(X_train2, y_train2, ax[1]) # type: ignore
# a helper function plots data;
# a mechanism inside it that lets you plot data with 2 features and binary outcome labels
ax[1].axis([0, 4, 0, 4])
ax[1].set_ylabel('$x_1$', fontsize=12)
ax[1].set_xlabel('$x_0$', fontsize=12)
ax[1].set_title('two variable plot')
ax[1].legend()
plt.tight_layout()

# a two-variable/feautre plot
# - plots the two features
# - each markers x and o represent binary label 1 and 0
# - x = 1, o = 0 
# - so one plot for x0 (one feature) and another for x1 (another feature)
# - and each marker i.e. for each feature value x0 and x1 (point on graph) the associated label is 
# marked with a marker x if associated label y = 1 or o if associated label y = 0

# linear regression approach

# using l.r. to predict if a tumor is benign or malignant based on tumor size

# it won't be sufficient to model categorical data; let's prove it

# let's do it from scratch – of course

def cost_func(x, y, w, b):
    
    m = x.shape[0]

    total_cost = 0

    for i in range(m):

        # compute prediction for each tr. eg.
        # NOTE: nv -> not a vector (s.l.r. with just one feature)

        f_wb_x_i_nv = w * x[i] + b

        cost = (f_wb_x_i_nv - y[i])**2

        total_cost += cost

    final_cost = total_cost / 2*m

    return final_cost

def compute_grad_terms(x, y, w, b):

    m = x.shape[0]

    dj_dw = 0
    dj_db = 0

    for i in range(m):

        f_wb_x_i_nv = w * x[i] + b

        dj_dw_i = (f_wb_x_i_nv - y[i]) * x[i]

        dj_db_i = (f_wb_x_i_nv - y[i])

        dj_dw += dj_dw_i

        dj_db += dj_db_i
    
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db

def gradient_descent(x, y, w_in, b_in, cost_func, gradient_terms, alpha, iterations):

    J_history = []
    p_history = []

    w = w_in
    b = b_in

    for i in range(iterations):

        dj_dw, dj_db = gradient_terms(x, y, w, b)

        # update

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        J_history.append(cost_func(x, y, w, b))
        p_history.append([w, b])

        if i % math.ceil(iterations / 10) == 0:
        
            print(f"Iteration {i}: Cost {J_history[-1]} ",
                  f"dj_dw: {dj_dw}, dj_db: {dj_db}  ",
                  f"w: {w}, b:{b}") 
    
    return w, b, J_history, p_history

# now, we have the best parameters possibly to implement l.r. on categorical data and see why
# the linear model won't do good

w_init = 0
b_init = 0

iterations = 10000

alpha = 1.0e-2

w_final, b_final, J_hist, p_hist = gradient_descent(x_train, y_train, w_init,
                                                    b_init, cost_func=cost_func, 
                                                    gradient_terms=compute_grad_terms, alpha=alpha,
                                                    iterations=iterations)

ax[1].scatter(x_train[neg], y_train[neg], marker='o', s=100, label="y=0", facecolors='none', 
              edgecolors='blue',lw=3)

ax[1].scatter(x_train[pos], y_train[pos], marker='x', s=80, label='y=1')

ax[1].legend()

# s -> size of marker
# label -> for when legend is used; sets up legend

# now to plot a linear regression model line in our categorical plot here:

# we need points!
# i.e. feature value x (x-coordinate) and prediction by our model (y-coordinate)

# right now, we have learned parameters w and b

# and our scatter plot of categorical data we want to plot on has x range from 0 to 5

# so our data that goes in model (till now, we've seen adding training data itself for prediction done
# by model with w and b we got) should be in the same range for it to exist clearly in the plot as this
# x_train is what we used to plot and imposes the 0 to 5 range in x-axis (where feature x is always 
# plotted)

# so we'll use the training data x_train itself for prediction

# we could also do:

x_for_pred = np.linspace(0, 5, 100)

# creates 100 values between 0 to 5 i.e. 100 feature values evenly spaced between 0 to 5
# where 0 to 5 is the range our x_train was also in

# you can use this for predicting but again, won't mirror your training examples everywhere – will do
# by some values as it is in the same range but won't have the EXACT feature values x_train that we used
# for training and plotting scatter plot

# we do linspace to, as said/understood earlier, evenly grab hold of the entire range x lies on as x 
# might not always not capture the range fully
# we want to see the prediction y_pred/y-hat on feature values x_train itself (exactly mirroring) or 
# feature_values that are in the exact same range as x_train and capture the range to the full length
# (100 fine divisions evenly up until 5 from 0) because we want to see how the model does in x_train-kind
# -of-data but with better range so that we can convince ourselves of performance on data that is 
# well obidient to its scale (x_for_pred is like a better upgrade of x where x is still the GOAT)

y_pred = w_final * x_train + b_final

# got predictions

# plot line

ax[1].plot(x_train, y_pred)

plt.show()

# we can see why a linear model won't be able to generalize categorical input data