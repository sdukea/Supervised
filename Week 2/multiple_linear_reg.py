# extending our linear model to support multiple features

import copy, math
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=2) # reduced display precision on numpy arrays
# what this does: when you print the arrays it rounds the numbers to 2 decimal places
#                 does not change its value; when printing/displaying alone, it shows those numbers
#                 rounded off to two decimal places

"""
a                 → scalar (non-bold)
a (bold)          → vector
A (bold, capital) → matrix

X        → training example matrix          (Python: X_train)
y        → training targets                 (Python: y_train)

x^(i)    → i-th training example            (Python: X[i])
y^(i)    → i-th target                      (Python: y[i])

m        → number of training examples
n        → number of features per example

w        → parameters (weights)             (Python: w)
b        → parameter (bias)                 (Python: b)

f_w,b(x^(i)) → model prediction for i-th example
              = w · x^(i) + b
                                            (Python: f_wb)

"""

# same motivating (running) example: housing price prediction

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

# features (in order): size, no: of bedrooms, no: of floors, age of home
# target: price of house

# X_train is now a matrix/2D array consisting of 1-D arrays as its values where each 1-D array
# is a training example/row with 4 values -> hence, we have 4 features
# for each training example, we have 1 target
# 3 training examples, each with 4 values/4 features and we have 3 targets for each tr.eg in y_train

# m = 3 (number of rows) - X_train.shape[0]
# n = 4 (number of columns) - X_train.shape[1]

print(f"X shape: {X_train.shape}")
print(f"y shape: {y_train.shape}")

# w is a vector (1D array) with n elements in it, for each feature

# x^(i) is a vector, representing one traininge example, unlike our previous univariate linear 
# reg. example where x^(i) was just a sinlge value as x_train (X_train here as it is a matrix now and
# 'x' is capitalized to 'X') had only one feature

# x^(i)j is a feature value from the x^(i) vector

b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])

# loading w and b that are already near optimal for demonstration 

# model prediction

# single prediction element by element

def predict_single_loop(x, w, b):
    
    # x - vector - its a training example; a 1-D array with all our n feature values (n = 4)
    # w - vector - for each feature value in the training example (X subscript 1, X subscript 2 ...)
    #              same size as 'x'
    # b - bias parameter

    n = x.shape[0]

    # as x is a training example 1D array, get the number of columns/features as the first parameter
    # for a 1-D array will be the number of values/features/columns (n,)

    pred = 0

    for i in range(n): # for each of the feature values in a training example
        pred_i = x[i] * w[i] # w1x1 = p1, w2x2 = p2, w3x3 = p3 and w[i]x[i] = p[i]
        pred += pred_i # add to pred
    
    pred = pred + b # add bias and return it below
    
    return pred # return a scalar prediction

# we are predicting element by element i.e. w1x1, w2x2, w3x3, w4x4 sepearately and then adding b and
# returning our prediction because

# f_wb_x_i is just the prediction for a single row vector as input x we give to predict is just a 
# house with 4 features/vector with 4 values in it and only for this vector, we have a predicition

# even our data X_train has, for each training example only, a specific target y

x_1 = X_train[0,:] # get a row/vector/1-D array/training example consisting of 4 feature values

f_wb_x_1 = predict_single_loop(x_1, w_init, b_init) # prediction for just row 1/x superscript 1
# consisting of x subscript 1 (feature value 1), x subscript 2 (feature value 2) and so on

print(f_wb_x_1)

# w_init -> exact same size as x_vector to match its feature values

# NOTE: we are only predicting for one training example with multiple features
# In SLR/ULR, we only had w1x1 + b type-shit and here we have w1x1, w2x2, w3x3 and so on + b only for
# one row/training example

# single prediction vector

# using vectorization now to predict for a single row/training example again

def predict(x, w, b):

    pred = np.dot(x, w) + b
    
    return pred

# Done! Using just vectorization, its faster as well

# NOTE: prediction/target is given for each training example only as said above

# so, the input x in real life of a house with 4 features is actually just a vector with 4 values
# and only for this vector, we predict y hat

# that's why we're training just a single training example to get our prediction

# its w (vector) * x_i (vector; x is a matrix like X_train) + b

# 'training' the model in MLR usually means finding the best 'w' and 'b'

# such that the model's input is just a vector like what predict_by_loop and predict inputs

# alongside the best choices for w and b

# so that the model inputs the house with 4 feature values, combines it with best 'w' and 'b'

# and outputs a prediction

# NOTE: 'training' should usually include the training data (X_train), right? Yes!
# Training the model (in MLR) - finding the best 'w' and 'b' via the cost function/G.D. and for these
# implemenetations, we do use X_train (the 2-D dataset we have) fully, not just a vector - see beloe

# NOTE: For predicting, we assume we already have w_init and b_init that we've we've got by training
# using the entire X_train and y_train so that we can predict by just getting in 1 input vector of a 
# house with 4 feature values and using those 'w' and 'b' to predict via the model in line 133.

x_1 = X_train[0,:]

f_wb_x_1 = predict(x_1, w_init, b_init)

print(f_wb_x_1)

# compute cost

def compute_cost(X, y, w, b):

    # returns cost: scalar

    # takes in:
    # X - entire data; the 2D array with (m,n) shape
    # y - 1-D array/vector with all our targets of shape (m,) (for each of our training examples)
    # w - initial w (vector) set of shape (n,)
    # b - model bias

    m = X.shape[0]

    # now, we get in the number of rows/tr. eg as we have a matrix X here

    cost = 0.0

    for i in range(m):
        f_wb_x_i = np.dot(X[i], w) + b

        cost = cost + (f_wb_x_i - y[i])**2

    cost = cost / (2 * m)

    return cost

