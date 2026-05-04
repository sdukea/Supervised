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

    for i in range(m): # for each training example
        f_wb_x_i = np.dot(X[i], w) + b # compute prediction for each training example
        
        # exactly what the predict_single_loop and predict (vectorized version) does

        # takes in a vector training example - X[i] here - and compute prediction

        cost = cost + (f_wb_x_i - y[i])**2

        # cost is actually for all training examples m

        # so, for each training example, compute prediction as we did above

        # subtract from actual target value to get error

        # square it

        # and add it to 'cost' variable.

        # and do it for all training examples and add its cost to the 'cost' to get total

        # un-averaged cost.

    cost = cost / (2 * m)

    # divide total cost by 1 2m to get average cost

    return cost

cost = compute_cost(X_train, y_train, w_init, b_init)

print(f"Cost: {cost}")

# gradient descent with multiple variables

def compute_gradient_terms(X, y, w, b):

    m, n = X.shape
    # get both

    dj_dw = np.zeros((n,))
    # why?
    # gradient term - measures the direction of the descent/slope and partially the size as well
    # in ULR, we only had one w in our G.D. algorithm and if we could plot J(w,b), w and b

    # but in MLR, we have n features and for each feature we have weights - so n weights as well
    # and for G.D., we will have to descent (in the direction that leads to a valley) for all
    # weights we have because the plot will be all w (w1 to wn), b and J(w(vector),b)

    # we need updation for EACH w because G.D. needs to happen for all w1 to w2 and b simultaneously
    # for it to reach a valley and hence, for each w, we need to compute gradient term specifying the
    # direction it will need to take to reach valley as the plot is now made up of w1 to wn and b to
    # illustrate J(w(vector), b)

    dj_db = 0
    
    # set for accumulation

    for i in range(m): # for each training example

        # error = prediction - actual
        error = (np.dot(X[i],w) + b) - y[i]
        

        # for dj_dw term

        for j in range(n):

            dj_dw[j] += error * X[i, j]
        
        # what's going on: 
        # we're dealing with the gradient descent term + the multiplied x subscript n superscript (i)
        # term as well

        # this is how that gradient term should be read:
        # from i = 1 to m i.e. for each training example -
        # find error w.r.t. to actual target
        # and for this training example (after finding the error),
        # for each feature j = 1 to n this training example has,
        # multiply error with each n feature value 
        
        # error for eg. 1 * x_1_(1) -- gives dj_dw_1 (dj_dw[0]) for w1 (w1_(1))
        # error for eg. 1 * x_2_(1), -- gives dj_dw_2 (dj_dw[1]) for w2 (w2_(1))
        # error for eg. 1 * x_3_(1), -- gives dj_dw_3 (dj_dw[2]) for w3 (w3_(1))
        # error for eg. 1 * x_4_(1) -- gives dj_dw_4 (dj_dw[3]) for w4 (w4_(1))

        # and so, for this training example (number (1)), we have 4 gradient terms
        
        # and each training example gives j = 1 to 4; 4 gradient terms and hence 4 weights 
        # that we can compute with this need to account for the input vector that will have 4 features

        # and now for the next training example,

        # do all of the same and

        # reupdate the gradient terms into dj_dw array i.e. 

        # for the next training example (eg. 2):
        
        # error for eg. 2 * x_1_(2) -- add to dj_dw_1 (dj_dw[0]) for w1 (w1_(2))
        # error for eg. 2 * x_2_(2), -- add to dj_dw_2 (dj_dw[1]) for w2 (w2_(2))
        # error for eg. 2 * x_3_(2), -- add to dj_dw_3 (dj_dw[2]) for w3 (w3_(2))
        # error for eg. 2 * x_4_(2) -- add to dj_dw_4 (dj_dw[3]) for w4 (w4_(2))

        # and so on for each training example

        # so, each training example partially contributes to the 4 gradient terms in total we've
        # got -- initially set by the first gradient term

        # for dj_db term

        dj_db += error
    
    dj_dw = dj_dw / m                                
    dj_db = dj_db / m 

    return dj_dw, dj_db

