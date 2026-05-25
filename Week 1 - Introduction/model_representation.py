# learning to implement the model f w,b for linear regression with one variable

# notations:

# a, unbold - scalar
# a, bold - vector (visible in PDF)
# x - dataset of features
# y - labels for each training example in x
# x_train - part of x taken for training l.r. model
# y_train - labels for the x_train taken part of x
# x_i, y_i - ith training example
# m - number of training examples
# w, b - parameters
# f_wb - result of the model evaluation at x_i parameterized by w, b

import numpy as np
import matplotlib.pyplot as plt

# set plot style

# plt.style.use('./deeplearning.mplstyle') 

# problem: housing price prediction
# - dataset is just two data points/training examples/values here - will constitute our tr. set fully
# - unit of size: 1000 sq ft | unit of price: 1000s of $

# 1 feature (size of house), 1 set of labels for that feature (price of house)

x_train = np.array([1.0, 2.0]) # input features for training - just 1
y_train = np.array([300.0, 500.0]) # corresponding labels for given feature

print(f"x_train: {x_train}")
print(f"y_train: {y_train}")

# tr. eg

m = x_train.shape[0]

print(f"Tr. eg.: {m}")

# also: len(x_train)
# gives you number of rows - a 1-D array at the end of the day so only has values as rows

# .shape gives you entry for each dimension (for row and column) in a tuple
# x_train.shape = (no: of rows/examples, no: of columns/features)

# so, examples = m = x_train.shape[0] = first value

# set x_i and y_i
# has to be iterable to set i = 0 ready for iteration and then
# index x_i indexed to i and y_i indexed to i, making both ready for iter.

i = 0

x_i = x_train[i]

y_i = y_train[i]

# plot

plt.scatter(x_train, y_train, marker='x', c='r')

plt.title('Housing prices')

plt.ylabel('Price (1000s)')

plt.xlabel('Size (1000 sq ft)')

plt.show()

# parameters

# let's set, for now,

w = 100 # w = 200; perfect

b = 100

print(f"w: {w}\nb: {b}")

# compute f_wb

# NOTE: x - the data set of input features - is just 1 feature and has two values/tr. eg. is just values
# /data points.
# so, x^(i)/x_i - a training example consisting of one value only - not a vector because
# only one value - would be x-arrow-on-top-vector^(i)/x_i_vector instead

# For one feature, you have one weight w - and for each feature value/tr. eg./data point, you use
# the same w because it is only of one feature

# for x_0, f_wb = w * x_0 + b (for first data point/tr. eg/feature value)
# for x_1, f_wb = w * x_1 + b (for the second)

# can't do it for all if there are more tr.eg/feature values

def compute_model_output(x, w, b):
    """
    Computes prediciton of a linear model

    returns f_wb: model prediction
    """

    m = x.shape[0]

    f_wb = np.zeros(m)

    # Create an array of m zeros i.e. for each tr. eg., you need a prediction
    # m tr. eg = m predictions (initialize predictions to zero now)

    for i in range(m):
        f_wb[i] = w * x[i] + b
    
    return f_wb

# plot output

f_wb = compute_model_output(x_train, w, b)

print(f_wb)

# an array of 2 predictions done for each tr. eg (just values as we only have 1 feature -  multiple features
# would make one tr. eg in MLR) in x_train (2 of them (values) and only one feature) 
# and corresp. y_train


# values/data points/x_train and model prediction
plt.plot(x_train, f_wb, c='b', label='Our prediction')

# values/data points and actual labels

plt.scatter(x_train, y_train, c='r', marker='x', label='Actual values')

# can now trace errors here.

plt.title('Housing prices')

plt.xlabel('Size (1000 sq ft)')

plt.ylabel('Price (1000s)')

plt.legend()

plt.show()