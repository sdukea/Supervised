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
# - dataset is just two data points/training examples - will constitute our tr. set fully
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

