# let's contrast classification and regression

import numpy as np
import matplotlib.pyplot as plt

# examples of classification include:
# 1. identifying email – spam or not spam
# 2. determining tumor – present or not present
# these are all examples of binary classifications where there are two possible outcomes/categories/
# classes
# and these binary outcomes can be defined as: 0/1, yes/no, true/false, positive/negative

x_train = np.array([0., 1, 2, 3, 4, 5])
y_train = np.array([0,  0, 0, 1, 1, 1])
X_train2 = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train2 = np.array([0, 0, 0, 1, 1, 1])

# some data; for each training example, we have 1 associated binary outcome/class – in our case, its 0
# or 1

