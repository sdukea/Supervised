# training a logistic reg. model using SKLearn
# (the boring part)

import numpy as np
import matplotlib.pyplot as plt

# load data

X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1])

# fit the model

from sklearn.linear_model import LogisticRegression

