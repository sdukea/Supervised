# plotting d.b. for a log. reg. model

import numpy as np
import matplotlib.pyplot as plt

# the following tr. set.
# X -> matrix of features here
X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])

# associated binary outcome y for each tr. eg (each tr. eg. here is a vector)
y = np.array([0, 0, 0, 1, 1, 1]).reshape(-1,1) 

