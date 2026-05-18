# update g.d. for logistic regression
# explore g.d. on a familiar data set

import copy, math
import numpy as np
import matplotlib.pyplot as plt

# data
X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])

# plot data

fig, ax = plt.subplots(1, 1, figsize=(4,4))

ax.scatter(X_train[:,0], X_train[:,1])
ax.axis([0, 4, 0, 3.5])
ax.set_ylabel("x1", fontsize=12)
ax.set_xlabel("x0", fontsize=12)

plt.show()