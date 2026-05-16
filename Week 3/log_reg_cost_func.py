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

ax.scatter(X_train[pos], y_train[pos], marker='x', s=80, c='red', label='y=1')

ax.scatter(X_train[neg], y_train[neg], marker='o', s=100, label='y=0')

ax[0].set_ylabel('y', fontsize=12)
ax[0].set_xlabel('x', fontsize=12)
ax.legend()

plt.show()