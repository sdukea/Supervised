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

pos = y_train == 1
neg = y_train == 0

print(pos)
# this is an array like: [False, False, False, True, True, True]
# all positive examples i.e. where ever there is 1s are given as 'True'

print(neg)
# [True, True, True, False, False, False]

# get all positive and negative outcomes

fig,ax = plt.subplots(1,2,figsize=(8,3))

#plot 1, single variable
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

plt.show()
