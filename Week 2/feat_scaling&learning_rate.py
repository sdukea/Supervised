"""
In this lab you will:
- Utilize  the multiple variables routines developed in the previous lab
- run Gradient Descent on a data set with multiple features
- explore the impact of the *learning rate alpha* on gradient descent
- improve performance of gradient descent by *feature scaling* using z-score normalization
"""

import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=2)

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

X_features = ['size(sqft)','bedrooms','floors','age']

fig,ax=plt.subplots(1, 4, figsize=(12, 3), sharey=True)

# subplots - multiple plots in a single run
# 1 - 1 row
# 4 - 4 different plots on that 1 row
# so creates 4 empty graphs sitting next to each other
# fig - the outer box/container holding inner graphs
# ax - [plot1, plot2, plot3, plot4]

# figsize - wide (12) and short (3)
# sharey - all plots use the same y axis -> price



for i in range(len(ax)): # for each plot
    ax[i].scatter(X_train[:,i],y_train) # plot the feature - the index [:, i] and y_train
    ax[i].set_xlabel(X_features[i]) # set xlabel
ax[0].set_ylabel("Price (1000's)") # you just have to set the label for one plot - all plots share it
plt.show()

# you can now see each feature being plotted with the target

# learning rate

# a lot of proprietary modules used to demonstrate choosing the best learning rate
# so, needed those but couldn't get them here so skip it - understand the intuition though

# feature scaling

