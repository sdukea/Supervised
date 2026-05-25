# finally, the open source implementation using sklearn

import numpy as np
import matplotlib.pylab as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
np.set_printoptions(precision=2)


# get data:

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

"""
X_train, y_train = load_house_data()
"""

X_features = ['size(sqft)','bedrooms','floors','age']

# scale data

scaler = StandardScaler()

X_norm = scaler.fit_transform(X_train)

# you have to 'fit' and 'transform' because you're transforming the data!

# doing just scaler.fit() won't change/transform data

# .fit() -> study the data and memorize the statistics needed for normalization
# specifically, it computes mean of each column and S.D. of each column
# so for each feature/colum, the mean and S.D. is just calculated

# and stores them internally under 'scaler'

# using .fit() says 'look at data and figure out how scaling should be done'

# if you did: scaler.transform(X_train)

# it applies the scaling i.e. 

# applies the formula x_feature_values - mean for that feature/column / S.D. for that feature/column
# where the mean and S.D. is what it had already calculated while we did fit()

# now, fit_transform() does this all under one single method!

print(f"Peak to peak range by column: {np.ptp(X_train, axis=0)}")
print(f"Peak to peak range by column after normalizing: {np.ptp(X_norm, axis=0)}")

# create and fit regression model

sgdr = SGDRegressor(max_iter=1000)
# Create a linear regression model trained via stochastic gradient descent.

sgdr.fit(X_norm, y_train)
# Repeatedly adjust weights to reduce prediction error.
# specifically: Learn the best weights (w) and bias (b) that map the 
# inputs X_norm to the outputs y_train.

# starts by randomly guessing (exactly like we did and in scratch implementation):
# w = 0, b = 0 (or)
# w = random, b = random
# predictions are terrible initially (we start somewhere in the plot of parameters and cost J)

# then it makes predictions y-hat = wx + b (varies if MLR is the case), measures error and readjusts
# parameters w (or w1 to wn) and b until they converge at minimum cost
# (until w and b is set such that error for all tr. eg. is the minimum; 
# the cost essentially (mean sq. error for all tr. eg.))

# SGD is good because normal G.D. updates w and b, predicts for all tr. eg. with this w and b, computes
# error for all tr. eg. with this w and b, finds cost (average sq. error/1 by 2m) for all tr. eg. with this \
# w and b and then updates w and b again until the cost converges
# so, it checks ALL tr. eg. to calculate cost for each update in w and b

# but S.G.D. starts/updates with w and b, predicts for ONE tr. eg. with this w and b, computes error for 
# that tr. eg. with this w and b, and adjusts weight to improve it to get new w and b right away

# so the next tr. eg. will use this new, improved w and b updated and got from the previous 
# tr. eg. error from prediction

# NOTE: cost in is the average sq. error for all tr. eg

# here, cost is = 1/2 (y-hat -y)^2; for one tr. eg.

# and with this, we get updated, new w and b to reduce cost of this tr. eg. so that the next

# tr. eg. will have a better w and b that is closer in trying to reduce the cost

# SGD: faster, noisier

# repeat this: thousands of times

print(sgdr)

print(f"Number of iterations completd: {sgdr.n_iter_}, number of weight updates: {sgdr.t_}")


# view parameters

b_norm = sgdr.intercept_
w_norm = sgdr.coef_
print(f"model parameters:                   w: {w_norm}, b:{b_norm}")

# make predictions

y_pred_sgd = sgdr.predict(X_norm) # predictions using sgdr.predict

# no test set; use tr. set again

y_pred = np.dot(X_norm, w_norm) + b_norm # predictions using np.dot()

# both should match

# plot predictions and targets vs original features    
fig,ax=plt.subplots(1,4,figsize=(12,3),sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:,i],y_train, label = 'target')
    ax[i].set_xlabel(X_features[i])
    ax[i].scatter(X_train[:,i],y_pred,color='orange', label = 'predict')
ax[0].set_ylabel("Price"); ax[0].legend();
fig.suptitle("target versus prediction using z-score normalized model")
plt.show()

