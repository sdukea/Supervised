# looking at NeuralNine's polynomial regression from scratch video

# first let's generate data that comes from a linear function
# i.e. the distribution or the randomness can be fitted with a linear
# function/linear regression model

# then we'll see different, more complex distributions and we'll
# try to fit them with a linear function to see that linear regression
# is not enough and so we have to polynomial-ize it

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X = 4 * np.random.rand(100, 1) - 2
# range is now made ro -2 to 2

# generate data; 2D array
# np.random.rand/np.random.random_sample all take the 
# THE 'SHAPE' AS THE ARGUMENT; ALWAYS UNDERSTAND
# but you don't have to specify them inside a tuple; just arguments

y = 4 + 2*X + 5 * X**2 + np.random.randn(100, 1)
# converted to a quadratic relationship (our target)

# the last part of randn adds some random normal 
# noise
# just 4 + 5*X will make the X, y relationship a 
# straight line when you plot it -- just a straight line
# and it won't even be a linear regression problem

# np.random.randn(100, 1) adds random values to each data
# point

# np.random.randn(100, 1) generates 100x1 array (2D) of
# random numbers drawn from a normal distributiion i.e.
# numbers that have mean = 0 and S.D = 1

# it is added so that it simulates real world noise

# y = 4 + 5*X + noise is much more realistic

# doing y = 4 + 5*X + 2 * noise (that randn term) will
# make you have more noise


# NOTE: It's like doing x + y where x and y are 2D arrays
# 4 + 5*X is a 2D array by itself
# and we're adding a 2D array of noise values to this
# so each 4 + 5*X 2D array (modifying X 2D array) 
# value will be associated with a noise to it as
# the noise 2D array is of the same size

# plt.scatter(X, y)
# plt.show()

# so, this is a more like a linear function; the 
# trend of this relationship is linear and now we can
# easily fit a linear function to it; a line to it

# let's use L.R. to fit our linear trend showing data

reg = LinearRegression()

reg.fit(X, y)

# fit a linear model/line to our data showing a linear
# trend

X_vals = np.linspace(-2, 2, 100).reshape(-1, 1) # converted
# range here as well - from -2 to 2
# linspace; generate linearly spaced values over an
# interval
# |
# starts at 0, ends at 1, creates 100 equaly spaced
# points
# |
# so 100 values from 0 to 1
# and the shape will be (100,) - a 1-D array
# |
# so reshape it into 2D so that we can use it to predict
# (a test-set)

y_vals = reg.predict(X_vals)

# why X_vals and y_vals?

# model is already fit/trained on X, so why can't I just
# use the same X to predict

# you can, but note this:

# only for the X values you have generated -- you've 
# plotted as well -- you get a prediction

# so, only for all the blue points that we've plotted
# in the scatter plot, we get a prediction

# but in real life predictions, this isn't always the case

# you don't have data to predict that will exactly
# mirror what you have learned

# some of it may be the exact same as our training data

# but it doesn't mean that every data we predict

# be present in the training data or else we haven't

# learned anything to apply it for unseen cases

# the data we get can/should be in the same range as 

# our training data but it shouldn't be all of the 

# exact values in training data that we get as unseen data

# to give a prediction for -- even data that is not there

# in training data (but still in the same range as our

# training data) should be given a prediction for

# that's why we use X_vals -- it is a set of X values

# that includes some exact values in X -- our training data 

# -- and mirrors them exactly and also has values that X

# doesn't have in that 0 to 1 range

# so X_vals covers the 0 to 1 range better and includes

# a real-world-like data matching our training data values

# (X) exactly as well as new, in-range values that are also

# acceptable;

# so that our prediction can be fully truthful to real,

# unseen data (still in range) and gives a better view on

# the model's performance.

# NOTE: The primary reason for X_vals is mainly for
# visualization and partially for simulating 'real-world
# prediction'

# we just want to see what the model will predict across
# the entire input range of 0 to 1 that the training data
# has failed to cover

# NOTE: modle can already predict unseen data without
# X_vals -- we use X_vals just to see how model performs
# across the entire range of your training data because
# training data will only cover not most of the range
# it is in (0 to 1) so we use X_vals to cover the entire
# range and see how the model is doing across the entire
# range

"""
We use X_vals not because the model needs new data to work, 
but because we want to systematically evaluate the model 
across the full input range.

Unlike X, which is just a sample of points, X_vals is 
evenly spaced and covers the range smoothly, allowing us 
to clearly visualize the function the model has learned.
"""

# NOTE: and of course, X_vals is in the same range as 0 to
# 1 like X -- just that X_vals mirrors the range much more
# distributedly and fully which X might fail to have done
# but is still in the same range so you can plot a scatter
# of X in the same range and plot a line of X_vals
# in the same range (both are in the same range)

plt.scatter(X,y)    
# our training data as a scatter plot

plt.plot(X_vals, y_vals, color='r')
# plots a line with X_va

plt.show()

# this is how linear regression actually works

# the problem: the function is simple
# it is after all linear and we're using linear regression
# to fit it

# what happens if we have a polynomial function

# say (we're changing the model to be a 
# polynomial (quadratic) function)
# changed 'X' as well to have a range of -2 to 2
# and similarly 'X_vals' to also have a range of -2 to 2
# (covers much more fully and mirrors X as though it has
# a full range from -2 to 2 now)to see the quadratic better

# now, you can see we're not able to fit a linear model
# to a quadratically related data (the target is quadratic)

# a linear model isn't enough to fit data that shows
# a quadratic behaviour -- between input features and 
# output target

# So we do need poly. reg.

# –––––––––––––

# there isn't anything explicit like 
# 'PolynomialRegression'; we use polynomial features
# instead -- we just engineer the features that is we
# square them or polynomial-ize them
# we have a new feature now -- not a new information
# per se because we already have X and X^2 is an extension
# only.

# we can combine X and X^2 to train a complex model

from sklearn.preprocessing import PolynomialFeatures

poly_features  = PolynomialFeatures(degree=2)
# choose degree -- don't over do it

# degree 2 -- X and squaring/X^2
# degreee 3 -- X, X^2 and X^3 will be added to features

# and you can go to even 10, 20 and 100

# to create a lot of poly features

# but too much is a problem -- overfitting

# doesn't generalize to new data points that are in the
# same range because overfitted model does not generalize
# to those new data points -- it becomes very strict and
# inflexible

# 2 is fine

