# looking at NeuralNine's polynomial regression from scratch video

# first let's generate data that comes from a linear function
# i.e. the distribution or the randomness can be fitted with a linear
# function/linear regression model

# then we'll see different, more complex distributions and we'll
# try to fit them with a linear function to see that linear regression
# is not enough and so we have to polynomial-ize it

import numpy as np
import matplotlib.pyplot as plt

X = np.random.rand(100, 1)
# generate data; 2D array
# np.random.rand/np.random.random_sample all take the 
# THE 'SHAPE' AS THE ARGUMENT; ALWAYS UNDERSTAND
# but you don't have to specify them inside a tuple; just arguments

y = 4 + 5*X

plt.scatter(X, y)
plt.show()