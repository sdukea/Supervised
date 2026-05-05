# exploring feature engineering and polynomial regression  which allows you to use
# machinery of linear regression to fit very complicated and even very non-linear functions.

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=2)

# generally, linear regression provides a means of building models of the form
# f(w(vector), b) = w0x0 + w1x1 + w2x2 + w3x3 + ... + wnxn + b (1)

# What if your features/data are non-linear or are combinations of features? 
# For example,  Housing prices do not tend to be linear with living area 
# but penalize very small or very large houses resulting in the curves shown in graph in JupyterNotebook.
# How can we use the machinery of linear regression to fit this curve? 
# Recall, the 'machinery' we have is the ability to modify the parameters w and b to 'fit' equation (1)
# to the training data.
# However, no amount of adjusting w and b in (1) will achieve a fit to a non-linear curve. 

# The only way to make our model fit these complex curves is to have polynomials

# What we will exactly need is a POLYNOMIAL FEATURE

