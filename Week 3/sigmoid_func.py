# exploring the sigmoid function
# exploring logistic regression; that uses the sigmoid function

import numpy as np
import matplotlib.pyplot as plt

# why a l.r. model won't work
# for classification tasks, we predict one outcome out of only a handful of categories

# and often, these handful of categories are just two of them: 0 (no), 1 (yes)
# NOTE: there can be many categories as well, but these are the most frequently used for classification
# task – the most obvious of course
# they are not infinitely many – only a handful that we can count – and this is what makes it
# a classfication task

# and so our prediction lies only between these two categorical outcomes we have; 0 or 1

# so a number between 0 or 1 is the outcome we will need for classification tasks;
# i.e. the probability

# a l.r. model predicts a random number based on parameters and input values via the model
# f_wb_x_i = w * x_i + b

# but we only need a number – that is probabilistic of the outcome – between 0 or 1

# and probabilities cannot be a number below 0 or above 1

# so a sigmoid function does this:
# 1. squishes every number into 0 to 1

# specifically, it takes in any number
# -1000, -500, 0, 8, 999... anything

# and converts it into:

# a number between 0 to 1 that is probabilistic of the outcome present in virtue of the two categories
# we have (2 in our case; that is the most general case)

# in classifc, the outcome/class/probabilistic value is from a finite set of categories (2 here - 0/
# no/absent or 1/yes/present)

# and in reg, we predict a value that is from a spectrum of values; infinite
# after plotting the regression model, this plot, as you know before, can be zoomed in and out and
# and go on forever infinitely
# and also the model also extends forever in both directions so there exists an unbounded, continous
# range FOREVER

# and our prediction is one of that – not from a finite set of categories but from an infinite spectrum

# that's the difference

# formula for s. func.

input_arr = np.array([1, 2, 3])

exp_arr = np.exp(input_arr)

# does e raised to all the elements in the given array 'input_arr'

print(input_arr)
print(exp_arr)

# input is just a single number
# because sigmoid functions operates on one value at a time

# so, the func. takes one number in (any) and one number out (0 to 1)

input_val = 1
exp_val = np.exp(input_val)

# implement.

def sigmoid(z):
    
    """
    Compute the sigmoid of z

    Args:
        z (ndarray): A scalar, numpy array of any size.

    Returns:
        g (ndarray): sigmoid(z), with the same shape as z
         
    """

    g = 1 / (1+np.exp(-z))

    return g

# let's see how the output of this function is for various values of z

# generate evenly spaced values between -10 and 10
z_tmp = np.arange(-10, 11)
