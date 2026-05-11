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

# NOTE:
# 1. a l.r. z = model wx + b (as log. reg. is still regression) learns parameters w and b from 
# categorical data i.e. feature values and binary labels
# 2. so that class = 1 examples push z positive
# i.e. when z is a positive number, it usually means that the predicted class = 1 but again, this isn't
# exactly shown as '1' – it won't be the exact number '1' in most cases and will be higher than 1
# 3. and class = 0 push z negative
# i.e. when z is a negative number, it usually means that the predicted class = 0 but won't be exactly = 0
# and it isn't insightful and won't be exactly shown as '1' – it will be lower than 0 in most cases

# so, regardless, z will be a positive or a negative number and this polarity attributes to the two classes
# or binary labels
# not neccassily 0 or 1; mostly will be greater than 1 and less than 0

# but we need between 0 and 1 only – that's why we have the sigmoid function

# and as you know it, a sigmoid function, when given any number, it compresses into 0 to 1
# because of its mathetmatical properties that make it special and extremely useful for this case

# so, when a high positive number prediction from z = wx + b is given, the function's output will be 
# close to 1, exactly attributing to class = 1, True, Yes, present

# and when a high negative number prediction goes into the sig. func., it turns out to be 0 from the
# function's output

# high positive number = z -> sig. func. output close to 1 – exactly attributing to class = 1 as
# sig. func. outputs are the probabilistic values/probabilities that outcome = class = 1

# high negative number = z -> sig. func. output clsoe to 0 - exactly attributing to class = 0

# that's why a sig. func. helps

# formula for s. func.
# demonstrating exponentiation

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

# NOTE: z_tmp is exactly how the output of our z = wx + b predictions (for each tr. eg and associated
# label = 1 or 0) will look like

# contains positive numbers and high positive numbers that inherently attribute to class = 1 (for each
# tr. eg. given as input for prediction or new test example)
# (here its just generated manually but mirrors in a good way as to what exactly z will look like)

# and contians negative numbers and high negative numbers that inherently attribute to class = 0 (for each
# tr. eg. given as input for prediction or new test example)

# does not exactly have 0 or 1 in its values – its a distributed range of values but they do infer class
# predictions of 0 or 1 – we just need to encode them using the sigmoid funcion

# use sigmoid

y = sigmoid(z_tmp)

# Code for pretty printing the two arrays next to each other
np.set_printoptions(precision=3) 
print("Input (z), Output (sigmoid(z))")
print(np.c_[z_tmp, y])

# so y = array returned from sigmoid() function
# which returns g = 1 / (1+np.exp(-z))

# by taking in input as 'z' which is z_tmp

# so g is the exact size as 'z' = 'z_tmp' with formula applied to those elements in 'z' = 'z_tmp'

# and y = g = same size as 'z_tmp' = 'z'

# plot z vs sigmoid(z)

fig, ax = plt.subplots(1, 1, figsize=(5,3))

ax.plot(z_tmp, y, c="b")

# z_tmp

ax.set_title("Sigmoid function")
ax.set_ylabel('sigmoid(z)')
ax.set_xlabel('z')

plt.show()

# x-axis contains all these z_tmp predictions that are not exactly 0 or 1 but infer these class 
# predictions by being greater than 1 (until 11 in our case) or less than 0 (until -10 in our case)

# log. reg.

# get some data in; as you know, its categorical – for each feature value/vector, we have an associated
# binary outcome only

x_train = np.array([0., 1, 2, 3, 4, 5])
y_train = np.array([0,  0, 0, 1, 1, 1])

# initialize w and b = 0 for now
w_in = np.zeros((1))
b_in = 0

