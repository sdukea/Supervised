# plotting d.b. for a log. reg. model

import numpy as np
import matplotlib.pyplot as plt

# load data

# the following tr. set.
# X -> matrix of features here
X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
# shape: (6, 2) - 2D vector

# associated binary outcome y for each tr. eg (each tr. eg. here is a vector)
y = np.array([0, 0, 0, 1, 1, 1]).reshape(-1,1) 

# shape: (6, 1) - 2D vector as reshape has two arguments (reshape is a method where it takes
# shape as plain arguments)

# plot data

fig, ax = plt.subplots(1, 1, figsize=(4,4))

ax.scatter(X[:,0], X[:,1],c=y)
# without 'c', the argument for color, all points are the same color

# NOTE: You are plotting the d.b.
# it is plotted over the graph of feature – one feature in one axis and another in another axis
# this is for when you have 2 features – for more, you see a high dimensional graph

# so we plot one featute values and other feature values in scatter plot

# with 'c=y', it says if class = 0, use one color – if class = 1, use another color
# so matplotlib sees that c = y = [0,0,0,1,1,1] and sees the boolean, binary difference (0 for some
# values and 1 for some values in y) and it understands this difference and sets differentiated
# color – one for 0/one class in y and another for 1/another class


ax.axis([0, 4, 0, 3.5])
ax.set_ylabel('$x_1$')
ax.set_xlabel('$x_0$')


# Plot sigmoid(z) over a range of values from -10 to 10
z = np.arange(-10,11)

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

fig,ax = plt.subplots(1,1,figsize=(5,3))
# Plot z vs sigmoid(z)
ax.plot(z, sigmoid(z), c="b")

ax.set_title("Sigmoid function")
ax.set_ylabel('sigmoid(z)')
ax.set_xlabel('z')

# NOTE: we have 2 features – model is f_wb_x_i_vec = w1x1 + w2x2 + b

# as you know, a d.b. can be found for when
# 1. z = wx + b = 0

# because when wx + b is > 0, g(z) > or = 0.5 and pred/y-hat = 1
# (vice versa)

# say you've found w1 = 1, w2 = 1, b = -3
# that minimized the cost function (will see later)

# so, model becomes

# x1 + x2 - 3 = z

# so, x1 + x2 - 3 = 0 is what yields d.b.

# so to plot it, yes, you plot x1 and x2


plt.show()