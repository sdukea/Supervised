import numpy as np
import matplotlib.pyplot as plt

# If one way to reduce overfitting is 
# to add more training examples, then why is another way, 
# to reduce overfitting, regularization or reduce the number/impact of features we have? 
# Isn't that counter-intuitive?

# the problem in overfitting:
# - flexibility: how easily the model can bend itself to fit data
# - a very flex. model can create extremely complicated decision boundaries or curves
# More flexibility means:
# better ability to capture complex real patterns
# but also better ability to memorize noise

# your tr. data is the only evidence your model has about the real world
# but data is: limited, noisy and incomplete
# with only a few examples, there is little-to-no evidence of the underlying pattern between our
# feature values/vector vs target output

# and because of this, a highly flexible model may start believing that the
# random fluctuations/noise in the training data are the actual patterns
# this happens because we do not yet have enough data/evidence
# to reliably distinguish:
# - true signal
# - from random coincidence/noise

# with only a small dataset, noise can accidentally look like
# a meaningful pattern

# but with much larger amounts of data:
# - true patterns tend to appear consistently
# - random noise becomes less consistent/statistically weaker

# therefore, even when noise exists,
# the underlying relationship becomes easier to detect

# so more data helps the model separate:
# signal vs noise

# but NOTE: you're still trying to fit a complex, flexible model to this highly evident
# data - you still have chances of overfitting
# and that's why you have to reduce model complexity as well

# one way to think: overfitting = model complexity / amount of data

# so we can solve overfitting by increasing denominator as it reduces the amount of overfitting
# (smaller denom., smaller overfitting)
# |
# and one more way is also to decrease numerator – decrease model complexity

# and this is exactly what explains regularization

# you have added more training examples to the data - that's cool
# you now have an improved visibility of the signal
# but this larger dataset still has noise/inconsistency/fluctuations

# more data makes the underlying pattern/generalizable pattern easier to detect by
# the model

# but a highly flexible, complex model may STILL learn the noise/fluctations in this pattern-evident
# data

# so you have to ensure your model isn't complex

# and that's why you regularize – you encourage the model to prefer smoother/simpler relationships
# without trying to learn noise and only stick to signal/generalizable pattern in the data

# more data: increases signal/more correct evidence
# regularization: controls model complexity/flexibility

# –––

# adding regularization

# cost functions with reg.

# c.f. for regularized linear reg.

def compute_cost_linear_reg(X, y, w, b, lambda_ = 1):

    m = X.shape[0]

    n = len(w)

    cost = 0

    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b

        cost = cost + (f_wb_i - y[i])**2

    cost = cost / (2*m)

    reg_cost = 0

    for j in range(n):
        reg_cost += (w[j]**2)
    
    reg_cost = (lambda_/(2*m)) * reg_cost

    total_cost = cost + reg_cost

    return total_cost


