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



# one way to think: overfitting = model complexity / amount of data


# –––

# adding regularization

# cost functions with reg.

# c.f. for reg. linear reg.

def compute_cost_linear_reg(X, y, w, b, lambda_ = 1):

    m, n = X.shape

    