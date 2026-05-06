"""
Goals
In this lab you will:
- explore feature engineering and polynomial regression which allows you to use the machinery of linear regression to fit very complicated, even very non-linear functions.
"""

import numpy as np
import matplotlib.pyplot as plt
from lab_utils_multi import zscore_normalize_features, run_gradient_descent_feng

np.set_printoptions(precision=2)

# NOTE some great understanding:
# a model: your guessing machine
# - its a function with adjustable parameters in it
# - when you find the best parameters that the model will have (after training/G.D./other optimization)
# - then your model 'predicts' and returns a prediction; returns y-hat for each training example

# in U.L.R.:
# - your model is – wx + b
# - and when you find the right w and b that minimize the cost function J(w, b) (w not a vector),
# - then your model returns prediction/y-hat

# NOTE: a pattern in regression is how y (actual label) changes as the input features change
# NOTE: there are three 'y's you have to know:
# 1. y = f(x)
# eg: y = 1 + x^2
# this is the ground truth function; this is essentially a function that tells you how the
# input features you have (like X_train) are related to the corresponding ground truth labels you
# have (like y_train)
# this is usually/in most cases unknown because you just get the actual labels only (with features as 
# well that correspond) and determining a relationship is totally undoable right off the bat
# the ground truth function is a rule that applies to any valid input – not just the training data
# works for all unseen and seen/trained/known inputs as well and is not just restricted to the ones
# in your training dataset
# the dataset is actually a finite sample from that underlying function
# what does this mean:
# 1. there are infinitely many inputs i.e. feature values/vectors
# 2. in regression, feature values are all real-values and hence there are infintely many of them
# 3. so out of the many possible feature values/vectors there exists, you only have a finite set in
# training; that's why when you have features values (just an eg.) like 1, 2, 3, 5, 6, 7, 9, 11 
# and their corresponding labels you'd think why don't we have the numbers in between 
# (a doubt you got eariler) it's because the space of possible inputs is mathematically infinite, 
# not just practically large. if you'd get all numbers from 1 to 100 as well, you'd still not have 
# the feat. values because you should've also got 1.1, 1.2, 1.3 and if so, you should've also got
# 1.05, 1.025, 1.0175 and it just keeps going on – its infinite values for features.
# that's why the data set is alwaya a finite sample
# |
# in your dataset, you might think that for every feature value/vector x, you apply the ground truth
# function and you get the associated true label 'y';
# but in reality, this is only partially true.
# the 1. equation, in practice, will actually be:
# y = f(x) + ϵ
# where ϵ is the noise or everything affecting y – your true output label – that is not captured
# by your input x (any of them)
# say x = hours studied (just one feature) and y = exam score
# you might think y = f(x) (some ground truth function that we assume is universal)
# but in reality, exam score is also affected by sleep quality, stress, luck, question difficulty
# and so forth, which isn't captured by our 'x'
# so what happens is this: for the same 'x'
# study 5 hours –> score 78
# study 5 hours -> score 83
# study 5 hours -> score 80
# our 'x' stays the same but in the real world, 'x' and ϵ – the features that our 'x' does not capture
# – affect 'y' the score
# the scores are different because even though 'x' stays unchanged, 'x' and some other feature that we
# don't know and haven't accounted for is changing and that affects the score
# and this other, unaccounted for features are given by ϵ – the noise 


# so, ground truth – continous, universal rule that exists for all seen/unseen input and what the output
# should be
# training data – a few observed points from it
# and the model that you train – y-hat = f_parameters(x) (or) model(x)
# will try to approximate to this ground truth function, but it may fail
# in reality, not enough data, noise and bad modelling will result in only close approximations#
# but the goal is always: y-hat (model) ≈ f(x)

#

# 2. y_train
# as mentioned just eariler in 1.
# this is the labels/values for each training example you hav 

# in M.L.R.:
# - your model is – w0x0 + w1x1 + w2x2 + ... + wnxn + b
# - and after finding the best possible parameters w0 to wn,
# - your model returns y-hat for each training example

# in P.L.R.
# – your model is; for one input feature x and degree d
# - 