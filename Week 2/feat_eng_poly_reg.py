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
# |
# so if y = f(x) + ϵ in the real world case, how is y = f(x) even relevant to talk about?
# think of f(x) – the ground truth universal function – as the average or expected outcome for a 
# given x; formally given by y = f(x) = E[y∣x] 
# and our model y-hat tries to learn/approximate to f(x) and not f(x) + ϵ because
# ϵ is random and cannot be learned; we know it will exist and bother us no matter what but it cannot
# be learned in any way because it keeps on changing as there are infinitely many possible features/
# variables that will affect 'y' and it is impossible to grab hold of all these variables no matter what
# takeaway: the world is not perfectly predictable, your features don't capture everything, so outcome/ground truth 
# (y) = signal (f(x)) + noise (ϵ) and model can only learn from signal
# so y = f(x) + ϵ is the real world equation; when you collect data and you have features with associated
# true output labels
# y = f(x) -> the systematic part; what's predictable from 'x'
# ϵ -> random part; not predictable by any means as there are infinitely many possible variables that
# could undergo change and affect 'y' and we cannot account to grab hold of all these variables to 
# estimate 'y'
# so y = f(x) is relevant because it is the only part you can ever hope to learn
# you can't learn ϵ:
# - it's random
# - changes every time
# - not tied to 'x' in a consistent way (if so, it would be part of f(x) – the function)
#   but still affects 'y'
#   knowing 'x' does NOT let you predict ϵ in any way and its totally random
#   if you know x = 5, you cannot predict if ϵ – totally existent and bothers us no matter what – will
#   push y up or down
#   if you knew what ϵ is, then that's not noise anymore – it'll be part of the function
#   and the function knows how y will change when 'x' changes
# |
# let's talk about the output label 'y' now:
# - for the same feature value/vector x, you might have different output labels
#   so, when x = 5, y = 78, when x = 5 again (somewhere down the rows in data), y = 80 now and when
#   x = 5 again after some while down in the data, y = 83 now
#   so you understand this – ϵ affects what 'y' is because everything that affects 'y' but is not
#   predictable from x is what ϵ is (noise)
#   there are missing features/variables that are affecting y that we still haven't accounted for
#   and x alone isn't predictive/deterministic of what 'y' is
# - and this can totally happen – you will see for a single feature value/vector x, different set
#   of possible values for 'y' because of ϵ noise

# NOTE: To explain polynomial regression and the use of 'y' = 1 + x^2 here
# - we use 'y = 1 + x^2' as the ground truth function to 
# - produce/generate synthetic data to work with
# - i.e. output labels for a given input/variable x that we have
# - so we defined this universal rule just for the sake of demonstration
# - and we generate data from it
# - and we know what f(x) is exactly
# - so y = 1 + x^2 is literally the true underlying relationship for the 'x' we have
# - and the y we synthetically generated
# - actually, y = f(x) stays unknown but here
# - we don't have data of output labels and features x to be in a state where
# - f(x) should stay unknown and we should actually model the relationship to generalize to new
# - unseen data (make a new model that learns the ground truth function and generalize well)
# - for demonstration purposes, we need some output labels and feature values 'x' because
# - any way, the data that we get from the real world will have a ground truth function relating
# - the features we have and the output labels associated (plus some ϵ)
# - so we keep this idea true; we don't have 'x' or 'y' that relate to each other right now
# - so why not create a rule that we can reverse-use to produce data of 'y' and we can create some
# - corresponding 'x' and yay – we have some data to use now (synthetic)
# - yes, to simulate real world data, y = 1 + x^2 + ϵ (noise) could be used but we don't want to do 
# - that now – we just want to create a noise-free world for now for easy demonstration and understanding
# NOTE: This y = 1 + x^2 is the ground truth WE CREATED to reverse-create data set i.e.
# - the output labels y that we created for some 'x' were synthetically reverse-created with our
# - pre-made ground truth 
# - this way of defining a rule first and then reverse-creating the data set that abides to this
# - rule makes this ground truth function rule NOT generalize-able to unseen inputs
# - because ground truth function is: the actual rule that generated the data you're working with
# - so, DATA first and then GROUND TRUTH is adapted/realized/approximated to
# - here, its GROUND TRUTH definition first and then DATA creation using it next
# - either way, this function relates to the data set you're working with and so we can still
# - call this ground truth even though we've taken the reverse way
# - 


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