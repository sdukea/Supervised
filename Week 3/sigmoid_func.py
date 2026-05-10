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

