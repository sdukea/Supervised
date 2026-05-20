# exploring situations where overfitting may occur
# and some solutions

import matplotlib.pyplot as pltplt

from ipywidgets import Output

# as implementation has helper functions, cannot see overfitting

# let's DIY

from sklearn.datasets import make_moons

# we're demonstrating overfitting on a dataset that has a complex relationship
# so data is like:
# two crescent moons side by side where one is facing upward and another facing downward

# so if we'd want to fit a model to this, we'd have to build a careful model (polynomial, of course)
# where there are high chances of overfitting

# that's why there's this example dataset
# 1. to be able to fit a complex polyn. model to the dataset
# 2. have high risk of overfitting so that we can see it and solve it



X, y = make_moons(
    n_samples=300,
    noise=0.25,
    random_state=42
)