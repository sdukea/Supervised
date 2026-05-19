# training a logistic reg. model using SKLearn
# (the boring part)

import numpy as np
import matplotlib.pyplot as plt

# load data

X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1])

# fit the model

from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression()
lr_model.fit(X, y)

# do the necessary setup with the data i.e.
# study the data and memorize the statistics needed for normalization
# so, .fit() finds mean of each column and S.D. for each column AND KEEPS IT READY
# only then can it transform (if we specify it like .transform() or inside .fit_transform())


# make predictions

y_pred = lr_model.predict(X)
# yes, using the same tr. set to predict

# for each tr. eg., y_pred will hold a predicted value (0 or 1 now; does all of the conversion to binary
# for us)

print(y_pred)