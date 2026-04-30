# implementing cost function for l.r. with one variable

import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

def compute_cost(x, y, w, b): # only computing cost, not trying to compute model predictions

    m = x.shape[0]

    total_cost = 0

    for i in range(m):

        f_wb = w * x[i] + b
        # just find first prediciton - don't need to store it like in 'model_representation'

        cost = (f_wb - y[i]) ** 2

        total_cost = total_cost + cost

    final_cost = ((1 / 2 * m)) * total_cost
    
    return final_cost

cost = compute_cost(x_train, y_train, w=200, b=100)

print(cost)