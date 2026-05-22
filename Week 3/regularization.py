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
# yeah, you also learn 'e' (epsilon) which is noise that cannot be predicted but exists no matter what

# one way to think: overfitting = model complexity / amount of data

