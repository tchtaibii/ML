from .hypothesis import hypothesis
from .rmse import rmse
import numpy as np

def hypothesis_array(w,b,X):
    y_predicted = []
    for x in X:
        y_predicted.append(hypothesis(w, b, x))
    return y_predicted

def gradient_descent(X, Y, w, b, learning_rate, num_iterations):
    data_len = len(X)
    cost_history = []

    for i in range(num_iterations):
        y_predicted = hypothesis_array(w, b, X)

        dw = (1 / data_len) * np.sum(X * (y_predicted - Y))
        db = (1 / data_len) * np.sum(y_predicted - Y)
        w -= learning_rate * dw
        b -= learning_rate * db

        cost = rmse(data_len, Y, y_predicted)
        cost_history.append(cost)
        
    return w, b, cost_history