
import math
import numpy as np

def rmse(data_len, Y, y_predicted):
    squared_error = np.sum((Y - y_predicted) ** 2)
    mean_squared_error = squared_error / data_len
    rmse_value = np.sqrt(mean_squared_error)
    return rmse_value