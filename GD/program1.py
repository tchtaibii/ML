import os
import csv
import numpy
import json
from functions.hypothesis import hypothesis

def check_input(value):
    if value.isdigit():
        return int(value)
    else:
        try :
            value = float(value)
            return value
        except ValueError:
            raise ValueError('please enter a valid number')

def main():
    try:
        theta0 = 0
        theta1 = 0
        X_mean = 0
        X_std = 1
        variables = 'variables.json'

        if os.path.exists(variables):
            file = open(variables, mode='r')
            data_read = json.load(file)
            theta0 = data_read['theta0']
            theta1 = data_read['theta1']
            X_mean = data_read['X_mean']
            X_std = 1 if data_read['X_std'] == 0 else data_read['X_std']

        x = input('Enter km of the car : ')
        x = check_input(x)
        x = (x - X_mean) / X_std

        y = hypothesis(theta0, theta1, x)
        print(y)
    except ValueError as e:
        print("check your data")

if __name__ == '__main__':
    main()