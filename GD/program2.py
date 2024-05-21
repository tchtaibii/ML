import os
import sys
import csv
import json
import numpy as np
from functions.gradient_descent import gradient_descent

def main():
    try:
        X = []
        Y = []
        with open('data.csv', newline='') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                X.append(float(row[0]))
                Y.append(float(row[1]))
        
        X = np.array(X)
        Y = np.array(Y)
        
        # Normalize the 'km' feature
        X_mean = np.mean(X)
        X_std = np.std(X)
        X_norm = (X - X_mean) / X_std
        print(f'X_mean = {X_mean}')
        print(f'X_std = {X_std}')
        print(f'X_norm = {X_norm}')
        
        # Initialize parameters
        theta0 = 0
        theta1 = 0
        learning_rate = 0.01
        num_iterations = 1000
        
        # Perform gradient descent
        theta0, theta1, cost_history = gradient_descent(X_norm, Y, 0, 0, learning_rate, num_iterations)

        variables = 'variables.json'
        if os.path.exists(variables):
            os.remove(variables)
        file = open(variables, mode='w')
        json.dump({
            'theta0': theta0,
            'theta1': theta1,
            'X_mean': X_mean,
            'X_std': X_std,
        }, file, indent=4)

        print(f'theta0 = {float(theta0):.2f}')
        print(f'theta1 = {float(theta1):.2f}')
        print(f'cost   = {float(cost_history[-1]):.2f}')
    finally:
        file.close


if __name__ == '__main__':
    main()