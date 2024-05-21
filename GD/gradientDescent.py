import csv
import numpy as np
import math
import matplotlib.pyplot as plt


def fn_hypothesis(w, b, x):
    return w * x + b

def fn_cost(size, Y, Y_new):
    return math.sqrt(1 / (2 * size)) * np.sum((Y_new - Y) ** 2)
    # return (1 / (2 * size)) * np.sum((Y_new - Y) ** 2)

def gradient_descent(X, Y, w, b, learning_rate, num_iterations):
    data_len = len(X)
    cost_history = []

    for i in range(num_iterations):
        y_predicted = fn_hypothesis(w, b, X)

        dw = (1 / data_len) * np.dot(X, (y_predicted - Y))
        db = (1 / data_len) * np.sum(y_predicted - Y)

        w -= learning_rate * dw
        b -= learning_rate * db

        cost = fn_cost(data_len, Y, y_predicted)
        cost_history.append(cost)

        # if i % 100 == 0:  # Print cost every 100 iterations
        #     print(f"Iteration {i+1}: Cost {cost}")
    return w, b, cost_history

try:
    X = []
    Y = []
    with open('data.csv', newline='') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            X.append(float(row[0]))
            Y.append(float(row[1]))
    
    X = np.array(X)
    Y = np.array(Y)
    
    # Normalize the 'km' feature
    X_mean = np.mean(X)
    X_std = np.std(X)
    X_norm = (X - X_mean) / X_std
    
    # Initialize parameters
    w = 0
    b = 0
    learning_rate = 0.001
    num_iterations = 100000
    
    # Perform gradient descent
    w, b, cost_history = gradient_descent(X_norm, Y, w, b, learning_rate, num_iterations)
            
    # Final parameters
    print(f"Final weight: {w}, Final bias: {b}")
    
    # Prediction on new data (using normalized values)
    new_km = 67000  # Example new data point
    new_km_norm = (new_km - X_mean) / X_std
    predicted_price = fn_hypothesis(w, b, new_km_norm)
    print(f"Predicted price for {new_km} km: {predicted_price}")

    Y_pred = fn_hypothesis(w, b, X_norm)
    
    # Calculate R² score
    SS_tot = np.sum((Y - np.mean(Y)) ** 2)
    SS_res = np.sum((Y - Y_pred) ** 2)
    r2_score = 1 - (SS_res / SS_tot)
    r2_percentage = r2_score * 100

    print(f"R² Score: {r2_score}")
    print(f"Model Precision: {r2_percentage}%")

    plt.figure(figsize=(10, 6))
    
    # Plot the original data points
    plt.scatter(X, Y, color='blue', label='Original data')
    
    # Plot the regression line
    X_range = np.linspace(min(X), max(X), 100)
    X_range_norm = (X_range - X_mean) / X_std
    Y_range_pred = fn_hypothesis(w, b, X_range_norm)
    plt.plot(X_range, Y_range_pred, color='red', label='Regression line')
    
    plt.xlabel('Kilometers')
    plt.ylabel('Price')
    plt.title('Linear Regression Model')
    plt.legend()
    plt.show()
    
except ValueError:
    print("Error: Argument passed is not a valid float number")
except FileNotFoundError:
    print("Error: The file 'data.csv' was not found.")
