import csv
import numpy as np
from PIL import Image, ImageDraw
import math

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
    
    img_width = 800
    img_height = 600
    img = Image.new('RGB', (img_width, img_height), color='white')
    draw = ImageDraw.Draw(img)

    # Plot the data points
    scaled_X = (X_norm - X_norm.min()) / (X_norm.max() - X_norm.min()) * img_width
    scaled_Y = (Y - Y.min()) / (Y.max() - Y.min()) * img_height
    for x, y in zip(scaled_X, scaled_Y):
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill='blue')

    # Plot the regression line
    x_values = np.linspace(scaled_X.min(), scaled_X.max(), 100)
    y_values = w * (x_values / img_width) * (X.max() - X.min()) / X_std + b
    scaled_y_values = (y_values - Y.min()) / (Y.max() - Y.min()) * img_height
    for i in range(len(scaled_y_values) - 1):
        draw.line((x_values[i], scaled_y_values[i], x_values[i + 1], scaled_y_values[i + 1]), fill='red')

    # Save the image
    img.save('linear_regression_plot.png')
        
    # Final parameters
    print(f"Final weight: {w}, Final bias: {b}")
    
    # Prediction on new data (using normalized values)
    new_km = 67000  # Example new data point
    new_km_norm = (new_km - X_mean) / X_std
    predicted_price = fn_hypothesis(w, b, new_km_norm)
    print(f"Predicted price for {new_km} km: {predicted_price}")
    
except ValueError:
    print("Error: Argument passed is not a valid float number")
except FileNotFoundError:
    print("Error: The file 'data.csv' was not found.")
