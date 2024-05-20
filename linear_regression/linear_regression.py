import csv
import sys

def calculate_slope(X, Y):
    n = len(X)
    sum_xy = sum(x * y for x, y in zip(X, Y))
    sum_x = sum(X)
    sum_y = sum(Y)
    sum_x_squared = sum(x ** 2 for x in X)
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x ** 2)
    return slope

def calculate_y_intercept(Y,X, slope):
    n = len(X)
    return (sum(Y) / n) - slope * (sum(X) / n)


if len(sys.argv) != 2:
    exit(0)
try:
    float_number = float(sys.argv[1])
    x = []
    y = []
    with open('data.csv', newline='') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            x.append(float(row[0]))
            y.append(float(row[1]))
    slope = calculate_slope(x, y)
    y_intercept = calculate_y_intercept(y, x, slope)
    linear_regression = slope * float_number + y_intercept
    print(linear_regression)
    
except ValueError:
    print("Error: Argument passed is not a valid float number")
    