import os
# Import necessary libraries
import numpy as np
import pandas as pd

# Load dataset
data = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Student-University.csv'), header=None, names=['Exam1', 'Exam2', 'Admitted'])

# Preprocess data: standardise the exam scores (zero mean, unit variance).
# Without this, exp(-z) overflows and gradient descent never converges.

# Convert DataFrame to NumPy arrays
X_raw = np.array(data[['Exam1', 'Exam2']])
y = np.array(data['Admitted'])
mu, sigma = X_raw.mean(axis=0), X_raw.std(axis=0)
X = (X_raw - mu) / sigma

# Add a column of ones to X for the intercept term
X = np.column_stack((np.ones(X.shape[0]), X))

# Initialize parameters
theta = np.zeros(X.shape[1])

# Hyperparameters
alpha = 0.1  # learning rate
iterations = 5000

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cost function
def cost_function(X, y, theta):
    m = len(y)
    h = np.clip(sigmoid(X @ theta), 1e-12, 1 - 1e-12)
    J = (1 / m) * (-y @ np.log(h) - (1 - y) @ np.log(1 - h))
    return J

# Gradient descent
def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    J_history = []

    for _ in range(iterations):
        h = sigmoid(X @ theta)
        gradient = (1 / m) * X.T @ (h - y)
        theta -= alpha * gradient
        J_history.append(cost_function(X, y, theta))

    return theta, J_history

# Run gradient descent
theta, J_history = gradient_descent(X, y, theta, alpha, iterations)

# Convert the parameters back to the original (unscaled) exam-score units
theta_orig = np.zeros(3)
theta_orig[1:] = theta[1:] / sigma
theta_orig[0] = theta[0] - np.sum(theta[1:] * mu / sigma)

# Print the learned parameters
print("Learned Parameters (original score units):")
print("Theta 0 (Intercept):", theta_orig[0])
print("Theta 1 (Exam1):", theta_orig[1])
print("Theta 2 (Exam2):", theta_orig[2])

# Predictions
predictions = np.round(sigmoid(X @ theta))

# Accuracy
accuracy = np.mean(predictions == y)
print("Accuracy:", accuracy)
