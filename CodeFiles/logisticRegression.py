# this is logistic regression file

# === libraries ===
import numpy as np


# === sigmoid ===
# description: the S shape function which gives us one or zero.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# === Logistic Regression ===
class LogisticRegression():

    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    # X ===> Training inputs samples
    # y ===> Target values    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # this is gradient descent
        for _ in range(self.n_iters):
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(linear_pred)

            # formula: dw = (1 / n_samples) * X.T * (predictions - y)
            # formula: db = (1 / n_samples) * sum(predictions - y)
            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

    # labeling the data
    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_pred)
        class_pred = [0 if y <= 0.5 else 1 for y in y_pred]
        return class_pred
