#Certainly! Below are simple implementations of Ridge and Lasso regression from scratch in Python. These implementations include the training functions, prediction functions, and a demonstration on synthetic data.


import numpy as np

class RidgeRegression:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.coefficients = None

    def fit(self, X, y):
        # Add a column of ones for the intercept term
        X_with_intercept = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

        # Calculate coefficients using Ridge regression
        identity_matrix = np.eye(X_with_intercept.shape[1])
        self.coefficients = np.linalg.inv(X_with_intercept.T.dot(X_with_intercept) +
                                          self.alpha * identity_matrix).dot(X_with_intercept.T).dot(y)

    def predict(self, X):
        # Add a column of ones for the intercept term
        X_with_intercept = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

        # Make predictions using learned coefficients
        predictions = X_with_intercept.dot(self.coefficients)

        return predictions

class LassoRegression:
    def __init__(self, alpha=1.0, max_iterations=1000, tol=1e-4):
        self.alpha = alpha
        self.max_iterations = max_iterations
        self.tol = tol
        self.coefficients = None

    def soft_threshold(self, x, threshold):
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

    def fit(self, X, y):
        # Add a column of ones for the intercept term
        X_with_intercept = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

        # Initialize coefficients
        self.coefficients = np.zeros(X_with_intercept.shape[1])

        for _ in range(self.max_iterations):
            # Store current coefficients for convergence check
            old_coefficients = self.coefficients.copy()

            # Update coefficients using coordinate descent
            for j in range(X_with_intercept.shape[1]):
                X_j = X_with_intercept[:, j]
                residuals = y - X_with_intercept.dot(self.coefficients) + self.coefficients[j] * X_j
                rho = X_j.dot(residuals)
                self.coefficients[j] = self.soft_threshold(rho, self.alpha) / (X_j.dot(X_j))

            # Check for convergence
            if np.linalg.norm(self.coefficients - old_coefficients) < self.tol:
                break

    def predict(self, X):
        # Add a column of ones for the intercept term
        X_with_intercept = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

        # Make predictions using learned coefficients
        predictions = X_with_intercept.dot(self.coefficients)

        return predictions

# Example usage:
if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)

    # Fit Ridge regression model
    ridge_model = RidgeRegression(alpha=1.0)
    ridge_model.fit(X, y)

    # Fit Lasso regression model
    lasso_model = LassoRegression(alpha=0.1)
    lasso_model.fit(X, y)

    # Make predictions
    X_test = np.array([[2]])
    ridge_predictions = ridge_model.predict(X_test)
    lasso_predictions = lasso_model.predict(X_test)

    # Display coefficients and predictions
    print("Ridge Coefficients:", ridge_model.coefficients)
    print("Ridge Predictions:", ridge_predictions)
    print("Lasso Coefficients:", lasso_model.coefficients)
    print("Lasso Predictions:", lasso_predictions)

"""
This code demonstrates how to implement Ridge and Lasso regression from scratch using simple coordinate descent. The synthetic data is generated, and both models are trained and used for predictions. Adjust the hyperparameters (`alpha`, `max_iterations`, and `tol`) as needed for your specific use case."""


"""Certainly! Below is the implementation of Ridge (L2 regularization) and Lasso (L1 regularization) regression from scratch in Python without using any external libraries. This implementation includes functions for training the models and making predictions.

```python
import numpy as np

class RidgeRegression:
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Regularization strength
        self.coefficients = None

    def fit(self, X, y):
        # Add a column of ones to X for the y-intercept term
        X_with_intercept = self.add_intercept(X)

        # Calculate coefficients using Ridge regression formula
        identity_matrix = np.eye(X_with_intercept.shape[1])
        self.coefficients = np.linalg.inv(X_with_intercept.T.dot(X_with_intercept) +
                                          self.alpha * identity_matrix).dot(X_with_intercept.T).dot(y)

    def predict(self, X):
        # Add a column of ones to X for the y-intercept term
        X_with_intercept = self.add_intercept(X)

        # Make predictions using the learned coefficients
        predictions = self.predictions(X_with_intercept)

        return predictions

    def predictions(self, X):
        # Make predictions using the learned coefficients
        predictions = X.dot(self.coefficients)

        return predictions

    def add_intercept(self, X):
        # Add a column of ones to X for the y-intercept term
        intercept = np.ones((X.shape[0], 1))
        X_with_intercept = np.concatenate((intercept, X), axis=1)

        return X_with_intercept


class LassoRegression:
    def __init__(self, alpha=1.0, max_iterations=1000, tolerance=1e-4):
        self.alpha = alpha  # Regularization strength
        self.max_iterations = max_iterations  # Maximum number of iterations for coordinate descent
        self.tolerance = tolerance  # Tolerance for convergence
        self.coefficients = None

    def fit(self, X, y):
        # Add a column of ones to X for the y-intercept term
        X_with_intercept = self.add_intercept(X)

        # Initialize coefficients to zeros
        self.coefficients = np.zeros(X_with_intercept.shape[1])

        # Perform coordinate descent
        for _ in range(self.max_iterations):
            old_coefficients = np.copy(self.coefficients)

            for j in range(1, X_with_intercept.shape[1]):  # Exclude the intercept term from regularization
                rho_j = X_with_intercept[:, j].T.dot(y - X_with_intercept.dot(self.coefficients) +
                                                     self.coefficients[j] * X_with_intercept[:, j])
                z_j = X_with_intercept[:, j].T.dot(X_with_intercept[:, j])
                self.coefficients[j] = self.soft_threshold(rho_j, self.alpha) / (z_j + 1e-6)

            if np.sum(np.abs(old_coefficients - self.coefficients)) < self.tolerance:
                break

    def predict(self, X):
        # Add a column of ones to X for the y-intercept term
        X_with_intercept = self.add_intercept(X)

        # Make predictions using the learned coefficients
        predictions = self.predictions(X_with_intercept)

        return predictions

    def predictions(self, X):
        # Make predictions using the learned coefficients
        predictions = X.dot(self.coefficients)

        return predictions

    def add_intercept(self, X):
        # Add a column of ones to X for the y-intercept term
        intercept = np.ones((X.shape[0], 1))
        X_with_intercept = np.concatenate((intercept, X), axis=1)

        return X_with_intercept

    def soft_threshold(self, rho, alpha):
        # Soft threshold function for L1 regularization
        if rho < - alpha / 2:
            return rho + alpha / 2
        elif rho > alpha / 2:
            return rho - alpha / 2
        else:
            return 0.0


# Example usage:
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    X_train = np.random.rand(100, 2)
    true_coefficients = np.array([3.0, 2.0, 1.5])
    noise = 0.1 * np.random.randn(100)
    y_train = X_train.dot(true_coefficients[1:]) + true_coefficients[0] + noise

    # Ridge Regression
    ridge_model = RidgeRegression(alpha=1.0)
    ridge_model.fit(X_train, y_train)
    ridge_predictions = ridge_model.predict(X_train)
    print("Ridge Coefficients:", ridge_model.coefficients)

    # Lasso Regression
    lasso_model = LassoRegression(alpha=0.1)
    lasso_model.fit(X_train, y_train)
    lasso_predictions = lasso_model.predict(X_train)
    print("Lasso Coefficients:", lasso_model.coefficients)
```

This code demonstrates how to implement Ridge and Lasso regression using Python. You can adjust the parameters and test the models on different datasets"""
