# Simple Linear Regression from Scratch

def mean(values):
    return sum(values) / float(len(values))


def covariance(x, mean_x, y, mean_y):
    covar = 0.0
    for i in range(len(x)):
        covar += (x[i] - mean_x) * (y[i] - mean_y)
    return covar


def variance(values, mean):
    return sum((x - mean) ** 2 for x in values)


def coefficients(x, y):
    mean_x, mean_y = mean(x), mean(y)
    b1 = covariance(x, mean_x, y, mean_y) / variance(x, mean_x)
    b0 = mean_y - b1 * mean_x
    return b0, b1


def simple_linear_regression(train_x, train_y, test_x):
    predictions = list()
    b0, b1 = coefficients(train_x, train_y)
    for x in test_x:
        y_pred = b0 + b1 * x
        predictions.append(y_pred)
    return predictions

# Example Usage:


# Training Data
train_x = [1, 2, 3, 4, 5]
train_y = [2, 4, 5, 4, 5]

# Test Data
test_x = [6, 7, 8, 9, 10]

# Making Predictions
predictions = simple_linear_regression(train_x, train_y, test_x)

# Displaying Predictions
for i in range(len(predictions)):
    print(f"X={test_x[i]}, Predicted Y={predictions[i]}")


    """# Linear Regression from Scratch

class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Initialize weights and bias
        self.weights = [0.0] * (len(X[0]) + 1)
        self.bias = 0.0

        # Perform gradient descent
        for epoch in range(self.epochs):
            predictions = self.predict(X)
            errors = [predictions[i] - y[i] for i in range(len(y))]

            # Update weights and bias using gradients
            for i in range(len(X[0])):
                gradient = sum([errors[j] * X[j][i] for j in range(len(X))]) / len(X)
                self.weights[i] -= self.learning_rate * gradient

            # Update bias
            gradient_bias = sum(errors) / len(X)
            self.bias -= self.learning_rate * gradient_bias

    def predict(self, X):
        predictions = []
        for i in range(len(X)):
            prediction = self.bias + sum([self.weights[j] * X[i][j] for j in range(len(X[0]))])
            predictions.append(prediction)
        return predictions

# Example usage:
if __name__ == "__main__":
    # Sample data
    X_train = [[1], [2], [3], [4], [5]]
    y_train = [2, 4, 5, 4, 5]

    # Instantiate and fit the model
    model = LinearRegression(learning_rate=0.01, epochs=1000)
    model.fit(X_train, y_train)

    # Make predictions
    X_test = [[6], [7]]
    predictions = model.predict(X_test)

    print("Predictions:", predictions)

    """
    

