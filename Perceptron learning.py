import numpy as np


class Perceptron:
    def __init__(self, input_size, lr=0.01):
        """Initialize the perceptron with given input size and learning rate."""
        self.W = np.zeros(input_size + 1)
        """Array of zeros is created by adding (input_size + 1) where 1 is bias term
        which is used to increase model's performance."""
        self.lr = lr

    def activation_fn(self, x):
        """Activation function."""
        return 1 if x >= 0 else 0

    def predict(self, x):
        """Predict the output for a given input."""
        x = np.insert(x, 0, 1)
        """x=array, 0=index value at which the element will be added, 1=value to be added."""
        z = self.W.T.dot(x)
        """Dot product between transposed weight vector and input vector is found out."""
        a = self.activation_fn(z)
        """Activation function is applied on z."""
        return a

    def train(self, X, Y, epochs):
        """Train the perceptron."""
        for _ in range(epochs):
            for i in range(Y.shape[0]):
                x = X[i]
                y = self.predict(x)
                e = Y[i] - y
                self.W = self.W + self.lr * e * np.insert(x, 0, 1)


# Define the input data and labels
X = np.array([
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # 0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # 1
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # 2
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 3
    [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],  # 4
    [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],  # 5
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],  # 6
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],  # 7
    [0, 0, 0, 0, 0, 0, 1, 0, 1, 1],  # 8
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],  # 9
])
Y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

# Create the perceptron and train it
perceptron = Perceptron(input_size=10, lr=0.1)
perceptron.train(X, Y, epochs=1000)

# Test the perceptron on some input data
test_X = np.array([
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # 0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # 1
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # 2
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 3
    [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],  # 4
    [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],  # 5
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],  # 6
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],  # 7
    [0, 0, 0, 0, 0, 0, 1, 0, 1, 1],  # 8
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],  # 9
])

for i in range(test_X.shape[0]):
    x = test_X[i]
    y = perceptron.predict(x)
    print(f'{x} is {"even" if y == 0 else "odd"}')
