import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris  # iris dataset is imported

# Load iris dataset
iris = load_iris()  # iris dataset is loaded

# Extract sepal length and petal length features
X = iris.data[:, [0, 2]]  # data is feature vector 0=1st feature and 2= 3rd feature
y = iris.target  # target contains labels

# Setosa is class 0, versicolor is class 1
y = np.where(y == 0, 1, 0)  # Labels are converted from multiclass format to binary format using where()

# Initialize weights and bias
w = np.zeros(2)
b = 0

# Set learning rate and number of epochs
lr = 0.1
epochs = 50  # Epochs refer to the number of times the entire dataset is passed forward and backward through the neural network during the training process

# Define perceptron function
def perceptron(x, w, b):
    # Calculate weighted sum of inputs
    z = np.dot(x, w) + b
    # Apply step function
    return np.where(z >= 0, 1, 0)

# Train the perceptron
for epoch in range(epochs):
    for i in range(len(X)):
        x = X[i]
        target = y[i]
        output = perceptron(x, w, b)
        error = target - output
        w += lr * error * x
        b += lr * error

# Plot decision boundary
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
Z = perceptron(np.c_[xx.ravel(), yy.ravel()], w, b)  # ravel flatten the 2D arrays xx and yy into 1D arrays, which can then be concatenated column-wise using np.c_[]
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)

# Plot data points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Petal length')
plt.title('Perceptron decision regions')
plt.show()
