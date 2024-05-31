import numpy as np

# Define sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define derivative of sigmoid activation function
def sigmoid_derivative(x):
    return x * (1 - x)

# Define training inputs
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# Define training outputs
outputs = np.array([[0], [1], [1], [0]])

# Define random weights for hidden layer
hidden_weights = np.random.uniform(size=(2, 3))

# Define random weights for output layer
output_weights = np.random.uniform(size=(3, 1))

# Set learning rate
learning_rate = 0.7

# Train the network
for i in range(10000):

    # Forward propagation
    hidden_layer_activation = sigmoid(np.dot(inputs, hidden_weights))
    output_layer_activation = sigmoid(np.dot(hidden_layer_activation, output_weights))

    # Calculate error
    error = outputs - output_layer_activation

    # Backpropagation
    output_layer_error = error * sigmoid_derivative(output_layer_activation)
    hidden_layer_error = np.dot(output_layer_error, output_weights.T) * sigmoid_derivative(hidden_layer_activation)

    # Update weights
    output_weights += np.dot(hidden_layer_activation.T, output_layer_error) * learning_rate
    hidden_weights += np.dot(inputs.T, hidden_layer_error) * learning_rate

# Test the network
hidden_layer_activation = sigmoid(np.dot(inputs, hidden_weights))
output_layer_activation = sigmoid(np.dot(hidden_layer_activation, output_weights))
print(output_layer_activation)
