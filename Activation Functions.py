import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


x = np.linspace(-10, 10, 50)
p = sigmoid(x)
plt.xlabel("x")
plt.ylabel("Sigmoid(x)")
plt.plot(x, p)
plt.show()


def RelU(x):
    return np.maximum(0, x)


x = np.linspace(-10, 10, 100)
p = RelU(x)
plt.xlabel("x")
plt.ylabel("Relu(x)")
plt.plot(x, p)
plt.show()


def Tanh(x):
    return np.tanh(x)


x = np.linspace(-10, 10, 100)
p = Tanh(x)

plt.xlabel("x")
plt.ylabel("Tanh(x)")
plt.plot(x, p)
plt.show()


def SoftMax(x):
    return np.exp(x) / np.sum(np.exp(x))


x = np.linspace(-10, 10, 100)
p = SoftMax(x)

plt.xlabel("x")
plt.ylabel("SoftMax(x)")
plt.plot(x, p)
plt.show()
