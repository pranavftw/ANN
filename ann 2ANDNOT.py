import numpy as np

# Function for checking threshold value
def linear_threshold_gate(dot, T):
    """Returns the binary threshold output"""
    if dot >= T:
        return 1
    else:
        return 0


# Matrix of inputs
input_table = np.array([
    [0, 0],  # Both no
    [1, 1],  # Both yes
    [0, 1],  # One no, one yes
    [1, 0]  # One yes, one no
])

print(f'Input table:\n{input_table}')

weights = np.array([1, -1])

dot_products = input_table @ weights
T = 1

for i in range(0, 4):
    activation = linear_threshold_gate(dot_products[i], T)
    print(f'Activation: {activation}')
