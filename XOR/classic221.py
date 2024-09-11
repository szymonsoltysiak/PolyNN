# x ‾‾‾‾‾‾‾\‾‾/‾‾‾‾‾‾‾‾\‾‾/‾‾‾‾‾‾‾‾\
#           \/          \/          \________ output
#           /\          /\          /
# y _______/__\________/__\________/

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = np.array([[0], [1], [1], [0]])

np.random.seed(42)
input_layer_neurons = 2
hidden_layer_neurons = 2
output_neurons = 1

weights_input_hidden = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
weights_hidden_output = np.random.uniform(size=(hidden_layer_neurons, output_neurons))

bias_hidden = np.random.uniform(size=(1, hidden_layer_neurons))
bias_output = np.random.uniform(size=(1, output_neurons))

lr = 0.1

for epoch in range(10000):
    hidden_layer_activation = np.dot(inputs, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_activation)

    output_layer_activation = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    predicted_output = sigmoid(output_layer_activation)

    error = outputs - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)

    error_hidden_layer = d_predicted_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    weights_hidden_output += hidden_layer_output.T.dot(d_predicted_output) * lr
    bias_output += np.sum(d_predicted_output, axis=0, keepdims=True) * lr
    weights_input_hidden += inputs.T.dot(d_hidden_layer) * lr
    bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * lr

print("Output:")
print(predicted_output)
print("\n----------------------------------")

print("\nW1:")
print(weights_input_hidden)

print("\nB1:")
print(bias_hidden)

print("\nW2:")
print(weights_hidden_output)

print("\nB2:")
print(bias_output)

x_min, x_max = 0, 1
y_min, y_max = 0, 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
grid = np.c_[xx.ravel(), yy.ravel()]

hidden_layer_activation = np.dot(grid, weights_input_hidden) + bias_hidden
hidden_layer_output = sigmoid(hidden_layer_activation)
output_layer_activation = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
grid_predictions = sigmoid(output_layer_activation)
grid_predictions = grid_predictions.reshape(xx.shape)

plt.figure(figsize=(6, 6))
plt.contourf(xx, yy, grid_predictions, levels=[0, 0.5, 1], colors=['skyblue', 'darkgreen'], alpha=0.6)
plt.scatter([0,1], [0,1], edgecolors='k', color='skyblue', marker='o', s=200)
plt.scatter([1,0], [0,1], edgecolors='k', color='darkgreen', marker='o', s=200)
plt.title('Output')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
