#  x  ‾‾‾‾‾‾‾‾‾‾‾‾‾\
#                   \
#  y  ------------------- output
#                   /
# x*y _____________/

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

inputs = np.array([[0, 0, 0],  # x,y,x*y
                   [0, 1, 0],  
                   [1, 0, 0],  
                   [1, 1, 1]]) 

outputs = np.array([[0], [1], [1], [0]])

np.random.seed(42)
input_layer_neurons = 3  
output_neurons = 1      

weights_input_output = np.random.uniform(size=(input_layer_neurons, output_neurons))

bias_output = np.random.uniform(size=(1, output_neurons))

lr = 0.1

for epoch in range(10000):
    output_layer_activation = np.dot(inputs, weights_input_output) + bias_output
    predicted_output = sigmoid(output_layer_activation)

    error = outputs - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)

    weights_input_output += inputs.T.dot(d_predicted_output) * lr
    bias_output += np.sum(d_predicted_output, axis=0, keepdims=True) * lr

print("Output:")
print(predicted_output)
print("\n----------------------------------")

print("\nW1:")
print(weights_input_output)

print("\nB1:")
print(bias_output)

x_min, x_max = 0, 1
y_min, y_max = 0, 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
xy = xx * yy
grid = np.c_[xx.ravel(), yy.ravel(), xy.ravel()]

output_layer_activation = np.dot(grid, weights_input_output) + bias_output
grid_predictions = sigmoid(output_layer_activation)
grid_predictions = grid_predictions.reshape(xx.shape)

plt.figure(figsize=(6, 6))
plt.contourf(xx, yy, grid_predictions, levels=[0, 0.5, 1], colors=['skyblue', 'darkgreen'], alpha=0.6)
plt.scatter(inputs[:, 0], inputs[:, 1], c=outputs[:, 0], edgecolor='k', marker='o', s=100)
plt.title('Output')
plt.xlabel('x')
plt.ylabel('y')
plt.show()