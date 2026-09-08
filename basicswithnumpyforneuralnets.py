import numpy as np

# ==========================================
# 1. Basic 1-D Array (Inputs & Weights)
# ==========================================
# In an MCP Neuron, inputs (X) and weights (W) are typically 1D arrays (vectors).
inputs = np.array([0.5, 0.2, 0.9])
weights = np.array([0.4, 0.8, -0.5])

print("Inputs vector:", inputs)
print("Weights vector:", weights)


# ==========================================
# 2. Indexing Elements of an Array
# ==========================================
# Indexing lets you access specific inputs or weights.
first_input = inputs[0]       # First element (0.5)
last_weight = weights[-1]     # Last element (-0.5)
weight_slice = weights[0:2]   # Slice first two elements ([0.4, 0.8])

print(f"\nFirst Input: {first_input}, Last Weight: {last_weight}")
print("First two weights:", weight_slice)


# ==========================================
# 3. initialising Basic Arrays
# ==========================================
# Perceptrons often initialise weights to zeros, ones, or small initial values.
zeros_arr = np.zeros(3)          # Array of zeros: [0., 0., 0.]
ones_arr = np.ones((2, 3))       # 2x3 matrix of ones
empty_arr = np.empty(3)          # Uninitialised array (contains garbage values from memory)

print("\nZeros Array (Weights initialisation):", zeros_arr)
print("Ones Array (2x3):\n", ones_arr)


# ==========================================
# 4. Shape, Size, and Dimensions
# ==========================================
# Crucial for verifying matrix dimensions before multiplication in neural networks.
matrix = np.array([[1, 2, 3], 
                   [4, 5, 6]])

print(f"\nDimensions (ndim): {matrix.ndim}")   # Returns 2 (2D array/matrix)
print(f"Total elements (size): {matrix.size}") # Returns 6
print(f"Shape (shape): {matrix.shape}")       # Returns (2, 3) -> 2 rows, 3 columns


# ==========================================
# 5. Array Reshaping
# ==========================================
# Transposing or reshaping vectors (e.g., converting a 1D vector to a column vector).
vector = np.array([1, 2, 3, 4])
row_vector = vector.reshape(1, 4)    # Shape: (1, 4)
column_vector = vector.reshape(4, 1) # Shape: (4, 1)

print("\nColumn Vector:\n", column_vector)


# ==========================================
# 6. Basic Array Operations & Dot Product
# ==========================================
# Perceptrons calculate weighted sums: z = sum(x_i * w_i) + b
x = np.array([1.0, 2.0, 3.0])
w = np.array([0.2, 0.8, -0.5])
bias = 0.5

# Element-wise multiplication
elementwise = x * w  # [0.2, 1.6, -1.5]

# Dot Product (Sum of element-wise products) - Central to MCP Neuron & Perceptron
weighted_sum = np.dot(x, w) + bias

print("\nElement-wise multiplication:", elementwise)
print("Weighted Sum (np.dot(x, w) + bias):", weighted_sum)


# ==========================================
# 7. Broadcasting
# ==========================================
# NumPy allows operations between arrays of different shapes by "stretching" the smaller one.
# Adding a scalar bias across a whole matrix of inputs:
batch_inputs = np.array([[1.0, 2.0], 
                         [3.0, 4.0]])
b = 0.5

# Bias scalar is broadcast across each row of the batch
batch_with_bias = batch_inputs + b

print("\nBatch Inputs with Broadcast Bias:\n", batch_with_bias)


# ==========================================
# 8. Working with Mathematical Formulas
# ==========================================
# Implementing the Perceptron Activation Function:
# Step Function: Output 1 if weighted sum >= 0, else 0
def step_function(z):
    return np.where(z >= 0, 1, 0)

# Example output from a batch of weighted sums
sample_z = np.array([-1.5, 0.0, 2.3, -0.1])
perceptron_output = step_function(sample_z)

print("\nSample Weighted Sums:", sample_z)
print("Perceptron Binary Outputs:", perceptron_output)