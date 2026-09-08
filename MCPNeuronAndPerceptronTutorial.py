import numpy as np

# Declare Neurons
binary_neuron_inputs = [
    np.array([0, 0]),
    np.array([0, 1]),
    np.array([1, 0]),
    np.array([1, 1])
]

# Declare Weights for AND/OR functions
weights_boolean = np.array([1, 1])  # Weights for AND/OR functions

# Logic for AND/OR functions using MCP Neuron
theta_or = 1
theta_and = 2

# Print the results of the MCP Neuron for OR and AND gates
print("\n--- MCP OR Gate (w=[1, 1], theta=1) ---")
for x in binary_neuron_inputs:
    neuron_output = np.dot(x, weights_boolean)
    print(f"OR {x}: {neuron_output >= theta_or}")

print("\n--- MCP AND Gate (w=[1, 1], theta=2) ---")
for x in binary_neuron_inputs:
    neuron_output = np.dot(x, weights_boolean)
    print(f"AND {x}: {neuron_output >= theta_and}")

# Define objects as 3-dimensional vectors with their features
all_objects = [
    np.array([0, 0, 0]),  # Hot Dog
    np.array([1, 0, 0]),  # Violet
    np.array([0, 1, 0]),  # Round item
    np.array([0, 0, 1]),  # Bouncy item
    np.array([0, 1, 1]),  # Golf Ball
    np.array([1, 1, 0]),  # Blueberry (TARGET)
    np.array([1, 1, 1])   # Purple, Round & Bouncy item
]

# Define weights and threshold for the "Eat" decision using an MCP Neuron
weights_eat = np.array([1, 1, -1])  # Weights for "Eat" decision
theta_eat = 2

print("\n--- Applied 3-Input MCP: Should I Eat This? ---")
for x in all_objects:
    score = np.dot(x, weights_eat)
    decision = int(score >= theta_eat)
    print(f"Features [Purple, Round, Bouncy]: {x} | Score: {score:2d} | Eat?: {decision}")



# Perceptron Model (Continuous Values, Bias, Activation)
print("\n--- Perceptron Model Operations ---")

# Step 1: Base model setup
x_real = np.array([2.3, 1.0, 0.6])    # Continuous inputs in R
w_real = np.array([-0.4, 0.5, 0.3])   # Continuous weights in R
bias = 0.5                            # Initial bias term
theta_perceptron = 1.0                # Threshold

raw_output = np.dot(x_real, w_real) + bias  # (-0.92 + 0.5 + 0.18) + 0.5 = 0.26
activated = int(raw_output >= theta_perceptron)
print(f"Base Output: {raw_output:.2f} | Fire: {activated}")

# Increase bias to force neuron activation
bias_high = 1.3
raw_output_high_bias = np.dot(x_real, w_real) + bias_high
activated_high_bias = int(raw_output_high_bias >= theta_perceptron)
print(f"High Bias (b={bias_high}) Output: {raw_output_high_bias:.2f} | Fire: {activated_high_bias}")

# Increase inhibitory input (x[0]) to suppress activation back to 0
x_suppressed = np.array([3.2, 1.0, 0.6])
raw_output_suppressed = np.dot(x_suppressed, w_real) + bias_high
activated_suppressed = int(raw_output_suppressed >= theta_perceptron)
print(f"Suppressed Input (x0={x_suppressed[0]}) Output: {raw_output_suppressed:.2f} | Fire: {activated_suppressed}")    