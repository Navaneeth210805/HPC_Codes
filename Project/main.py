import numpy as np

# Set NUM_FEATURES (must match the C code constant)
NUM_FEATURES = 1000  
he_std = np.sqrt(6.0 / NUM_FEATURES)

# Generate parameters with He initialization
attention_weights = np.random.uniform(-he_std, he_std, (NUM_FEATURES, NUM_FEATURES))
weights = np.random.uniform(-he_std, he_std, (NUM_FEATURES, NUM_FEATURES))
attention_bias = np.random.uniform(-he_std, he_std, NUM_FEATURES)

# Save parameters to text files
np.savetxt("attention_weights.txt", attention_weights, fmt="%.8f")
np.savetxt("weights.txt", weights, fmt="%.8f")
np.savetxt("attention_bias.txt", attention_bias, fmt="%.8f")

print("Parameter files written successfully!")
