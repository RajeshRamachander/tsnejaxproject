import jax
import jax.numpy as jnp
from jax import device_put, jit
import time

# Define the size of the matrix
size = 3000

# Function to generate a random matrix
def generate_random_matrix(size, key):
    return jax.random.uniform(key, (size, size), dtype=jnp.float32)

# JIT-compile the matrix multiplication function
@jit
def matmul(x, y):
    return jnp.dot(x, y)

# Create a random key
key = jax.random.PRNGKey(0)
# Split the key for generating two different matrices
key, subkey = jax.random.split(key)

# Generate two large random matrices
x = generate_random_matrix(size, key)
y = generate_random_matrix(size, subkey)

# Ensure matrices are on the GPU (if available)
x_gpu = device_put(x, jax.devices('gpu')[0] if jax.devices('gpu') else jax.devices('cpu')[0])
y_gpu = device_put(y, jax.devices('gpu')[0] if jax.devices('gpu') else jax.devices('cpu')[0])

# Perform matrix multiplication on CPU and time it
start_time = time.time()
cpu_result = matmul(x, y).block_until_ready()  # This will use the CPU or GPU depending on your JAX default device
cpu_time = time.time() - start_time
print(f"Execution time on default device: {cpu_time} seconds")

# Perform matrix multiplication on explicitly chosen device and time it
start_time = time.time()
gpu_result = matmul(x_gpu, y_gpu).block_until_ready()  # This will use the GPU if available
gpu_time = time.time() - start_time
print(f"Execution time on GPU (if available): {gpu_time} seconds")
