import timeit
import jax.numpy as jnp
from jax import jit
from jax import random

# Define your function
def compute_pairwise_distances_jax(high_dimensional_data):
    pairwise_distances = jnp.sum((high_dimensional_data[:, None, :] - high_dimensional_data) ** 2, axis=-1)
    return pairwise_distances

# Generate a random key for random number generation
key = random.PRNGKey(0)

# Generate some sample data
num_data_points = 1000
high_dimensional_data = random.normal(key, (num_data_points, 10))  # Adjust the dimensions as needed

# Time the function without jit
time_no_jit = timeit.timeit(lambda: compute_pairwise_distances_jax(high_dimensional_data), number=100)

# Decorate the function with jit
compute_pairwise_distances_jax_jit = jit(compute_pairwise_distances_jax)

# Time the function with jit
time_with_jit = timeit.timeit(lambda: compute_pairwise_distances_jax_jit(high_dimensional_data), number=100)

# Print the timings
print("Time without JIT: {:.5f} seconds".format(time_no_jit))
print("Time with JIT: {:.5f} seconds".format(time_with_jit))
