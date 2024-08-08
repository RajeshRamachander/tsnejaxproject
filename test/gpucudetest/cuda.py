import jax
import jax.numpy as jnp

# Check available devices
print("Available devices:", jax.devices())

# Create a random tensor and move it to the GPU
x = jnp.ones((1000, 1000))
x_gpu = jax.device_put(x)

# Print the device information correctly
print("Tensor on GPU:", x_gpu.device)
