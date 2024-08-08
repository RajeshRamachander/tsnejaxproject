import jax
import jax.numpy as jnp
import time

# pip install --upgrade pip

# # Installs the wheel compatible with CUDA 12 and cuDNN 8.9 or newer.
# # Note: wheels only available on linux.
# pip install --upgrade "jax[cuda12_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# # Installs the wheel compatible with CUDA 11 and cuDNN 8.6 or newer.
# # Note: wheels only available on linux.
# pip install --upgrade "jax[cuda11_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

def complex_computation(x):
    y = jnp.sin(x) ** 2 + jnp.cos(x) ** 2
    z = jnp.exp(-x)
    w = jnp.sum(jnp.log(x + 1))
    return y + z + w

def run_on_cpu():
    x = jnp.arange(1e6)
    start_time = time.time()
    result = complex_computation(x)
    end_time = time.time()
    return result, end_time - start_time

def run_on_gpu():
    x = jnp.arange(1e6)
    x_gpu = jax.device_put(x, jax.devices('gpu')[0])
    start_time = time.time()
    result = complex_computation(x_gpu)
    end_time = time.time()
    return result, end_time - start_time

if __name__ == "__main__":
    from jax import devices


    def detect_jax_device():
        all_devices = devices()
        if any('gpu' in dev.platform.lower() for dev in all_devices):
            gpu_result, gpu_time = run_on_gpu()
            print("GPU result:", gpu_result)
            print("GPU execution time:", gpu_time)
            return 'GPU'
        else:
            cpu_result, cpu_time = run_on_cpu()
            print("CPU result:", cpu_result)
            print("CPU execution time:", cpu_time)
            return 'CPU'


    device_type = detect_jax_device()
    print("JAX is running on:", device_type)


import jax
import jax.numpy as jnp
from jax.lib import xla_bridge

print("JAX is using platform:", xla_bridge.get_backend().platform)

key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (1000, 1000))
y = jnp.dot(x, x.T)
print(y)
