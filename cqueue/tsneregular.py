from jax import random
import jax
import jax.numpy as jnp
from jax import jit
from tqdm import trange
from jax import devices

from tsne_common import (
    pca_jax,
    compute_pairwise_distances,
    run_tsne_algorithm,
)

def constant_learning_rate(t, eta_init, last_eta, c = 100):
    """Constant learning rate."""
    return c

def step_based(t, eta_init, last_eta, d = 0.01, r = 50):
    """Step-based learning rate with decay d and rate r."""
    return eta_init*d**jnp.floor((1+t)/r)

def compute_low_dimensional_embedding_regular_tsne(high_dimensional_data, num_dimensions,
                                      perplexity, max_iterations=100,
                                      learning_rate=100, scaling_factor=4.,
                                      random_state=42,
                                      perp_tol=.1):

    all_devices = devices()
    if any('gpu' in dev.platform.lower() for dev in all_devices):
        jax.config.update('jax_platform_name', 'gpu')
        print('Using GPU')
        high_dimensional_data = jax.device_put(high_dimensional_data, jax.devices('gpu')[0])
        print('Data is on GPU')

    if high_dimensional_data.shape[1] > 30:
        high_dimensional_data = pca_jax(high_dimensional_data)

    data_mat = compute_pairwise_distances(high_dimensional_data)

    Y = run_tsne_algorithm(data_mat, perplexity, perp_tol, scaling_factor,
                           num_dimensions, max_iterations,
                           learning_rate, random_state, is_ntk=False)

    print(Y.shape)
    return Y[-1]

