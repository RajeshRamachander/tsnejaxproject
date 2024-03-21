from jax import random
import jax
import jax.numpy as jnp
from jax import jit
from tqdm import trange
from jax import devices

from tsne_common import (
    all_sym_affinities,
    pca_jax,
    compute_pairwise_distances,
    low_dim_affinities,
    compute_grad,
    momentum_func
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

    data_mat = compute_pairwise_distances(jax.device_put(high_dimensional_data, jax.devices('gpu')[0]))

    P = all_sym_affinities(data_mat, perplexity, perp_tol,
                           attempts=75) * scaling_factor

    size = (P.shape[0], num_dimensions)
    Y = jnp.zeros(shape=(max_iterations + 2, size[0], num_dimensions))
    key = random.PRNGKey(random_state)
    initial_vals = random.normal(key, shape=size) * jnp.sqrt(1e-4)

    Y = Y.at[0, :, :].set(initial_vals)
    Y = Y.at[1, :, :].set(initial_vals)
    Y_m1 = initial_vals
    Y_m2 = initial_vals

    last_learning_rate = learning_rate

    for i in trange(2, max_iterations + 2, disable=False):


        Q, Y_dists = low_dim_affinities(Y_m1)

        grad = compute_grad(P - Q, Y_dists, Y_m1)

        learning_rate = step_based(i, learning_rate, last_learning_rate)
        last_learning_rate = learning_rate

        # Update embeddings.
        Y_new = Y_m1 - learning_rate * grad + momentum_func(i) * (Y_m1 - Y_m2)

        Y_m2, Y_m1 = Y_m1, Y_new
        Y = Y.at[i, :, :].set(Y_new)

    print(Y.shape)
    return Y[-1]


