
import jax
from jax import jit
from jax.nn import softmax
from jax.experimental import host_callback
from jax.numpy.linalg import svd
from tqdm import trange
import jax.numpy as jnp
from jax import random


@jit
def pca_jax(X, k=30):
    # Center and scale the data
    s = jnp.std(X, axis=0)
    s = jnp.where(s == 0, 1, s)  # Avoid division by zero
    X_centered_scaled = (X - jnp.mean(X, axis=0)) / s

    # Compute SVD
    U, S, Vh = svd(X_centered_scaled, full_matrices=False)

    # Project data onto the first k principal components
    X_pca = X_centered_scaled @ Vh.T[:, :k]

    return X_pca

@jit
def compute_pairwise_distances(dim_data):
    # Efficient broadcasting for pairwise squared Euclidean distances
    sum_X = jnp.sum(jnp.square(dim_data), axis=1)
    D = sum_X[:, None] - 2 * jnp.dot(dim_data, dim_data.T) + sum_X[None, :]
    return D

@jit
def get_eculidean_probability_at_ij(d, scale, i):
    d_scaled = -d / scale
    d_scaled -= jnp.max(d_scaled)
    exp_D = jnp.exp(d_scaled)
    exp_D = exp_D.at[i].set(0)
    epsilon = 1e-8
    return exp_D / (jnp.sum(exp_D)+epsilon)

@jit
def get_ntk_probability_at_ij(d, sigma, i):
    # No negation for NTK values, as larger values indicate stronger influence.
    d_scaled = d / (sigma**2)
    d_scaled -= d_scaled.min()  # Min normalization
    d_scaled = d_scaled.at[i].set(-jnp.inf)
    probabilties = softmax(d_scaled/.5)

    return probabilties

@jit
def get_probability_at_ij(d, sigma, i, is_ntk):

    def true_fun(_):
        return get_ntk_probability_at_ij(d, sigma, i)

    def false_fun(_):
        return get_eculidean_probability_at_ij(d, sigma, i)

    return jax.lax.cond(is_ntk, true_fun, false_fun, None)


@jit
def get_shannon_entropy(p):
    """2 ** H(p) of array p, where H(p) is the Shannon entropy."""
    return 2 ** jnp.sum(-p * jnp.log2(p + 1e-10))

def print_attempts(value):
  """Prints the value on the host machine."""
  print(f"attempts done: {value}")

@jit
def all_sym_affinities(data_mat, perp, tol,attempts=250,  is_ntk=True):

    n_samples = data_mat.shape[0]

    # Correctly initialize P outside the loop
    P = jnp.zeros(data_mat.shape)

    def body_fun(i, P):
        sigma_max = 1e4
        sigma_min = 0.0
        d = data_mat[i, :]

        def cond_fun(val):
            sigma_min, sigma_max, _, attempts_counter = val
            # host_callback.call(print_attempts, attempts_counter)
            return (jnp.abs(sigma_max - sigma_min) > tol) & (attempts_counter < attempts)

        def body_fun(val):
            sigma_min, sigma_max, p_ij, attempts_counter = val
            sigma_mid = (sigma_min + sigma_max) / 2
            scale = 2 * sigma_mid ** 2
            p_ij = get_probability_at_ij(d, scale, i, is_ntk)
            current_perp = get_shannon_entropy(p_ij)

            update_cond = current_perp < perp
            sigma_min = jax.lax.cond(update_cond, lambda: sigma_mid, lambda: sigma_min)
            sigma_max = jax.lax.cond(update_cond, lambda: sigma_max, lambda: sigma_mid)
            return sigma_min, sigma_max, p_ij, attempts_counter + 1

        _, _, p_ij, attempts_counter = jax.lax.while_loop(cond_fun, body_fun, (sigma_min, sigma_max, jnp.zeros_like(d), 0))
        # host_callback.call(print_attempts, attempts_counter)
        # Update P correctly using the result from while_loop
        P = P.at[i, :].set(p_ij)
        return P

    # Use lax.fori_loop to iterate over samples and update P
    P = jax.lax.fori_loop(0, n_samples, body_fun, P)
    return (P + P.T) / (2 * n_samples)

@jit
def compute_grad(R, Y_dists, Y):

    # Expand dimensions to support broadcasting for vectorized subtraction
    Y_expanded = Y[:, None, :]  # Shape becomes (n, 1, num_dimensions)

    # Compute pairwise differences using broadcasting, result has shape (n, n, num_dimensions)
    pairwise_diff = Y_expanded - Y[None, :, :]

    # Element-wise multiplication of R and Y_dists, then further element-wise multiplication by pairwise differences
    # R and Y_dists are expanded to match the shape for broadcasting
    grad_contributions = 4 * R[:, :, None] * pairwise_diff * Y_dists[:, :, None]

    # Sum over the second axis (j index in the loop) to aggregate contributions to each point i
    dY = jnp.sum(grad_contributions, axis=1)

    return dY

@jit
def low_dim_affinities(Y):
    D = compute_pairwise_distances(Y)
    Y_dists = jnp.power(1 + D , -1)
    n = Y_dists.shape[0]
    Y_dists_no_diag = Y_dists.at[jnp.diag_indices(n)].set(0)
    # Ensure denominator is not too close to zero by adding a small constant epsilon
    epsilon = 1e-8
    normalized_Y_dists_no_diag = Y_dists_no_diag / (jnp.sum(Y_dists_no_diag) + epsilon)

    return normalized_Y_dists_no_diag, Y_dists


@jit
def momentum_func(t):
    return jax.lax.cond(t < 250, lambda _: 0.5, lambda _: 0.8, operand=None)

def initialize_embedding(P, num_dimensions, max_iterations, random_state):
    size = (P.shape[0], num_dimensions)

    key = random.PRNGKey(random_state)
    initial_vals = random.normal(key, shape=size) * jnp.sqrt(1e-4)

    embedding_matrix_container = jnp.zeros(shape=(max_iterations + 2, size[0], num_dimensions))
    # Set initial values for embedding_matrix_container at t=0 and t=1
    embedding_matrix_container = embedding_matrix_container.at[0, :, :].set(initial_vals)
    embedding_matrix_container = embedding_matrix_container.at[1, :, :].set(initial_vals)

    return embedding_matrix_container, initial_vals
def setup_device_for_jax():
    if any('gpu' in device.platform.lower() for device in jax.devices()):
        jax.config.update('jax_platform_name', 'gpu')
        print('Using GPU')
        return True
    return False

def put_data_on_gpu(data):
    return jax.device_put(data, jax.devices('gpu')[0])


def run_tsne_algorithm(high_dimensional_data, perplexity, perp_tol, scaling_factor,
                       num_dimensions, max_iterations,
                       learning_rate, random_state,
                       is_ntk = False, matrix_function = None):

    # Setup device for JAX computations and move data if GPU is used
    if setup_device_for_jax():
        high_dimensional_data = put_data_on_gpu(high_dimensional_data)
        print('Data is on GPU')

    if high_dimensional_data.shape[1] > 30:
        high_dimensional_data = pca_jax(high_dimensional_data)

    data_mat = matrix_function(high_dimensional_data) if is_ntk else compute_pairwise_distances(high_dimensional_data)

    # Compute pairwise affinities in high-dimensional space, scaled by a factor
    P = all_sym_affinities(data_mat, perplexity, perp_tol, attempts=75, is_ntk = is_ntk) * scaling_factor


    embedding_matrix_container, Y_m1 = initialize_embedding(P, num_dimensions, max_iterations, random_state)
    Y_m2 = Y_m1

    # Iterative optimization
    for i in trange(2, max_iterations + 2, disable=False):
        # Compute low-dimensional affinities and distances
        Q, Y_dists = low_dim_affinities(Y_m1)

        # Compute the gradient
        grad = compute_grad(P - Q, Y_dists, Y_m1)

        # Update embeddings
        Y_new = Y_m1 - learning_rate * grad + momentum_func(i) * (Y_m1 - Y_m2)

        # Update historical embeddings for momentum calculation
        Y_m2, Y_m1 = Y_m1, Y_new

        # Record the new embeddings
        embedding_matrix_container = embedding_matrix_container.at[i, :, :].set(Y_new)

    return embedding_matrix_container


