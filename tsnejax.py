from jax import random
import jax
import jax.numpy as jnp
from jax import jit

EPSILON = 1e-12

@jit
def compute_pairwise_distances(high_dimensional_data):
    """
    Compute pairwise distances between data points.

    Args:
        high_dimensional_data (jnp.ndarray): High-dimensional input data.

    Returns:
        jnp.ndarray: Pairwise distances matrix.
    """
    # Compute pairwise squared Euclidean distances using JAX
    X_squared = jnp.square(high_dimensional_data)
    sum_X = jnp.sum(X_squared, axis=1)
    pairwise_distances = -2 * jnp.dot(high_dimensional_data, high_dimensional_data.T) + sum_X[:, jnp.newaxis] + sum_X

    return pairwise_distances

@jit
def calculate_row_wise_entropy(asym_affinities):
    """
    Row-wise Shannon entropy of pairwise affinity matrix P

    Parameters:
    asym_affinities: pairwise affinity matrix of shape (n_samples, n_samples)

    Returns:
    jnp.ndarray: Row-wise Shannon entropy of shape (n_samples,)
    """
    asym_affinities = jnp.clip(
        asym_affinities, EPSILON, None
    )  # Some are so small that log2 fails.
    return -jnp.sum(asym_affinities * jnp.log2(asym_affinities), axis=1)

@jit
def calculate_row_wise_perplexities(asym_affinities):
    """
    Compute perplexities of pairwise affinity matrix P

    Parameters:
    asym_affinities: pairwise affinity matrix of shape (n_samples, n_samples)

    Returns:
    jnp.ndarray: Row-wise perplexities of shape (n_samples,)
    """
    return 2 ** calculate_row_wise_entropy(asym_affinities)

@jit
def pairwise_affinities(data, sigmas, dist_mat):
    """Calculates the pairwise affinities p_{j|i} using the given values of sigma

    Parameters:
    data : jnp.ndarray of shape (n_samples, n_features)
    sigmas : column array of shape (n_samples, 1)
    dist_mat : data distance matrix; jnp.ndarray of shape (n_samples, n_samples)

    Returns:
    jnp.ndarray: Pairwise affinity matrix of size (n_samples, n_samples)

    """
    assert sigmas.shape == (data.shape[0], 1)
    inner = (-dist_mat) / (2 * (sigmas ** 2))
    numers = jnp.exp(inner)
    denoms = jnp.sum(numers, axis=1) - jnp.diag(numers)
    denoms = denoms[:, None]
    denoms += EPSILON  # Avoid div/0
    P = numers / denoms
    P = fill_diagonal(P,0)
    return P

@jit
def fill_diagonal(arr, value):
    return arr.at[jnp.diag_indices(arr.shape[0])].set(value)

@jit
def all_sym_affinities(data, perp, tol, attempts=100):
    dist_mat = compute_pairwise_distances(data)
    sigma_maxs = jnp.full(data.shape[0], 1e12)
    sigma_mins = jnp.full(data.shape[0], 1e-12)
    current_perps = jnp.full(data.shape[0], jnp.inf)

    def condition(args):
        current_perps, perp, tol, attempts, sigma_maxs, sigma_mins = args
        return jnp.logical_and(
            jnp.logical_not(jnp.allclose(current_perps, perp, atol=tol)),
            attempts > 0
        )

    def body(args):
        current_perps, perp, tol, attempts, sigma_maxs, sigma_mins = args
        sigmas = (sigma_mins + sigma_maxs) / 2
        P = pairwise_affinities(data, sigmas[:, None], dist_mat)
        current_perps = calculate_row_wise_perplexities(P)
        attempts -= 1

        # Calculate new sigma_maxs and sigma_mins
        new_sigma_maxs = jnp.where(current_perps > perp, sigmas, sigma_maxs)
        new_sigma_mins = jnp.where(current_perps < perp, sigmas, sigma_mins)

        return (current_perps, perp, tol, attempts, new_sigma_maxs, new_sigma_mins)

    initial_args = (current_perps, perp, tol, attempts, sigma_maxs, sigma_mins)
    (_, _, _, _, final_sigma_maxs, final_sigma_mins) = jax.lax.while_loop(condition, body, initial_args)
    P = pairwise_affinities(data, final_sigma_maxs[:, None], dist_mat)
    P = (P + P.T) / (2 * data.shape[0])
    P = fill_diagonal(P, 0.0)  # Fill diagonal with zeros
    return P

@jit
def low_dim_affinities(Y, Y_dist_mat):
    """
    Computes the low-dimensional affinities matrix Q.

    Parameters:
    Y : Low-dimensional representation of the data, ndarray of shape (n_samples, n_components)
    Y_dist_mat : Y distance matrix; ndarray of shape (n_samples, n_samples)

    Returns:
    Q : Symmetric low-dimensional affinities matrix of shape (n_samples, n_samples)
    """
    numers = (1 + Y_dist_mat) ** (-1)
    denom = jnp.sum(numers) - jnp.sum(jnp.diag(numers))
    denom += EPSILON  # Avoid div/0
    Q = numers / denom
    Q = jnp.where(jnp.eye(Q.shape[0], dtype=bool), 0.0, Q)
    return Q

@jit
def compute_grad(P, Q, Y, Y_dist_mat):
    """
    Computes the gradient vector needed to update the Y values.

    Parameters:
    P : Symmetric affinities matrix of shape (n_samples, n_samples)
    Q : Symmetric low-dimensional affinities matrix of shape (n_samples, n_samples)
    Y : Low-dimensional representation of the data, ndarray of shape (n_samples, n_components)
    Y_dist_mat : Y distance matrix; ndarray of shape (n_samples, n_samples)

    Returns:
    grad : The gradient vector, shape (n_samples, n_components)
    """
    Ydiff = Y[:, jnp.newaxis, :] - Y[jnp.newaxis, :, :]
    pq_factor = (P - Q)[:, :, jnp.newaxis]
    dist_factor = ((1 + Y_dist_mat) ** (-1))[:, :, jnp.newaxis]
    return jnp.sum(4 * pq_factor * Ydiff * dist_factor, axis=1)

@jit
def momentum_func(t):
    """Returns the optimization parameter.

    Parameters:
    t (int): The current iteration step.

    Returns:
    float: Represents the momentum term added to the gradient.
    """
    return jnp.where(t < 250, 0.5, 0.8)



@jit
def compute_pairwise_distances(high_dimensional_data):
    """
    Compute pairwise distances between data points.

    Args:
        high_dimensional_data (jnp.ndarray): High-dimensional input data (JAX array).

    Returns:
        jnp.ndarray: Pairwise distances matrix (JAX array).
    """
    # Compute pairwise squared Euclidean distances using JAX functions
    X_squared = jnp.square(high_dimensional_data)
    sum_X = jnp.sum(X_squared, axis=1)
    pairwise_distances = -2 * jnp.dot(high_dimensional_data, high_dimensional_data.T) + sum_X[:, jnp.newaxis] + sum_X
    return pairwise_distances

def initialize_embeddings(high_dimensional_data, num_dimensions, rand_key):
    init_mean = jnp.zeros(num_dimensions, dtype=jnp.float32)
    init_cov = jnp.eye(num_dimensions, dtype=jnp.float32) * 1e-4
    embeddings = random.multivariate_normal(rand_key, mean=init_mean, cov=init_cov, shape=(high_dimensional_data.shape[0],))
    return embeddings

# Create a closure to encapsulate the non-hashable random key
def initialize_embeddings_with_key(high_dimensional_data, num_dimensions, rand_key):
    return initialize_embeddings(high_dimensional_data, num_dimensions, rand_key)


def compute_low_dimensional_embedding(high_dimensional_data, num_dimensions,
                                      target_perplexity, max_iterations=100,
                                      learning_rate=100, scaling_factor=4.,
                                      pbar=False, random_state=None,
                                      perp_tol=1e-8):
    def body(args):
        t, Y = args
        Y_dist_mat = compute_pairwise_distances(Y)
        Q = low_dim_affinities(Y, Y_dist_mat)
        Q = jnp.clip(Q, EPSILON, None)
        grad = compute_grad(P, Q, Y, Y_dist_mat)
        Y = Y - learning_rate * grad + momentum_func(t) * (Y - Y_old)
        return t + 1, Y

    # Ensure the random key is generated correctly
    if random_state is None:
        rand = random.PRNGKey(42)
    else:
        rand = random_state

    P = all_sym_affinities(high_dimensional_data, target_perplexity, perp_tol) * scaling_factor
    P = jnp.clip(P, EPSILON, None)

    # init_mean = jnp.zeros(num_dimensions, dtype=jnp.float32)
    # init_cov = jnp.eye(num_dimensions, dtype=jnp.float32) * 1e-4
    #
    # Y = random.multivariate_normal(rand, mean=init_mean, cov=init_cov, shape=(high_dimensional_data.shape[0],))

    initialize_embeddings_jit = jax.jit(initialize_embeddings, static_argnums=(1,))
    # Call the JIT-compiled function with appropriate arguments

    Y = initialize_embeddings_jit(high_dimensional_data, num_dimensions, rand)

    Y_old = jnp.zeros_like(Y)

    # Define the condition function for the while loop
    def condition(args):
        t, _ = args
        return t < max_iterations

    _, Y = jax.lax.while_loop(condition, body, (0, Y))

    return Y
