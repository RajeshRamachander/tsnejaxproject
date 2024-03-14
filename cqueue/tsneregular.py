from jax import random
import jax
import jax.numpy as jnp
from jax import jit
from tqdm import tqdm
from jax.experimental import host_callback
from jax import devices


EPSILON = 1e-12


@jit
def compute_pairwise_distances(dim_data):
    """
    Compute pairwise distances between data points in an optimized manner.

    Args:
        high_dimensional_data (jnp.ndarray): High-dimensional input data.

    Returns:
        jnp.ndarray: Pairwise distances matrix.
    """
    # Efficient broadcasting for pairwise squared Euclidean distances
    sum_X = jnp.sum(jnp.square(dim_data), axis=1)
    D = sum_X[:, None] - 2 * jnp.dot(dim_data, dim_data.T) + sum_X
    return D

@jit
def calculate_row_wise_entropy(asym_affinities):
    """
    Row-wise Shannon entropy of pairwise affinity matrix P

    Parameters:
    asym_affinities: pairwise affinity matrix of shape (n_samples, n_samples)

    Returns:
    jnp.ndarray: Row-wise Shannon entropy of shape (n_samples,)
    """
    asym_affinities = jnp.clip(asym_affinities, EPSILON, None)
    return -jnp.sum(asym_affinities * jnp.log(asym_affinities), axis=1)

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
def compute_pairwise_affinities(dist_mat, sigmas):
    """
    Computes a pairwise affinity matrix based on the distance matrix and sigma values,
    applying a Gaussian kernel with per-point bandwidths determined by sigma values.

    Parameters:
    dist_mat (jax.numpy.DeviceArray): A JAX array representing the distance matrix.
    sigmas (jax.numpy.DeviceArray): A JAX array of sigma values for scaling, one per data point.

    Returns:
    jax.numpy.DeviceArray: The computed pairwise affinity matrix with diagonal elements set to zero.
    """
    # Compute the Gaussian kernel values
    numers = jnp.exp(-dist_mat / (2 * (sigmas ** 2)))

    # Compute the normalization factors, excluding diagonal elements
    denoms = jnp.sum(numers, axis=1) - jnp.diag(numers)
    denoms = denoms[:, None] + EPSILON  # Reshape and ensure non-zero denominator

    # Calculate the pairwise affinities
    P = numers / denoms

    # Set the diagonal elements to zero
    P = P.at[jnp.diag_indices_from(P)].set(0)

    return P

@jit
def all_sym_affinities(data, perp, tol,  attempts=100):
    dist_mat = compute_pairwise_distances(data)
    n_samples = data.shape[0]

    sigma_maxs = jnp.full(data.shape[0], 1e12)
    sigma_mins = jnp.full(data.shape[0], 1e-12)
    sigmas = (sigma_mins + sigma_maxs) / 2

    P = compute_pairwise_affinities(dist_mat, sigmas.reshape(-1, 1))

    current_perps = calculate_row_wise_perplexities(P)

    # Define the condition for continuing the binary search
    def condition(vals):
        _, attempts, _, _, _, _ = vals
        return attempts > 0

    # Define the body of the loop for binary search
    def body(vals):
        sigmas, attempts, sigma_maxs, sigma_mins, current_perps, P = vals
        P = compute_pairwise_affinities(dist_mat, sigmas.reshape(-1, 1))
        new_perps = calculate_row_wise_perplexities(P)

        # Update sigma bounds based on whether perplexity is too high or too low
        sigma_maxs = jnp.where(new_perps > perp, sigmas, sigma_maxs)
        sigma_mins = jnp.where(new_perps <= perp, sigmas, sigma_mins)
        sigmas = (sigma_mins + sigma_maxs) / 2.0

        return (sigmas, attempts - 1, sigma_maxs, sigma_mins, new_perps, P)

    # Execute the loop
    sigmas, _, sigma_maxs, sigma_mins, current_perps, P = jax.lax.while_loop(
        condition,
        body,
        (sigmas, attempts, sigma_maxs, sigma_mins, current_perps, P)
    )

    # Symmetrize the P matrix
    P = (P + P.T) / (2 * n_samples)

    return P


@jit
def low_dim_affinities(Y_dist_mat):
    """
    Optimized version of computing the low-dimensional affinities matrix Q.
    """
    # Directly compute the numerators and the normalization factor in one step
    numers = 1 / (1 + Y_dist_mat)

    # Avoid computing the diagonal sum by subtracting it after summing all elements
    sum_all = jnp.sum(numers)
    sum_diag = jnp.sum(jnp.diag(numers))
    denom = sum_all - sum_diag + EPSILON  # Adjust for division by zero

    # Compute Q without explicitly filling the diagonal with zeros
    Q = numers / denom

    # Ensure the diagonal is zero by subtracting its values divided by denom
    # This step is more efficient than setting the diagonal to zero explicitly
    Q -= jnp.diag(jnp.diag(numers) / denom)

    return Q

@jit
def compute_grad(P, Q, Y, Y_dist_mat):
    # Compute pairwise differences more directly
    Ydiff = Y[:, None, :] - Y[None, :, :]

    # Compute the pq_factor considering broadcasting, no need for explicit newaxis
    pq_factor = P - Q

    # Compute the dist_factor considering broadcasting
    dist_factor = 1 / (1 + Y_dist_mat)

    # Compute the gradient without explicitly expanding pq_factor and dist_factor
    grad = 4 * jnp.sum(pq_factor[:, :, None] * Ydiff * dist_factor[:, :, None], axis=1)

    return grad


@jit
def momentum_func(t):
    """Returns the optimization parameter.

    Parameters:
    t (int): The current iteration step.

    Returns:
    float: Represents the momentum term added to the gradient.
    """
    return jax.lax.cond(t < 250, lambda _: 0.5, lambda _: 0.8, operand=None)


def compute_low_dimensional_embedding_regular_tsne(high_dimensional_data, num_dimensions,
                                      perplexity, max_iterations=100,
                                      learning_rate=100, scaling_factor=4.,
                                      random_state=42,
                                      perp_tol=1e-8):

    all_devices = devices()
    if any('gpu' in dev.platform.lower() for dev in all_devices):
        jax.config.update('jax_platform_name', 'gpu')
        print('Using GPU')
        high_dimensional_data = jax.device_put(high_dimensional_data, jax.devices('gpu')[0])
        print('Data is on GPU')


    P = all_sym_affinities(jax.device_put(high_dimensional_data, jax.devices('gpu')[0]), perplexity, perp_tol) * scaling_factor
    P = jnp.clip(P, EPSILON, None)

    init_mean = jnp.zeros(num_dimensions, dtype=jnp.float32)
    init_cov = jnp.eye(num_dimensions, dtype=jnp.float32) * 1e-4

    # Ensure the random key is generated correctly
    rand = random.PRNGKey(random_state)
    Y = random.multivariate_normal(rand, mean=init_mean, cov=init_cov, shape=(high_dimensional_data.shape[0],))

    Y_old = jnp.zeros_like(Y)

    iter_range = range(max_iterations)

    iter_range = tqdm(iter_range, "Iterations")
    for t in iter_range:
        Y_dist_mat = compute_pairwise_distances(Y)
        Q = low_dim_affinities(Y_dist_mat)
        Q = jnp.clip(Q, EPSILON, None)
        grad = compute_grad(P, Q, Y, Y_dist_mat)
        Y = Y - learning_rate * grad + momentum_func(t) * (Y - Y_old)
        Y_old = Y.copy()
        if t == 100:
            P = P / scaling_factor
            pass
        pass

    return Y

