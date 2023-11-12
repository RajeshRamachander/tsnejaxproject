from jax import random
import jax
import jax.numpy as jnp
from jax import jit
from tqdm import tqdm

EPSILON = 1e-12

import jax
import jax.numpy as jnp

@jax.jit
def compute_pairwise_distances(high_dimensional_data):
    """
    Compute pairwise distances between data points.

    Args:
        high_dimensional_data (jnp.ndarray): High-dimensional input data.

    Returns:
        jnp.ndarray: Pairwise distances matrix.
    """
    X_squared = jnp.square(high_dimensional_data)
    sum_X = jnp.sum(X_squared, axis=1, keepdims=True)

    # Compute pairwise squared Euclidean distances efficiently
    pairwise_distances = jnp.dot(high_dimensional_data, high_dimensional_data.T)
    pairwise_distances *= -2.0
    pairwise_distances += sum_X
    pairwise_distances += sum_X.T

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
    asym_affinities = jnp.clip(asym_affinities, EPSILON, None)
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
def fill_diagonal(arr, value):
    return arr.at[jnp.diag_indices(arr.shape[0])].set(value)
@jit
def gaussian_kernel(x_i, x_j, sigma=1.0):
    # Gaussian kernel function
    squared_distance = jnp.sum((x_i - x_j) ** 2)
    return jnp.exp(-squared_distance / (2 * sigma**2))

@jit
def compute_ntk(data):
    # Compute the NTK-like matrix using the specified kernel function
    n_samples = data.shape[0]
    ntk_matrix = jnp.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(n_samples):
            ntk_matrix = ntk_matrix.at[i, j].set(gaussian_kernel(data[i], data[j]))

    return ntk_matrix

@jit
def rbf_kernel(dist_mat, sigmas):
    """
    Compute the Gaussian Radial Basis Function (RBF) kernel.

    Parameters:
    - dist_mat (array): A square matrix of distances.
    - sigmas (array): A vector of standard deviations for the Gaussian kernel.

    Returns:
    - kernel_mat (array): The RBF kernel matrix.
    """
    # Compute the RBF kernel
    kernel_mat = jnp.exp(-dist_mat / (2 * (sigmas ** 2)))
    return kernel_mat


@jit
def compute_normalized_matrix(dist_mat, sigmas):
    """
    Compute a normalized matrix based on the distance matrix and scaling parameters.

    Parameters:
    - dist_mat (array): A square matrix of distances.
    - sigmas (array): Scaling parameters for each row of the distance matrix.
    - EPSILON (float): A small constant to avoid division by zero.

    Returns:
    - P (array): The normalized matrix.
    """
    # Compute the RBF kernel
    kernel_mat = rbf_kernel(dist_mat, sigmas)

    # Compute the numerators and denominators for normalization
    numers = kernel_mat
    denoms = jnp.sum(numers, axis=1) - jnp.diag(numers)
    denoms = denoms[:, None] + EPSILON  # Avoid division by zero

    # Compute the normalized matrix
    P = numers / denoms
    return P


@jit
def pairwise_affinities(data, sigmas, dist_mat):
    """Calculates the pairwise affinities using NTK-like principles

    Parameters:
    data : jnp.ndarray of shape (n_samples, n_features)

    Returns:
    jnp.ndarray: Pairwise affinity matrix of size (n_samples, n_samples)

    """

    P = compute_normalized_matrix(dist_mat, sigmas)
    P = fill_diagonal(P,0)
    return P



@jit
def all_sym_affinities(data, perp, tol, attempts=100):
    dist_mat = compute_pairwise_distances(data)
    sigma_maxs = jnp.full(data.shape[0], 1e12)
    sigma_mins = jnp.full(data.shape[0], 1e-12)
    current_perps = jnp.full(data.shape[0], jnp.inf)
    sigmas = (sigma_mins + sigma_maxs) / 2
    P = pairwise_affinities(data, sigmas.reshape(-1, 1), dist_mat)

    def outer_while_condition(args):
        current_perps, perp, tol, attempts, sigma_maxs, sigma_mins, P = args
        return jnp.logical_and(
            jnp.logical_not(jnp.allclose(current_perps, perp, atol=tol)),
            attempts > 0
        )

    def outer_while_body(args):
        current_perps, perp, tol, attempts, sigma_maxs, sigma_mins, P = args
        sigmas = (sigma_mins + sigma_maxs) / 2
        P = pairwise_affinities(data, sigmas.reshape(-1,1), dist_mat)
        current_perps = calculate_row_wise_perplexities(P)
        attempts -= 1

        def inner_while_condition(args):
            i, sigmas, sigma_maxs, sigma_mins, current_perps, perp = args
            return i < len(current_perps)

        def inner_while_body(args):
            i, sigmas, sigma_maxs, sigma_mins, current_perps, perp = args
            current_perp = current_perps[i]
            sigma_maxs = sigma_maxs.at[i].set(jax.numpy.where(current_perp > perp, sigmas[i], sigma_maxs[i]))
            sigma_mins = sigma_mins.at[i].set(jax.numpy.where(current_perp <= perp, sigmas[i], sigma_mins[i]))
            return i + 1, sigmas, sigma_maxs, sigma_mins, current_perps, perp

        inner_while_args = (0, sigmas, sigma_maxs, sigma_mins, current_perps, perp)
        (_, _, sigma_maxs, sigma_mins, _, _ ) = jax.lax.while_loop(inner_while_condition, inner_while_body, inner_while_args)

        return current_perps, perp, tol, attempts, sigma_maxs, sigma_mins, P

    outer_while_args = (current_perps, perp, tol, attempts, sigma_maxs, sigma_mins, P)
    (_, _, _, _, _, _, P) = jax.lax.while_loop(outer_while_condition, outer_while_body, outer_while_args)

    P = (P + P.T) / (2 * data.shape[0])

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
    Q = fill_diagonal(Q,0)
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


def compute_low_dimensional_embedding(high_dimensional_data, num_dimensions,
                                      target_perplexity, max_iterations=100,
                                      learning_rate=100, scaling_factor=4.,
                                      pbar=False, random_state=None,
                                      perp_tol=1e-8):

    # Ensure the random key is generated correctly
    if random_state is None:
        rand = random.PRNGKey(42)
    else:
        rand = random_state

    P = all_sym_affinities(high_dimensional_data, target_perplexity, perp_tol) * scaling_factor
    P = jnp.clip(P, EPSILON, None)

    init_mean = jnp.zeros(num_dimensions, dtype=jnp.float32)
    init_cov = jnp.eye(num_dimensions, dtype=jnp.float32) * 1e-4

    # Ensure the random key is generated correctly
    if random_state is None:
        rand = random.PRNGKey(42)
    else:
        rand = random_state
    Y = random.multivariate_normal(rand, mean=init_mean, cov=init_cov, shape=(high_dimensional_data.shape[0],))

    Y_old = jnp.zeros_like(Y)

    iter_range = range(max_iterations)
    if pbar:
        iter_range = tqdm(iter_range, "Iterations")
    for t in iter_range:
        Y_dist_mat = compute_pairwise_distances(Y)
        Q = low_dim_affinities(Y, Y_dist_mat)
        Q = jnp.clip(Q, EPSILON, None)
        grad = compute_grad(P, Q, Y, Y_dist_mat)
        Y = Y - learning_rate * grad + momentum_func(t) * (Y - Y_old)
        Y_old = Y.copy()
        if t == 100:
            P = P / scaling_factor
            pass
        pass


    return Y
#
# import matplotlib.pyplot as plt
# from matplotlib import rcParams
# from sklearn.datasets import load_digits
# import numpy as np
#
# plt.style.use("seaborn-whitegrid")
# rcParams["font.size"] = 18
# rcParams["figure.figsize"] = (12, 8)
#
#
# digits, digit_class = load_digits(return_X_y=True)
# rand_idx = np.random.choice(np.arange(digits.shape[0]), size=500, replace=False)
# data = digits[rand_idx, :].copy()
# classes = digit_class[rand_idx]
#
# low_dim = compute_low_dimensional_embedding(data, 2, 30, 500, 100, pbar=True)
#
# scatter = plt.scatter(low_dim[:, 0], low_dim[:, 1], cmap="tab10", c=classes)
# plt.legend(*scatter.legend_elements(), fancybox=True, bbox_to_anchor=(1.05, 1))
# plt.show()