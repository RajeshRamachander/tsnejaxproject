from jax import random
import jax
import jax.numpy as jnp
from jax import jit
from tqdm import tqdm
from neural_tangents import stax
from jax.experimental import host_callback
from jax import devices


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
def compute_probabilities_from_ntk(data):
    """
    Computes a probability matrix from input data using a scaled Neural Tangent Kernel (NTK).

    Parameters:
    data (jax.numpy.DeviceArray): A JAX array of input data points.
    dist_mat (jax.numpy.DeviceArray): A JAX array representing the distance matrix.
    sigmas (jax.numpy.DeviceArray): A JAX array of sigma values for scaling.
    layer_size (int): The size of each hidden layer in the neural network.

    Returns:
    jax.numpy.DeviceArray: The computed probability matrix.
    """

    # Define a function to compute the NTK matrix
    def compute_ntk_matrix(inputs):

        # # Reshape the data for the convolutional network
        X = inputs.reshape(-1, 8, 8, 1)  # Reshape to [batch_size, height, width, channels]
        X = X / 16.0  # Normalize pixel values

        # Define your neural network architecture
        # init_fn, apply_fn, kernel_fn = stax.serial(
        #     stax.Conv(32, (3, 3), padding="SAME"), stax.Relu(),
        #     stax.AvgPool((2, 2)),  # Average pooling can be used to reduce dimensionality
        #     stax.Flatten(),
        #     stax.Dense(100), stax.Relu(),
        #     stax.Dense(10)
        # )

        # Define your neural network architecture
        init_fn, apply_fn, kernel_fn = stax.serial(
            stax.Dense(2048), stax.Relu(),
            stax.Dense(2048), stax.Relu(),
            stax.Flatten(), stax.Dense(1)
        )

        # Compute the neural tangent kernel
        return kernel_fn(X, X, 'ntk')


    # Compute the NTK matrix
    ntk_matrix = compute_ntk_matrix(data)
    # Apply softmax for normalization
    P = jnp.exp(ntk_matrix) / jnp.sum(jnp.exp(ntk_matrix), axis=1, keepdims=True)

    # Set the diagonal to zero (optional)
    P = P.at[jnp.diag_indices_from(P)].set(0)

    return P

@jit
def compute_pairwise_probabilities(dist_mat, sigmas):
    """
    Computes a pairwise probability matrix based on the distance matrix and sigma values.

    Parameters:
    dist_mat (jax.numpy.DeviceArray): A JAX array representing the distance matrix.
    sigmas (jax.numpy.DeviceArray): A JAX array of sigma values for scaling.
    epsilon (float): A small value to avoid division by zero.

    Returns:
    jax.numpy.DeviceArray: The computed probability matrix.
    """

    # Compute numerators
    inner = (-dist_mat) / (2 * (sigmas ** 2))
    numers = jnp.exp(inner)

    # Compute denominators
    denoms = jnp.sum(numers, axis=1) - jnp.diag(numers)
    denoms = denoms[:, None] + EPSILON  # Reshape and avoid division by zero

    # Calculate probabilities
    P = numers / denoms

    # Set the diagonal to zero
    P = P.at[jnp.diag_indices_from(P)].set(0)
    return P

@jit
def pairwise_affinities(data, sigmas, dist_mat, use_ntk):
    """Calculates the pairwise affinities p_{j|i} using the given values of sigma

    Parameters:
    data : jnp.ndarray of shape (n_samples, n_features)
    sigmas : column array of shape (n_samples, 1)
    dist_mat : data distance matrix; jnp.ndarray of shape (n_samples, n_samples)

    Returns:
    jnp.ndarray: Pairwise affinity matrix of size (n_samples, n_samples)

    """
    assert sigmas.shape == (data.shape[0], 1)
    assert data.shape[0] == sigmas.shape[0] == dist_mat.shape[0], "Inconsistent dimensions"
    assert sigmas.shape[1] == 1, "Sigmas must be a column array"

    def true_fun(_):
        # host_callback.id_print(use_ntk, what="use_ntk")
        return compute_probabilities_from_ntk(data)

    def false_fun(_):
        # host_callback.id_print(use_ntk, what="use_ntk")
        return compute_pairwise_probabilities(dist_mat, sigmas)

    return jax.lax.cond(use_ntk, true_fun, false_fun, None)

@jit
def all_sym_affinities(data, perp, tol,  use_ntk, attempts=100):
    dist_mat = compute_pairwise_distances(data)
    sigma_maxs = jnp.full(data.shape[0], 1e12)
    sigma_mins = jnp.full(data.shape[0], 1e-12)
    current_perps = jnp.full(data.shape[0], jnp.inf)
    sigmas = (sigma_mins + sigma_maxs) / 2
    P = pairwise_affinities(data, sigmas.reshape(-1, 1), dist_mat, use_ntk)

    def outer_while_condition(args):
        current_perps, perp, tol, attempts, sigma_maxs, sigma_mins, P, use_ntk = args
        return jnp.logical_and(
            jnp.logical_not(jnp.allclose(current_perps, perp, atol=tol)),
            attempts > 0
        )

    def outer_while_body(args):
        current_perps, perp, tol, attempts, sigma_maxs, sigma_mins, P, use_ntk = args
        sigmas = (sigma_mins + sigma_maxs) / 2
        P = pairwise_affinities(data, sigmas.reshape(-1,1), dist_mat, use_ntk)
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

        return current_perps, perp, tol, attempts, sigma_maxs, sigma_mins, P, use_ntk

    outer_while_args = (current_perps, perp, tol, attempts, sigma_maxs, sigma_mins, P, use_ntk)
    (_, _, _, _, _, _, P, use_ntk) = jax.lax.while_loop(outer_while_condition, outer_while_body, outer_while_args)

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
                                      pbar=False, random_state=42,
                                      perp_tol=1e-8, use_ntk = True):

    all_devices = devices()
    if any('gpu' in dev.platform.lower() for dev in all_devices):
        jax.config.update('jax_platform_name', 'gpu')
        print('Using GPU')
        high_dimensional_data = jax.device_put(high_dimensional_data, jax.devices('gpu')[0])
        print('Data is on GPU')

    # Ensure the random key is generated correctly
    if random_state is None:
        rand = random.PRNGKey(42)
    else:
        rand = random_state

    P = all_sym_affinities(jax.device_put(high_dimensional_data, jax.devices('gpu')[0]), target_perplexity, perp_tol, use_ntk) * scaling_factor
    P = jnp.clip(P, EPSILON, None)

    init_mean = jnp.zeros(num_dimensions, dtype=jnp.float32)
    init_cov = jnp.eye(num_dimensions, dtype=jnp.float32) * 1e-4

    # Ensure the random key is generated correctly
    rand = random.PRNGKey(random_state)
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

