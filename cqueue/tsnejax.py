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
        # X = inputs.reshape(-1, 8, 8, 1)  # Reshape to [batch_size, height, width, channels]
        # X = X / 16.0  # Normalize pixel values

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
        return kernel_fn(inputs, inputs, 'ntk')


    # Compute the NTK matrix
    ntk_matrix = compute_ntk_matrix(data)
    # Apply softmax for normalization
    P = jnp.exp(ntk_matrix) / jnp.sum(jnp.exp(ntk_matrix), axis=1, keepdims=True)

    # Set the diagonal to zero (optional)
    P = P.at[jnp.diag_indices_from(P)].set(0)

    return P


@jit
def pairwise_affinities(data):

    return compute_probabilities_from_ntk(data)

@jit
def all_sym_affinities(data, perp, tol, attempts=100):

    current_perps = jnp.full(data.shape[0], jnp.inf)
    P = pairwise_affinities(data)

    def outer_while_condition(args):
        current_perps, perp, tol, attempts, P = args
        return jnp.logical_and(
            jnp.logical_not(jnp.allclose(current_perps, perp, atol=tol)),
            attempts > 0
        )

    def outer_while_body(args):
        current_perps, perp, tol, attempts, P= args
        P = pairwise_affinities(data)
        current_perps = calculate_row_wise_perplexities(P)
        attempts -= 1

        return current_perps, perp, tol, attempts, P

    outer_while_args = (current_perps, perp, tol, attempts,  P)
    (_, _, _, _, P) = jax.lax.while_loop(outer_while_condition, outer_while_body, outer_while_args)

    P = (P + P.T) / (2 * data.shape[0])

    return P

@jit
def low_dim_affinities(Y_dist_mat):
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
def low_dim_affinities3(Y_dist_mat):
    """
    Computes the low-dimensional affinities matrix Q.

    Parameters:
    Y : Low-dimensional representation of the data, ndarray of shape (n_samples, n_components)
    Y_dist_mat : Y distance matrix; ndarray of shape (n_samples, n_samples)

    Returns:
    Q : Symmetric low-dimensional affinities matrix of shape (n_samples, n_samples)
    """

    Q = jnp.exp(Y_dist_mat) / jnp.sum(jnp.exp(Y_dist_mat), axis=1, keepdims=True)

    # Set the diagonal to zero (optional)
    Q = Q.at[jnp.diag_indices_from(Q)].set(0)

    return Q

@jit
def low_dim_affinities2(Y):
    """Computes the low-dimensional affinities matrix Q using NTK."""
    Q = compute_probabilities_from_ntk(Y)  # Call the NTK function
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
def compute_grad2(P, Q, Y):
    """
    Computes the gradient vector needed to update the Y values using NTK affinities,
    with added stability checks.

    Parameters:
        P: Symmetric affinities matrix of shape (n_samples, n_samples)
        Q: Symmetric low-dimensional affinities matrix (NTK-based) of shape (n_samples, n_samples)
        Y: Low-dimensional representation of the data, ndarray of shape (n_samples, n_components)

    Returns:
        grad: The gradient vector, shape (n_samples, n_components)
    """

    # Calculate difference between points with added stability
    Ydiff = Y[:, jnp.newaxis, :] - Y[jnp.newaxis, :, :]  # Shape: (n_samples, n_samples, n_components)

    # Compute difference in affinities with stability in mind
    pq_factor = P - Q  # Shape: (n_samples, n_samples)

    # Expand pq_factor across the feature dimension for element-wise multiplication with Ydiff
    pq_factor_expanded = pq_factor[:, :, None]  # Shape: (n_samples, n_samples, 1)

    # Apply a small epsilon to avoid division by zero in subsequent operations (if any)
    EPSILON = 1e-12
    pq_factor_safe = pq_factor_expanded + EPSILON

    # Ensure Ydiff does not contribute to instability
    # This step is optional and based on the assumption that extreme values in Ydiff could cause issues
    Ydiff_safe = jnp.where(jnp.abs(Ydiff) < EPSILON, 0, Ydiff)

    # Correct approach to apply pq_factor to each feature dimension of Ydiff
    # Using safe versions of pq_factor and Ydiff to ensure stability
    grad = 4 * jnp.sum(pq_factor_safe * Ydiff_safe, axis=1)

    return grad



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


    P = all_sym_affinities(jax.device_put(high_dimensional_data, jax.devices('gpu')[0]), target_perplexity, perp_tol) * scaling_factor
    P = jnp.clip(P, EPSILON, None)

    init_mean = jnp.zeros(num_dimensions, dtype=jnp.float32)
    init_cov = jnp.eye(num_dimensions, dtype=jnp.float32) * 1e-4

    # Ensure the random key is generated correctly
    rand = random.PRNGKey(random_state)
    Y = random.multivariate_normal(rand, mean=init_mean, cov=init_cov, shape=(high_dimensional_data.shape[0],))

    print(f'Shape of P {P.shape}')

    Y_old = jnp.zeros_like(Y)

    iter_range = range(max_iterations)
    if pbar:
        iter_range = tqdm(iter_range, "Iterations")
    for t in iter_range:
        Y_dist_mat = compute_pairwise_distances(Y)
        Q = low_dim_affinities(Y_dist_mat)
        # Q = low_dim_affinities3(Y_dist_mat)
        Q = jnp.clip(Q, EPSILON, None)
        print(f'Shape of Q {Q.shape}')
        print(f'Shape of Y {Y.shape}')
        grad = compute_grad(P, Q, Y, Y_dist_mat)
        # grad = compute_grad2(P, Q, Y)
        Y = Y - learning_rate * grad + momentum_func(t) * (Y - Y_old)
        Y_old = Y.copy()
        if t == 100:
            P = P / scaling_factor
            pass
        pass

    return Y

