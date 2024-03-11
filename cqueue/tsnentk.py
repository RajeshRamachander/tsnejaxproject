from jax import random
import jax
import jax.numpy as jnp
from jax import jit
from tqdm import tqdm
from neural_tangents import stax
from jax.experimental import host_callback
from jax import devices
from jax.nn import softmax



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

        # Reshape the data for the convolutional network
        # X = inputs.reshape(-1, 8, 8, 1)  # Reshape to [batch_size, height, width, channels]
        # X = X / 16.0  # Normalize pixel values

        # init_fn, apply_fn, kernel_fn = stax.serial(
        #     stax.Conv(32, (3, 3), padding="SAME"), stax.Relu(),
        #     stax.Conv(64, (3, 3), padding="SAME"), stax.Relu(),
        #     stax.Conv(128, (3, 3), padding="SAME"), stax.Relu(),
        #     stax.AvgPool((2, 2)),  # Average pooling can be used to reduce dimensionality
        #     stax.Flatten(),
        #     stax.Dense(100), stax.Relu(),
        #     stax.Dense(10)
        # )

        # init_fn, apply_fn, kernel_fn = stax.serial(
        #     # First Convolutional Block
        #     stax.Conv(32, (3, 3), padding='SAME'),  # 32 filters, 3x3 kernel
        #     stax.Relu(),
        #
        #     # Second Convolutional Block
        #     stax.Conv(64, (3, 3), padding='SAME'),  # 64 filters, 3x3 kernel
        #     stax.Relu(),
        #
        #     # Third Convolutional Block
        #     stax.Conv(128, (3, 3), padding='SAME'),  # 128 filters, 3x3 kernel
        #     stax.Relu(),
        #     stax.AvgPool((2, 2), strides=(2, 2)),  # Average pooling with 2x2 kernel
        #
        #     # Fully Connected Layers
        #     stax.Flatten(),  # Flatten the output of the last pooling layer
        #     stax.Dense(256),  # Fully connected layer with 256 units
        #     stax.Relu(),
        #
        #     # Output Layer
        #     stax.Dense(10),  # Adjust the number of units to match the number of classes
        # )

        # Define your neural network architecture
        init_fn, apply_fn, kernel_fn = stax.serial(
            stax.Dense(8192), stax.Relu(),
            stax.Dense(4098), stax.Relu(),
            stax.Dense(2048), stax.Relu(),
            stax.Flatten(), stax.Dense(1)
        )

        # Compute the neural tangent kernel
        return kernel_fn(inputs, inputs, 'ntk')


    # Compute the NTK matrix
    ntk_matrix = compute_ntk_matrix(data)
    # Apply softmax for normalization
    P = softmax(ntk_matrix, axis=1)

    #exp(Xi) / sum over i (exp(Xi))

    #exp(Xi/sigma i) / sum over i (exp (Xi/ sigma i))
    # Set the diagonal to zero (optional)
    P = P.at[jnp.diag_indices_from(P)].set(0)

    return P


@jit
def pairwise_affinities(data):

    P = compute_probabilities_from_ntk(data)
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


def compute_low_dimensional_embedding_ntk(high_dimensional_data, num_dimensions,max_iterations=100,
                                          learning_rate=100, scaling_factor=4.,random_state=42,
                                          ):

    all_devices = devices()
    if any('gpu' in dev.platform.lower() for dev in all_devices):
        jax.config.update('jax_platform_name', 'gpu')
        print('Using GPU')
        high_dimensional_data = jax.device_put(high_dimensional_data, jax.devices('gpu')[0])
        print('Data is on GPU')


    P = pairwise_affinities(jax.device_put(high_dimensional_data, jax.devices('gpu')[0])) * scaling_factor
    P = jnp.clip(P, EPSILON, None)

    init_mean = jnp.zeros(num_dimensions, dtype=jnp.float32)
    init_cov = jnp.eye(num_dimensions, dtype=jnp.float32) * 1e-4

    # Ensure the random key is generated correctly
    rand = random.PRNGKey(random_state)
    Y = random.multivariate_normal(rand, mean=init_mean, cov=init_cov, shape=(high_dimensional_data.shape[0],))

    print(f'Shape of P {P.shape}')

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

