from jax import random
import jax
import jax.numpy as jnp
from jax import jit
from tqdm import tqdm
from neural_tangents import stax
from jax.experimental import host_callback
from jax import devices
from jax.nn import softmax
from jax.numpy.linalg import svd



EPSILON = 1e-12

@jit
def pca_jax(X, k=30):
    """
    Use PCA to project X to k dimensions using JAX.

    Parameters:
    X (jax.numpy.ndarray): The input data array.
    k (int): The number of principal components to retain.

    Returns:
    jax.numpy.ndarray: The projected data in k dimensions.
    """
    # Center and scale the data
    s = jnp.std(X, axis=0)
    s = jnp.where(s == 0, 1, s)  # Avoid division by zero
    X_centered_scaled = (X - jnp.mean(X, axis=0)) / s

    # Compute SVD
    U, S, Vh = svd(X_centered_scaled, full_matrices=False)

    # Project data onto the first k principal components
    X_pca = U[:, :k] * S[:k]

    return X_pca

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

def preprocess_inputs(inputs):
    X = inputs.reshape(-1, 8, 8, 1)  # Reshape
    X = X / 255.0  # Normalize
    return X

def basic_cnn_architecture():
    return stax.serial(
        stax.Conv(32, (3, 3), padding='SAME'),
        stax.Relu(),
        stax.AvgPool((2, 2), strides=(2, 2)),
        stax.Flatten(),
        stax.Dense(256),
        stax.Relu(),
        stax.Dense(10)
    )

def advanced_cnn_architecture():
    return stax.serial(
        stax.Conv(32, (3, 3), padding='SAME'),
        stax.Relu(),
        stax.AvgPool((2, 2), strides=(2, 2)),
        stax.Dropout(0.1),
        stax.Conv(64, (3, 3), padding='SAME'),
        stax.Relu(),
        stax.AvgPool((2, 2), strides=(2, 2)),
        stax.Dropout(0.2),
        stax.Conv(128, (3, 3), padding='SAME'),
        stax.Relu(),
        stax.GlobalSumPool(),
        stax.Dropout(0.3),
        stax.Dense(256),
        stax.Relu(),
        stax.Dropout(0.5),
        stax.Dense(10)
    )


def get_kernel_by_convlution(inputs):

    preprocessed_inputs = preprocess_inputs(inputs)

    init_fn, apply_fn, kernel_fn = advanced_cnn_architecture()

    return kernel_fn(preprocessed_inputs, preprocessed_inputs, 'nngp')

def get_kernel_by_deep_network(input):

    # Define your neural network architecture
    init_fn, apply_fn, kernel_fn = stax.serial(
        stax.Dense(8192), stax.Relu(),
        stax.Dense(4096), stax.Relu(),
        stax.Dense(2048), stax.Relu(),
        stax.Dense(1024), stax.Relu(),
        stax.Dense(512), stax.Relu(),
        stax.Dense(256), stax.Relu(),
        stax.Dense(128), stax.Relu(),
        stax.Flatten(), stax.Dense(10)
    )

    return kernel_fn(input, input, 'ntk')

def SimplifiedWideResnetBlock(channels, strides=(1, 1), channel_mismatch=False):
    Main = stax.serial(
        stax.Relu(), stax.Conv(channels, (3, 3), strides, padding='SAME'),
        stax.Relu(), stax.Conv(channels, (3, 3), padding='SAME'))
    Shortcut = stax.Identity() if not channel_mismatch else stax.Conv(
        channels, (3, 3), strides, padding='SAME')
    return stax.serial(stax.FanOut(2),
                       stax.parallel(Main, Shortcut),
                       stax.FanInSum())

def SimplifiedWideResnetGroup(n, channels, strides=(1, 1)):
    blocks = [SimplifiedWideResnetBlock(channels, strides, channel_mismatch=True)]
    for _ in range(n - 1):
        blocks.append(SimplifiedWideResnetBlock(channels, (1, 1)))
    return stax.serial(*blocks)

def SimplifiedWideResnet(block_size, k, num_classes):
    return stax.serial(
        stax.Conv(16, (3, 3), padding='SAME'),
        SimplifiedWideResnetGroup(block_size, int(16 * k)),  # Reduced number of groups
        SimplifiedWideResnetGroup(block_size, int(32 * k), ),
        SimplifiedWideResnetGroup(block_size, int(64 * k), ),
        SimplifiedWideResnetGroup(block_size, int(128 * k), ),
        SimplifiedWideResnetGroup(block_size, int(64 * k), ),
        SimplifiedWideResnetGroup(block_size, int(32 * k), ),
        SimplifiedWideResnetGroup(block_size, int(16 * k), ),
        stax.AvgPool((8, 8)),
        stax.Flatten(),
        stax.Dense(num_classes))



def get_kernel_by_resnet(inputs):

    preprocessed_inputs = preprocess_inputs(inputs)

    # Define your neural network architecture
    init_fn, apply_fn, kernel_fn = SimplifiedWideResnet(block_size=4, k=1, num_classes=10)

    return kernel_fn(preprocessed_inputs, preprocessed_inputs, 'ntk')



def HeNormal():
    def init(key, shape, dtype=jnp.float32):
        """He normal initializer."""
        fan_in = shape[1]  # Assuming shape is (n_features, n_units_out)
        std = jnp.sqrt(2.0 / fan_in)
        return std * random.normal(key, shape, dtype)
    return init

def get_kernel_by_deep_network2(input):

    # Define your neural network architecture
    init_fn, apply_fn, kernel_fn = stax.serial(
        stax.Dense(8192, W_std=1.5, b_std=0.05), stax.LayerNorm(), stax.Gelu(),
        stax.Dense(4096, W_std=1.5, b_std=0.05), stax.LayerNorm(), stax.Gelu(),
        stax.Dense(2048, W_std=1.5, b_std=0.05), stax.LayerNorm(), stax.Gelu(),
        stax.Dense(1024, W_std=1.5, b_std=0.05), stax.LayerNorm(), stax.Gelu(),
        stax.Dense(512, W_std=1.5, b_std=0.05), stax.LayerNorm(), stax.Gelu(),
        stax.Dense(256, W_std=1.5, b_std=0.05), stax.LayerNorm(), stax.Gelu(),
        stax.Dense(128, W_std=1.5, b_std=0.05), stax.LayerNorm(), stax.Gelu(),
        stax.Flatten(),
        stax.Dense(10)
    )

    return kernel_fn(input, input, 'ntk')

def get_kernel_by_deep_network2_adjusted(input):

    # Define an adjusted neural network architecture
    init_fn, apply_fn, kernel_fn = stax.serial(
        stax.Dense(8192 * 2, W_std=1.5, b_std=0.05), stax.LayerNorm(), stax.Gelu(),
        stax.Dense(4096 * 2, W_std=1.5, b_std=0.05), stax.LayerNorm(), stax.Gelu(),
        stax.Dense(2048 * 2, W_std=1.5, b_std=0.05), stax.LayerNorm(), stax.Gelu(),
        # Additional layer to increase depth
        stax.Dense(1024, W_std=1.5, b_std=0.05), stax.LayerNorm(), stax.Gelu(),
        stax.Dense(512, W_std=1.5, b_std=0.05), stax.LayerNorm(), stax.Gelu(),
        stax.Dense(256, W_std=1.5, b_std=0.05), stax.LayerNorm(), stax.Gelu(),
        stax.Dense(128, W_std=1.5, b_std=0.05), stax.LayerNorm(), stax.Gelu(),
        # Extra layer for more depth
        stax.Dense(64, W_std=1.5, b_std=0.05), stax.LayerNorm(), stax.Gelu(),
        stax.Flatten(),
        stax.Dense(10)
    )

    return kernel_fn(input, input, 'ntk')


def get_kernel_by_deep_network2_conv(input_shape):

    preproscessed_inputs = preprocess_inputs(input_shape)
    # Define a neural network architecture with convolutional layers
    init_fn, apply_fn, kernel_fn = stax.serial(
        # Convolutional Block 1
            stax.Conv(64, (3, 3), padding='SAME', W_std=1.5, b_std=0.05), stax.LayerNorm(), stax.Gelu(),
            stax.Conv(64, (3, 3), padding='SAME', W_std=1.5, b_std=0.05), stax.LayerNorm(), stax.Gelu(),
            stax.AvgPool((2, 2), strides=(2, 2)),

            # Convolutional Block 2
            stax.Conv(128, (3, 3), padding='SAME', W_std=1.5, b_std=0.05), stax.LayerNorm(), stax.Gelu(),
            stax.Conv(128, (3, 3), padding='SAME', W_std=1.5, b_std=0.05), stax.LayerNorm(), stax.Gelu(),
            stax.AvgPool((2, 2), strides=(2, 2)),

        # Transition to Dense Layers
        stax.Flatten(),
        stax.Dense(2048, W_std=1.5, b_std=0.05), stax.LayerNorm(), stax.Gelu(),
        stax.Dense(1024, W_std=1.5, b_std=0.05), stax.LayerNorm(), stax.Gelu(),

        # Output Layer
        stax.Dense(10, W_std=1.5, b_std=0.05)
    )

    return kernel_fn(preproscessed_inputs, preproscessed_inputs, 'ntk')

def get_kernel_by_deep_network2_conv2(input_shape):

    preproscessed_inputs = preprocess_inputs(input_shape)
    # Define a neural network architecture with convolutional layers
    init_fn, apply_fn, kernel_fn = stax.serial(
        # Convolutional Block 1
            stax.Conv(64, (3, 3), padding='SAME', W_std=1.5, b_std=0.05), stax.LayerNorm(), stax.Gelu(),
            stax.Conv(64, (3, 3), padding='SAME', W_std=1.5, b_std=0.05), stax.LayerNorm(), stax.Gelu(),
            stax.AvgPool((2, 2), strides=(2, 2)),

            # Convolutional Block 2
            stax.Conv(128, (3, 3), padding='SAME', W_std=1.5, b_std=0.05), stax.LayerNorm(), stax.Gelu(),
            stax.Conv(128, (3, 3), padding='SAME', W_std=1.5, b_std=0.05), stax.LayerNorm(), stax.Gelu(),
            stax.AvgPool((2, 2), strides=(2, 2)),

        # Additional Convolutional Block
        stax.Conv(512, (3, 3), padding='SAME', W_std=1.5, b_std=0.05), stax.LayerNorm(), stax.Gelu(),
        stax.Conv(512, (3, 3), padding='SAME', W_std=1.5, b_std=0.05), stax.LayerNorm(), stax.Gelu(),
        stax.AvgPool((2, 2), strides=(2, 2)),


        # Transition to Dense Layers
        stax.Flatten(),
        stax.Dropout(0.5),  # Adjust dropout rate as necessary
        stax.Dense(8192 * 2, W_std=1.5, b_std=0.05), stax.LayerNorm(), stax.Gelu(),
        stax.Dropout(0.5),  # Adjust dropout rate as necessary
        stax.Dense(4096 * 2, W_std=1.5, b_std=0.05), stax.LayerNorm(), stax.Gelu(),
        stax.Dropout(0.5),  # Adjust dropout rate as necessary
        stax.Dense(2048 * 2, W_std=1.5, b_std=0.05), stax.LayerNorm(), stax.Gelu(),
        stax.Dropout(0.2),


        # Output Layer
        stax.Dense(10, W_std=1.5, b_std=0.05)
    )

    return kernel_fn(preproscessed_inputs, preproscessed_inputs, 'ntk')


def get_kernel_by_deep_network2_conv3(input_shape):
  preprocessed_inputs = preprocess_inputs(input_shape)

  # Define convolutional block with reuse
  def conv_block(num_filters):
    return stax.serial(
        stax.Conv(num_filters, (3, 3), padding='SAME', W_std=1.5, b_std=0.05),
        stax.LayerNorm(),
        stax.Gelu(),
        stax.AvgPool((2, 2), strides=(2, 2))
    )

  # Define the network architecture
  init_fn, apply_fn, kernel_fn = stax.serial(
      # Convolutional blocks
      conv_block(64),
      conv_block(128),
      # Additional convolutional block (optional)
      # stax.serial(...),

      # Transition to dense layers
      stax.Flatten(),
      stax.Dropout(0.5),
      stax.Dense(8192, W_std=1.5, b_std=0.05),
      stax.LayerNorm(),
      stax.Gelu(),
      stax.Dropout(0.5),
      stax.Dense(4096, W_std=1.5, b_std=0.05),
      stax.LayerNorm(),
      stax.Gelu(),
      stax.Dropout(0.5),
      stax.Dense(2048, W_std=1.5, b_std=0.05),
      stax.LayerNorm(),
      stax.Gelu(),
      stax.Dropout(0.2),

      # Output layer
      stax.Dense(10, W_std=1.5, b_std=0.05)
  )

  return kernel_fn(preprocessed_inputs, preprocessed_inputs, 'ntk')



def get_kernel_by_deep_network2_conv_enhanced(input_shape):
    # Simulated preprocessing step (ensure this matches your real preprocessing)
    preprocessed_inputs = preprocess_inputs(input_shape)

    init_fn, apply_fn, kernel_fn = stax.serial(
            # Convolutional Block 1
            stax.Conv(64, (3, 3), padding='SAME', W_std=1.5, b_std=0.05), stax.LayerNorm(), stax.Gelu(),
            stax.Conv(64, (3, 3), padding='SAME', W_std=1.5, b_std=0.05), stax.LayerNorm(), stax.Gelu(),
            stax.AvgPool((2, 2), strides=(2, 2)),

            # Convolutional Block 2
            stax.Conv(128, (3, 3), padding='SAME', W_std=1.5, b_std=0.05), stax.LayerNorm(), stax.Gelu(),
            stax.Conv(128, (3, 3), padding='SAME', W_std=1.5, b_std=0.05), stax.LayerNorm(), stax.Gelu(),
            stax.AvgPool((2, 2), strides=(2, 2)),

            # Additional Convolutional Block
            stax.Conv(256, (3, 3), padding='SAME', W_std=1.5, b_std=0.05), stax.LayerNorm(), stax.Gelu(),
            stax.Conv(256, (3, 3), padding='SAME', W_std=1.5, b_std=0.05), stax.LayerNorm(), stax.Gelu(),
            stax.AvgPool((2, 2), strides=(2, 2)),

            # Additional Convolutional Block
            stax.Conv(512, (3, 3), padding='SAME', W_std=1.5, b_std=0.05), stax.LayerNorm(), stax.Gelu(),
            stax.Conv(512, (3, 3), padding='SAME', W_std=1.5, b_std=0.05), stax.LayerNorm(), stax.Gelu(),
            stax.AvgPool((2, 2), strides=(2, 2)),

            # Additional Convolutional Block
            stax.Conv(1024, (3, 3), padding='SAME', W_std=1.5, b_std=0.05), stax.LayerNorm(), stax.Gelu(),
            stax.Conv(1024, (3, 3), padding='SAME', W_std=1.5, b_std=0.05), stax.LayerNorm(), stax.Gelu(),
            stax.AvgPool((2, 2), strides=(2, 2)),

        stax.Conv(2048, (3, 3), padding='SAME', W_std=1.5, b_std=0.05), stax.LayerNorm(), stax.Gelu(),
        stax.Conv(2048, (3, 3), padding='SAME', W_std=1.5, b_std=0.05), stax.LayerNorm(), stax.Gelu(),
        stax.AvgPool((2, 2), strides=(2, 2)),

        stax.Conv(4096, (3, 3), padding='SAME', W_std=1.5, b_std=0.05), stax.LayerNorm(), stax.Gelu(),
        stax.Conv(4096, (3, 3), padding='SAME', W_std=1.5, b_std=0.05), stax.LayerNorm(), stax.Gelu(),
        stax.AvgPool((2, 2), strides=(2, 2)),

        # Transition to Dense Layers with Dropout for Regularization
        stax.Flatten(),
        stax.Dropout(0.5),  # Adjust dropout rate as necessary
        stax.Dense(8192 * 2, W_std=1.5, b_std=0.05), stax.LayerNorm(), stax.Gelu(),
        stax.Dropout(0.5),  # Adjust dropout rate as necessary
        stax.Dense(4096 * 2, W_std=1.5, b_std=0.05), stax.LayerNorm(), stax.Gelu(),
        stax.Dropout(0.5),  # Adjust dropout rate as necessary
        stax.Dense(2048 * 2, W_std=1.5, b_std=0.05), stax.LayerNorm(), stax.Gelu(),
        stax.Dropout(0.5),  # Adjust dropout rate as necessary
        # # Additional layer to increase depth
        # stax.Dense(1024, W_std=1.5, b_std=0.05), stax.LayerNorm(), stax.Gelu(),
        # stax.Dropout(0.5),  # Adjust dropout rate as necessary
        # stax.Dense(512, W_std=1.5, b_std=0.05), stax.LayerNorm(), stax.Gelu(),
        # stax.Dropout(0.5),  # Adjust dropout rate as necessary
        # stax.Dense(256, W_std=1.5, b_std=0.05), stax.LayerNorm(), stax.Gelu(),
        # stax.Dropout(0.5),  # Adjust dropout rate as necessary
        # stax.Dense(128, W_std=1.5, b_std=0.05), stax.LayerNorm(), stax.Gelu(),
        # stax.Dropout(0.5),  # Adjust dropout rate as necessary
        # # Extra layer for more depth
        # stax.Dense(64, W_std=1.5, b_std=0.05), stax.LayerNorm(), stax.Gelu(),
        # stax.Dropout(0.5),  # Adjust dropout rate as necessary

        # Output Layer
        stax.Dense(10)
    )

    return kernel_fn(preprocessed_inputs, preprocessed_inputs, 'ntk')


@jit
def compute_ntk_matrix(inputs):

    return get_kernel_by_deep_network2_adjusted(inputs)


@jit
def compute_pairwise_affinities(ntk_matrix, sigmas):
    """
    Optimized function that computes a probability matrix directly from the NTK matrix and sigma values.

    Parameters:
    ntk_matrix (jax.numpy.DeviceArray): A JAX array representing the Neural Tangent Kernel (NTK) matrix.
    sigmas (jax.numpy.DeviceArray): A JAX array of sigma values for scaling, one per row of the NTK matrix.

    Returns:
    jax.numpy.DeviceArray: The computed probability matrix with diagonal elements set to zero.
    """
    # Ensure sigmas is correctly shaped for row-wise broadcasting
    sigmas = sigmas.reshape(-1, 1)

    # # Compute the Gaussian kernel values
    # numers = jnp.exp(-ntk_matrix / (2 * (sigmas ** 2)))
    #
    # # Compute the normalization factors, excluding diagonal elements
    # denoms = jnp.sum(numers, axis=1) - jnp.diag(numers)
    # denoms = denoms[:, None] + EPSILON  # Reshape and ensure non-zero denominator
    #
    # # Calculate the pairwise affinities
    # P = numers / denoms


    # Normalize the NTK matrix by the sigma values for each row
    # normalized_scores = ntk_matrix / sigmas

    normalized_scores = ntk_matrix  / sigmas * sigmas.T
    # Apply softmax to the normalized scores
    P = softmax(normalized_scores, axis=1)

    # Set the diagonal elements to zero
    P = P.at[jnp.diag_indices_from(P)].set(0)

    return P

def print_attempts(value):
  """Prints the value on the host machine."""
  print(f"attempts remaining: {value}")

def print_perplexity_diff(value):
  """Prints the value on the host machine."""
  print(f"Perplexity difference: {value}")

def print_intial_perplexity(value):
  """Prints the value on the host machine."""
  print(f"Perplexity initial value: {value}")

@jit
def all_sym_affinities(data, perp, tol,  attempts=100):
    ntk_mat = compute_ntk_matrix(data)
    n_samples = data.shape[0]

    sigma_maxs = jnp.full(data.shape[0], 1e12)
    sigma_mins = jnp.full(data.shape[0], 1e-12)
    sigmas = (sigma_mins + sigma_maxs) / 2

    P = compute_pairwise_affinities(ntk_mat, sigmas.reshape(-1, 1))
    current_perps = calculate_row_wise_perplexities(P)

    host_callback.call(print_intial_perplexity,
                       current_perps)

    def condition(vals):
        _, attempts, _, _, current_perps, _ = vals
        # Calculate the absolute difference between current and desired perplexities
        perp_diff = jnp.abs(current_perps - perp)
        host_callback.call(print_attempts, attempts)
        host_callback.call(print_perplexity_diff,
                           jnp.mean(perp_diff))  # Calculate and print average perplexity difference
        # Check if average perplexity is within tolerance and if there are attempts left
        return jnp.logical_and(jnp.mean(perp_diff) > tol, attempts > 0)

    # Define the body of the loop for binary search
    def body(vals):
        sigmas, attempts, sigma_maxs, sigma_mins, current_perps, P = vals
        P = compute_pairwise_affinities(ntk_mat, sigmas.reshape(-1, 1))
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


def compute_low_dimensional_embedding_ntk(high_dimensional_data, num_dimensions,
                                      perplexity, max_iterations=100,
                                      learning_rate=10, scaling_factor=1.,
                                      random_state=42,
                                      perp_tol=0.1):
    all_devices = devices()
    if any('gpu' in dev.platform.lower() for dev in all_devices):
        jax.config.update('jax_platform_name', 'gpu')
        print('Using GPU')
        high_dimensional_data = jax.device_put(high_dimensional_data, jax.devices('gpu')[0])
        print('Data is on GPU')

    if high_dimensional_data.shape[1] > 30:
        high_dimensional_data = pca_jax(high_dimensional_data)

    P = all_sym_affinities(jax.device_put(high_dimensional_data, jax.devices('gpu')[0]), perplexity, perp_tol,
                           attempts=75) * scaling_factor
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

