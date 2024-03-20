from jax import random
import jax
import jax.numpy as jnp
from jax import jit
from tqdm import tqdm, trange
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
def get_pij(d, scale, i):
    """
    Compute probabilities conditioned on point i from a row of distances
    d and a Gaussian scale (scale = 2*sigma^2).
    """

    d_scaled = -d / scale
    d_scaled -= jnp.max(d_scaled)
    exp_D = jnp.exp(d_scaled)
    exp_D = exp_D.at[i].set(0)
    return exp_D / jnp.sum(exp_D)

@jit
def entropy_py(p):
    """Calculates 2 ** H(p) of array p, where H(p) is the Shannon entropy."""
    return 2 ** jnp.sum(-p * jnp.log2(p + 1e-10))
@jit
def all_sym_affinities(data, perp, tol, attempts=250):
    ntk_mat = compute_ntk_matrix(data)  # Ensure this function is JAX-compatible
    n_samples = data.shape[0]

    # Correctly initialize P outside the loop
    P = jnp.zeros(ntk_mat.shape)

    def body_fun(i, P):
        sigma_max = 1e4
        sigma_min = 0.0
        d = ntk_mat[i, :]

        def cond_fun(val):
            sigma_min, sigma_max, _ = val
            return jnp.abs(sigma_max - sigma_min) > tol

        def body_fun(val):
            sigma_min, sigma_max, p_ij = val
            sigma_mid = (sigma_min + sigma_max) / 2
            scale = 2 * sigma_mid ** 2
            p_ij = get_pij(d, scale, i)  # Ensure get_pij is JAX-compatible
            current_perp = entropy_py(p_ij)  # Ensure entropy_py is JAX-compatible

            update_cond = current_perp < perp
            sigma_min = jax.lax.cond(update_cond, lambda: sigma_mid, lambda: sigma_min)
            sigma_max = jax.lax.cond(update_cond, lambda: sigma_max, lambda: sigma_mid)
            return sigma_min, sigma_max, p_ij

        _, _, p_ij = jax.lax.while_loop(cond_fun, body_fun, (sigma_min, sigma_max, jnp.zeros_like(d)))

        # Update P correctly using the result from while_loop
        P = P.at[i, :].set(p_ij)
        return P

    # Use lax.fori_loop to iterate over samples and update P
    P = jax.lax.fori_loop(0, n_samples, body_fun, P)
    return (P + P.T) / (2 * n_samples)


@jit
def compute_grad(P, Q, Y_dists, Y):
    pq_factor = P-Q

    # Vectorized operation to compute gradient contributions for all pairs
    Ydiff = Y[:, None, :] - Y[None, :, :]  # Shape: (n, n, num_dims)
    grad = 4 * jnp.sum(pq_factor[:, :, None] * Ydiff * Y_dists[:, :, None], axis=1)

    return grad

@jit
def low_dim_affinities(Y):
    D = compute_pairwise_distances(Y)
    Y_dists = jnp.power(1 + D , -1)
    n = Y_dists.shape[0]
    Y_dists_no_diag = Y_dists.at[jnp.diag_indices(n)].set(0)
    return Y_dists_no_diag / jnp.sum(Y_dists_no_diag), Y_dists


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
                                      perp_tol=1e-6):

    if high_dimensional_data.shape[1] > 30:
        print(f'No of columns in the dataset: {high_dimensional_data.shape[1]}')
        high_dimensional_data = pca_jax(high_dimensional_data)
        print(f'After PCA the number of columns reduced to: {high_dimensional_data.shape[1]}')
    else:
        print('no reduction as number of colums: {high_dimensional_data.shape[1]}')

    all_devices = devices()
    if any('gpu' in dev.platform.lower() for dev in all_devices):
        jax.config.update('jax_platform_name', 'gpu')
        print('Using GPU')

    P = all_sym_affinities(jax.device_put(high_dimensional_data, jax.devices('gpu')[0]), perplexity, perp_tol,
                           attempts=75) * scaling_factor


    size = (P.shape[0], num_dimensions)
    Y = jnp.zeros(shape=(max_iterations + 2, size[0], num_dimensions))
    key = random.PRNGKey(random_state)
    initial_vals = random.normal(key, shape=size) * jnp.sqrt(1e-4)

    Y = Y.at[0].set(initial_vals)
    Y = Y.at[1].set(initial_vals)
    Y_m1 = initial_vals
    Y_m2 = initial_vals


    for i in trange(2, max_iterations + 2, disable=False):

        Q, Y_dist_mat = low_dim_affinities(Y_m1)


        grad = compute_grad(P, Q, Y_dist_mat, Y_m1)

        # Update embeddings.
        Y_new = Y_m1 - learning_rate * grad + momentum_func(i) * (Y_m1 - Y_m2)


        Y_m2, Y_m1 = Y_m1, Y_new
        Y = Y.at[i, :, :].set(Y_new)

    print(Y.shape)
    return Y[-1]

