from jax import random
import jax
import jax.numpy as jnp
from neural_tangents import stax
from jax import devices
from jax import jit

from tsne_common import (
    pca_jax,
    run_tsne_algorithm
)

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


def compute_low_dimensional_embedding_ntk(high_dimensional_data, num_dimensions,
                                      perplexity, max_iterations=100,
                                      learning_rate=10, scaling_factor=1.,
                                      random_state=42,
                                      perp_tol=1e-6):
    all_devices = devices()
    if any('gpu' in dev.platform.lower() for dev in all_devices):
        jax.config.update('jax_platform_name', 'gpu')
        print('Using GPU')
        high_dimensional_data = jax.device_put(high_dimensional_data, jax.devices('gpu')[0])
        print('Data is on GPU')

    if high_dimensional_data.shape[1] > 30:
        high_dimensional_data = pca_jax(high_dimensional_data)

    data_mat = compute_ntk_matrix(high_dimensional_data)

    Y = run_tsne_algorithm(data_mat, perplexity, perp_tol, scaling_factor,
                       num_dimensions, max_iterations,
                       learning_rate, random_state)

    print(Y.shape)
    return Y[-1]

