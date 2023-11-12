import jax
import jax.numpy as jnp
from jax import random, grad, jit
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from jax import jacrev

from sklearn.utils import shuffle

from sklearn.datasets import load_iris

def load_dataset():
    data = load_iris()
    features = data.data
    features = (features - features.mean(axis=0)) / features.std(axis=0)  # Normalize
    return features

# Initialize parameters
def init_params(layer_sizes, rng_keys):
    params = []
    for i in range(len(layer_sizes) - 1):
        output_shape = layer_sizes[i+1]
        W = random.normal(rng_keys[i], (layer_sizes[i], output_shape)) / jnp.sqrt(layer_sizes[i])
        b = jnp.zeros(output_shape)
        params.append([W, b])
    return params

# Define the encoder and decoder
@jit
def encoder(params, x):
    activations = x
    for W, b in params:  # Ensure params is a list of tuples (W, b)
        activations = jax.nn.relu(jnp.dot(activations, W) + b)
    return activations

@jit
def decoder(params, z):
    activations = z
    for W, b in params[:-1]:
        activations = jax.nn.relu(jnp.dot(activations, W) + b)
    final_W, final_b = params[-1]
    return jax.nn.sigmoid(jnp.dot(activations, final_W) + final_b)

# Define the loss function
@jit
def compute_loss(params, x):
    encoded = encoder(params[0], x)
    decoded = decoder(params[1], encoded)
    return jnp.mean(jnp.square(decoded - x))

# Gradient function
grad_loss = jit(grad(compute_loss))

# Training function
@jit
def train(params, x, learning_rate=0.005):
    enc_params, dec_params = params
    grads = grad_loss([enc_params, dec_params], x)
    enc_grads, dec_grads = grads
    enc_params = [(W - learning_rate * dW, b - learning_rate * db) for (W, b), (dW, db) in zip(enc_params, enc_grads)]
    dec_params = [(W - learning_rate * dW, b - learning_rate * db) for (W, b), (dW, db) in zip(dec_params, dec_grads)]
    return enc_params, dec_params


from jax.tree_util import tree_flatten

def compute_ntk(enc_params, input_data):
    def single_input_jacobian(x):
        return jacrev(encoder, argnums=0)(enc_params, x)

    n_samples = input_data.shape[0]
    ntk_matrix = jnp.zeros((n_samples, n_samples))

    for i in range(n_samples):
        jac_i = single_input_jacobian(input_data[i])
        # Flatten the Jacobian structure
        flat_jac_i, _ = tree_flatten(jac_i)
        flat_jac_i = jnp.concatenate([jnp.ravel(a) for a in flat_jac_i])

        for j in range(n_samples):
            jac_j = single_input_jacobian(input_data[j])
            # Flatten the Jacobian structure
            flat_jac_j, _ = tree_flatten(jac_j)
            flat_jac_j = jnp.concatenate([jnp.ravel(a) for a in flat_jac_j])

            # Compute the dot product between flattened Jacobians
            ntk_matrix = ntk_matrix.at[i, j].set(jnp.sum(flat_jac_i * flat_jac_j))

    return ntk_matrix

# from jax import jacrev, vmap, jit
# import jax.numpy as jnp
# from jax.tree_util import tree_flatten
#
# @jit
# def compute_ntk(enc_params, input_data):
#     def single_input_jacobian(x):
#         return jacrev(encoder, argnums=0)(enc_params, x)
#
#     # Vectorized computation of Jacobians for all inputs
#     batch_jacobian = vmap(single_input_jacobian)(input_data)
#
#     # Flatten the Jacobians
#     flat_all_jacs = [jnp.concatenate([jnp.ravel(a) for a in tree_flatten(jacs)[0]]) for jacs in batch_jacobian]
#
#     # Ensure all flattened Jacobians have the same size
#     jac_lengths = [len(flat_jac) for flat_jac in flat_all_jacs]
#     if len(set(jac_lengths)) > 1:
#         raise ValueError("Inconsistent Jacobian sizes detected: ", set(jac_lengths))
#
#     n_samples = input_data.shape[0]
#     ntk_matrix = jnp.zeros((n_samples, n_samples))
#
#     # Compute the NTK matrix, leveraging its symmetry
#     for i in range(n_samples):
#         flat_jac_i = flat_all_jacs[i]
#         for j in range(i, n_samples):
#             flat_jac_j = flat_all_jacs[j]
#             ntk_ij = jnp.dot(flat_jac_i, flat_jac_j)
#             ntk_matrix = ntk_matrix.at[i, j].set(ntk_ij)
#             if i != j:
#                 ntk_matrix = ntk_matrix.at[j, i].set(ntk_ij)
#
#     return ntk_matrix
#


@jit
def ntk_to_similarity(ntk_matrix):
    # Convert NTK to similarity - assuming NTK represents a kind of 'distance'
    # Using exponential function to convert distances to similarity scores
    similarity_matrix = jnp.exp(-ntk_matrix)
    return similarity_matrix
@jit
def normalize_rows(matrix):
    # Normalize each row of the matrix so that it sums up to 1
    row_sums = matrix.sum(axis=1)
    normalized_matrix = matrix / row_sums[:, None]
    return normalized_matrix



# Main function
def main():
    # Load data
    features = load_dataset()

    # Simplify the network architecture
    encoder_layer_sizes = [4, 3, 2]
    decoder_layer_sizes = [2, 3, 4]
    rng_key = random.PRNGKey(0)
    rng_keys = random.split(rng_key, num=6)  # Generate 6 separate keys
    enc_params = init_params(encoder_layer_sizes, rng_keys[:3])
    dec_params = init_params(decoder_layer_sizes, rng_keys[3:])

    # Compute NTK before training

    ntk_matrix_before = compute_ntk(enc_params, features)
    plt.imshow(ntk_matrix_before, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title("NTK Matrix Heatmap")
    plt.show()
    return
    num_epochs = 100
    for epoch in range(num_epochs):
        enc_params, dec_params = train([enc_params, dec_params], features)

    # Compute NTK after training
    ntk_matrix_after = compute_ntk(enc_params, features)
    plt.imshow(ntk_matrix_after, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title("NTK Matrix Heatmap")
    plt.show()

    similarity_matrix = ntk_to_similarity(ntk_matrix_after)

    probability_matrix = normalize_rows(similarity_matrix)



    print(probability_matrix)
    print(similarity_matrix)
if __name__ == "__main__":
    main()
