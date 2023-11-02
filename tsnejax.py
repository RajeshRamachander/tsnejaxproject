import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import jit
from jax import lax
import jax
from jax import random

def compute_pairwise_distances(high_dimensional_data):
    """
    Compute pairwise distances between data points using JAX.

    Args:
        high_dimensional_data (jnp.ndarray): High-dimensional input data.

    Returns:
        jnp.ndarray: Pairwise distances matrix.
    """
    # Calculate pairwise distances using JAX
    pairwise_distances = jnp.sum((high_dimensional_data[:, None] - high_dimensional_data) ** 2, axis=-1)

    return pairwise_distances


def initialize_beta_values(num_data_points):
    """
    Initialize beta values for each data point.

    Args:
        num_data_points (int): Number of data points.

    Returns:
        jnp.ndarray: Initialized beta values.
    """
    return jnp.ones((num_data_points, 1))

@jit
def calculate_entropy_and_probabilities(pairwise_distances, beta=1.0):
    """
    Calculate entropy and probabilities from pairwise distances.

    Args:
        pairwise_distances (jnp.ndarray): Pairwise distances matrix.
        beta (float, optional): Beta parameter.

    Returns:
        tuple: Tuple containing entropy and probabilities.
    """
    # Calculate probabilities using the formula: exp(-pairwise_distances * beta)
    probabilities = jnp.exp(-pairwise_distances * beta)

    # Calculate the sum of probabilities
    sum_of_probabilities = jnp.sum(probabilities)

    # Calculate entropy using the formula
    entropy = jnp.log(sum_of_probabilities) + beta * jnp.sum(
        pairwise_distances * probabilities) / sum_of_probabilities

    # Normalize probabilities
    probabilities /= sum_of_probabilities
    return entropy, probabilities


def update_beta(beta, entropy_difference, beta_min, beta_max):
    """
    Update the beta value based on entropy difference.

    Args:
        beta (float): Current beta value.
        entropy_difference (float): Current entropy difference.
        beta_min (float): Minimum beta value.
        beta_max (float): Maximum beta value.

    Returns:
        tuple: Updated beta value, beta_min, and beta_max.
    """
    if entropy_difference > 0:
        beta_min = beta
        if beta_max == jnp.inf:
            beta = beta * 2.
        else:
            beta = (beta + beta_max) / 2.
    else:
        beta_max = beta
        if beta_min == -jnp.inf:
            beta = beta / 2.
        else:
            beta = (beta + beta_min) / 2.
    return beta, beta_min, beta_max



def compute_pairwise_probabilities(high_dimensional_data, tolerance=1e-5, target_perplexity=30.0, scaling_factor = 4.):
    num_data_points = high_dimensional_data.shape[0]
    pairwise_distances = compute_pairwise_distances(high_dimensional_data)
    pairwise_probabilities = jnp.zeros((num_data_points, num_data_points))
    beta_values = initialize_beta_values(num_data_points)
    log_target_perplexity = jnp.log(target_perplexity)

    for i in range(num_data_points):

        beta_min = -jnp.inf
        beta_max = jnp.inf
        distances_i = pairwise_distances[i, compute_neighboring_indices(i, num_data_points)]
        entropy, this_probabilities = calculate_entropy_and_probabilities(distances_i, beta_values[i])
        entropy_difference = entropy - log_target_perplexity
        num_tries = 0
        while jnp.abs(entropy_difference) > tolerance and num_tries < 50:
            beta_value = beta_values[i]
            beta_value, beta_min, beta_max = update_beta(beta_value, entropy_difference, beta_min, beta_max)
            beta_values = beta_values.at[i].set(beta_value)
            entropy, this_probabilities = calculate_entropy_and_probabilities(distances_i, beta_values[i])
            entropy_difference = entropy - log_target_perplexity
            num_tries += 1

        pairwise_probabilities = pairwise_probabilities.at[i, compute_neighboring_indices(i, num_data_points)].set(
            this_probabilities)

    pairwise_probabilities += jnp.transpose(pairwise_probabilities)
    pairwise_probabilities /= jnp.sum(pairwise_probabilities)
    pairwise_probabilities *= scaling_factor

    # Apply element-wise maximum using JAX
    pairwise_probabilities = jnp.maximum(pairwise_probabilities, 1e-12)

    return pairwise_probabilities


def preprocess_high_dimensional_data(high_dimensional_data):
    """
    Preprocess high-dimensional data using JAX operations.

    Args:
        high_dimensional_data (np.ndarray): High-dimensional input data.

    Returns:
        np.ndarray: Preprocessed high-dimensional data.
    """
    num_data_points = high_dimensional_data.shape[0]

    # Convert high-dimensional data to JAX array
    high_dimensional_data = jnp.array(high_dimensional_data)

    # Subtract the mean along axis 0
    high_dimensional_data -= jnp.mean(high_dimensional_data, axis=0)

    # Singular Value Decomposition (SVD)
    _, S, _ = jnp.linalg.svd(high_dimensional_data, full_matrices=False)

    # Normalize high-dimensional data
    high_dimensional_data /= jnp.sqrt(S[0])

    return high_dimensional_data


def initialize_embedding_and_gradients(num_data_points, num_dimensions):
    """
    Initialize low-dimensional embedding, gradient, and gains.

    Args:
        num_data_points (int): Number of data points.
        num_dimensions (int): Number of dimensions for the low-dimensional embedding.

    Returns:
        (np.ndarray, np.ndarray, np.ndarray, np.ndarray): Tuple containing initialized
        low-dimensional embedding, gradient, previous gradient, and gains.
    """
    key = random.PRNGKey(0)  # Initialize a random key

    # Initialize low-dimensional embedding
    low_dimensional_embedding = random.normal(key, (num_data_points, num_dimensions))

    # Initialize gradient and gains
    gradient = jnp.zeros((num_data_points, num_dimensions))
    previous_gradient = jnp.zeros((num_data_points, num_dimensions))
    gains = jnp.ones((num_data_points, num_dimensions))

    return low_dimensional_embedding, gradient, previous_gradient, gains


def compute_neighboring_indices(i, num_data_points):
    """
    Compute indices of neighboring data points.

    Args:
        i (int): Index of the current data point.
        num_data_points (int): Number of data points.

    Returns:
        jnp.ndarray: Indices of neighboring data points.
    """
    return jnp.concatenate((jnp.arange(i), jnp.arange(i + 1, num_data_points)))



def compute_low_dimensional_embedding(high_dimensional_data, num_dimensions,
                                      target_perplexity, max_iterations=1000,
                                      learning_rate=500):
    num_data_points = high_dimensional_data.shape[0]

    high_dimensional_data = preprocess_high_dimensional_data(high_dimensional_data)

    low_dimensional_embedding, gradient, previous_gradient, gains = initialize_embedding_and_gradients(num_data_points,
                                                                                                       num_dimensions)

    def body_carry(carry):
        low_dim_emb, grad, prev_grad, gains, iteration = carry

        sum_of_squared_low_dimensional_embedding = jnp.sum(jnp.square(low_dim_emb), axis=1)
        num = -2. * jnp.dot(low_dim_emb, low_dim_emb.T)
        num = 1. / (1. + num + sum_of_squared_low_dimensional_embedding + sum_of_squared_low_dimensional_embedding[
                                                                          :, None])
        diag_indices = jnp.arange(num_data_points)
        num = jnp.where(diag_indices[:, None] == diag_indices, 0., num)

        pairwise_similarity_q = num / jnp.sum(num)
        pairwise_similarity_q = jnp.maximum(pairwise_similarity_q, 1e-12)

        KL_divergence = jnp.sum(pairwise_probabilities * jnp.log(
            pairwise_probabilities / pairwise_similarity_q))

        pairwise_similarity_pq = pairwise_probabilities - pairwise_similarity_q

        def compute_gradient(i):
            product_matrix = pairwise_similarity_pq[:, i] * num[:, i]
            tiled_matrix = jnp.tile(product_matrix, (num_dimensions, 1)).T
            difference_matrix = low_dim_emb[i, :] - low_dim_emb
            element_wise_product = tiled_matrix * difference_matrix
            gradient_value = jnp.sum(element_wise_product, axis=0)
            gradient_value = jnp.sum(
                (pairwise_similarity_pq[:, i] * num[:, i])[:, jnp.newaxis] *
                (low_dim_emb[i, :] - low_dim_emb), axis=0)
            return gradient_value

        gradient = jax.vmap(compute_gradient)(jnp.arange(num_data_points))

        # Use separate functions for true_fun and false_fun
        def true_fun(x):
            return 0.5

        def false_fun(x):
            return 0.8

        # Conditionally determine momentum
        momentum = lax.cond(iteration < 20, true_fun, false_fun, None)

        gains = (gains + 0.2) * ((gradient > 0.) != (prev_grad > 0.)) + (
                gains * 0.8) * ((gradient > 0.) == (prev_grad > 0.))
        gains = jnp.maximum(gains, 0.01)
        prev_grad = momentum * prev_grad - learning_rate * (gains * gradient)
        low_dim_emb += prev_grad
        low_dim_emb -= jnp.tile(jnp.mean(low_dim_emb, axis=0), (num_data_points, 1))

        return (low_dim_emb, gradient, prev_grad, gains, iteration + 1)

    def cond_carry(carry):
        _, _, _, gains, iteration = carry
        return jnp.all(gains > 0.01) & (iteration < max_iterations)

    pairwise_probabilities = compute_pairwise_probabilities(high_dimensional_data, 1e-5, target_perplexity)
    (low_dimensional_embedding, _, _, _, _) = lax.while_loop(cond_carry, body_carry,
                                                             ((
                                                             low_dimensional_embedding, gradient, previous_gradient,
                                                             gains, 0)))

    return low_dimensional_embedding


# Synthetic Data
# Define constant synthetic cluster data
cluster_1 = np.array([[1.0, 1.0] for _ in range(100)])
cluster_2 = np.array([[2.0, 2.0] for _ in range(100)])
cluster_3 = np.array([[3.0, 3.0] for _ in range(100)])
data = np.vstack([cluster_1, cluster_2, cluster_3])

perplexity_value = 30

Y_custom = compute_low_dimensional_embedding(data, num_dimensions=2, target_perplexity=perplexity_value)

plt.scatter(Y_custom[:, 0], Y_custom[:, 1], 20, range(len(Y_custom)))
plt.title('Custom t-SNE transformed Data')
plt.grid(True)
plt.show()
#
# # Generate a synthetic dataset with three clusters
# np.random.seed(42)
#
# # Cluster 1: centered at (0, 0)
# cluster_1 = np.random.randn(100, 2)
#
# # Cluster 2: centered at (10, 10)
# cluster_2 = np.random.randn(100, 2) + 10
#
# # Cluster 3: centered at (-10, 10)
# cluster_3 = np.random.randn(100, 2) + np.array([-10, 10])
#
# X = np.vstack([cluster_1, cluster_2, cluster_3])
#
# # Plotting
# plt.figure(figsize=(18, 6))
#
# # Original Data
# plt.subplot(1, 3, 1)
# plt.scatter(X[:, 0], X[:, 1], 20, range(len(X)))
# plt.title('Original Data')
# plt.grid(True)
#
# # Custom t-SNE transformation
# perplexity_value = 30
# Y_custom = compute_low_dimensional_embedding(X, 2, 2, perplexity_value)
# plt.subplot(1, 3, 2)
# plt.scatter(Y_custom[:, 0], Y_custom[:, 1], 20, range(len(Y_custom)))
# plt.title('Custom t-SNE transformed Data')
# plt.grid(True)
#
