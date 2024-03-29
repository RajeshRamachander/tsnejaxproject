
import jax
from jax import jit
from jax.nn import softmax
from jax.numpy.linalg import svd
import jax.numpy as jnp
from jax import random
from ntk import compute_ntk_matrix

Low_dimensional_chunking_limit_size = 2000
High_dimensional_chunking_limit_size = 200

@jit
def pca_jax(X, k=30):
    # Center and scale the data
    s = jnp.std(X, axis=0)
    s = jnp.where(s == 0, 1, s)  # Avoid division by zero
    X_centered_scaled = (X - jnp.mean(X, axis=0)) / s

    # Compute SVD
    U, S, Vh = svd(X_centered_scaled, full_matrices=False)

    # Project data onto the first k principal components
    X_pca = X_centered_scaled @ Vh.T[:, :k]

    return X_pca

@jit
def compute_pairwise_distances(chunk1, chunk2):
    # Example for Euclidean distances
    sum_X = jnp.sum(jnp.square(chunk1), axis=1)
    sum_Y = jnp.sum(jnp.square(chunk2), axis=1)
    D = sum_X[:, None] + sum_Y[None, :] - 2 * jnp.dot(chunk1, chunk2.T)
    return D
# def compute_pairwise_distances(dim_data):
#     # Efficient broadcasting for pairwise squared Euclidean distances
#     sum_X = jnp.sum(jnp.square(dim_data), axis=1)
#     D = sum_X[:, None] - 2 * jnp.dot(dim_data, dim_data.T) + sum_X[None, :]
#     return D

@jit
def get_eculidean_probability_at_ij(d, scale, i):
    d_scaled = -d / scale
    d_scaled -= jnp.max(d_scaled)
    exp_D = jnp.exp(d_scaled)
    exp_D = exp_D.at[i].set(0)
    epsilon = 1e-8
    return exp_D / (jnp.sum(exp_D)+epsilon)

@jit
def get_ntk_probability_at_ij(d, sigma, i):
    # No negation for NTK values, as larger values indicate stronger influence.
    d_scaled = d / (sigma**2)
    d_scaled -= d_scaled.min()  # Min normalization
    d_scaled = d_scaled.at[i].set(-jnp.inf)
    probabilties = softmax(d_scaled/.5)

    return probabilties

@jit
def get_probability_at_ij(d, sigma, i, is_ntk):

    def true_fun(_):
        return get_ntk_probability_at_ij(d, sigma, i)

    def false_fun(_):
        return get_eculidean_probability_at_ij(d, sigma, i)

    return jax.lax.cond(is_ntk, true_fun, false_fun, None)


@jit
def get_shannon_entropy(p):
    """2 ** H(p) of array p, where H(p) is the Shannon entropy."""
    return 2 ** jnp.sum(-p * jnp.log2(p + 1e-10))

def print_attempts(value):
  """Prints the value on the host machine."""
  print(f"attempts done: {value}")

@jit
def all_sym_affinities(data_mat, perp, tol,attempts=250,  is_ntk=True):

    n_samples = data_mat.shape[0]

    # Correctly initialize P outside the loop
    P = jnp.zeros(data_mat.shape)

    def body_fun(i, P):
        sigma_max = 1e4
        sigma_min = 0.0
        d = data_mat[i, :]

        def cond_fun(val):
            sigma_min, sigma_max, _, attempts_counter = val
            # host_callback.call(print_attempts, attempts_counter)
            return (jnp.abs(sigma_max - sigma_min) > tol) & (attempts_counter < attempts)

        def body_fun(val):
            sigma_min, sigma_max, p_ij, attempts_counter = val
            sigma_mid = (sigma_min + sigma_max) / 2
            scale = 2 * sigma_mid ** 2
            p_ij = get_probability_at_ij(d, scale, i, is_ntk)
            current_perp = get_shannon_entropy(p_ij)

            update_cond = current_perp < perp
            sigma_min = jax.lax.cond(update_cond, lambda: sigma_mid, lambda: sigma_min)
            sigma_max = jax.lax.cond(update_cond, lambda: sigma_max, lambda: sigma_mid)
            return sigma_min, sigma_max, p_ij, attempts_counter + 1

        _, _, p_ij, attempts_counter = jax.lax.while_loop(cond_fun, body_fun, (sigma_min, sigma_max, jnp.zeros_like(d), 0))
        # host_callback.call(print_attempts, attempts_counter)
        # Update P correctly using the result from while_loop
        P = P.at[i, :].set(p_ij)
        return P

    # Use lax.fori_loop to iterate over samples and update P
    P = jax.lax.fori_loop(0, n_samples, body_fun, P)
    return (P + P.T) / (2 * n_samples)

@jit
def compute_grad(R, Y_dists, Y):

    # Expand dimensions to support broadcasting for vectorized subtraction
    Y_expanded = Y[:, None, :]  # Shape becomes (n, 1, num_dimensions)

    # Compute pairwise differences using broadcasting, result has shape (n, n, num_dimensions)
    pairwise_diff = Y_expanded - Y[None, :, :]

    # Element-wise multiplication of R and Y_dists, then further element-wise multiplication by pairwise differences
    # R and Y_dists are expanded to match the shape for broadcasting
    grad_contributions = 4 * R[:, :, None] * pairwise_diff * Y_dists[:, :, None]

    # Sum over the second axis (j index in the loop) to aggregate contributions to each point i
    dY = jnp.sum(grad_contributions, axis=1)

    return dY

@jit
def compute_data_matrix(chunk1, chunk2, is_ntk):
    # def compute_data_matrix(high_dimensional_data, is_ntk):
    def true_fun(_):
        # return compute_ntk_matrix(high_dimensional_data)
        return compute_ntk_matrix(chunk1, chunk2)

    def false_fun(_):
        # return compute_pairwise_distances(high_dimensional_data)
        return compute_pairwise_distances(chunk1, chunk2)

    return jax.lax.cond(is_ntk,
                        true_fun,
                        false_fun,
                        None)


def chunk_compute_assemble_distances(Y, chunk_size=Low_dimensional_chunking_limit_size):
    n = int(Y.shape[0])
    # Initialize a zero matrix for the full pairwise distances
    full_distance_matrix = jnp.zeros((n, n))

    if n > chunk_size:
        # Compute pairwise distances in chunks
        for i in range(0, n, int(chunk_size)):
            for j in range(0, n, int(chunk_size)):
                # Extract chunks
                chunk1 = Y[i:i + chunk_size]
                chunk2 = Y[j:j + chunk_size]

                # Compute distances for the current chunks
                distances_chunk = compute_pairwise_distances(chunk1, chunk2)

                # Update the corresponding part of the full distance matrix
                # Note: JAX arrays are immutable, so we need to use the .at[].set method for updates
                full_distance_matrix = full_distance_matrix.at[i:i + chunk_size, j:j + chunk_size].set(distances_chunk)

        return full_distance_matrix
    else:
        return compute_pairwise_distances(Y, Y)


def low_dim_affinities(Y):
    # D = compute_pairwise_distances(Y)
    D = chunk_compute_assemble_distances(Y)
    Y_dists = jnp.power(1 + D , -1)
    n = Y_dists.shape[0]
    Y_dists_no_diag = Y_dists.at[jnp.diag_indices(n)].set(0)
    # Ensure denominator is not too close to zero by adding a small constant epsilon
    epsilon = 1e-8
    normalized_Y_dists_no_diag = Y_dists_no_diag / (jnp.sum(Y_dists_no_diag) + epsilon)

    return normalized_Y_dists_no_diag, Y_dists



def momentum_func(t):
    return jax.lax.cond(t < 250, lambda _: 0.5, lambda _: 0.8, operand=None)

def setup_device_for_jax():
    if any('gpu' in device.platform.lower() for device in jax.devices()):
        jax.config.update('jax_platform_name', 'gpu')
        print('Using GPU')
        return True
    return False

def put_data_on_gpu(data):
    return jax.device_put(data, jax.devices('gpu')[0])

def initialize_embedding(P, num_dimensions, max_iterations, random_state):
    size = (P.shape[0], num_dimensions)

    key = random.PRNGKey(random_state)
    initial_vals = random.normal(key, shape=size) * jnp.sqrt(1e-4)

    embedding_matrix_container = jnp.zeros(shape=(max_iterations + 2, size[0], num_dimensions))
    # Set initial values for embedding_matrix_container at t=0 and t=1
    embedding_matrix_container = embedding_matrix_container.at[0, :, :].set(initial_vals)
    embedding_matrix_container = embedding_matrix_container.at[1, :, :].set(initial_vals)

    return embedding_matrix_container, initial_vals


def update_step(i, state):
    Y_m1, Y_m2, embedding_matrix_container, learning_rate, P = state
    Q, Y_dists = low_dim_affinities(Y_m1)  # Assuming these can be JIT compiled or are already JAX ops
    grad = compute_grad(P - Q, Y_dists, Y_m1)  # Ditto
    Y_new = Y_m1 - learning_rate * grad + momentum_func(i) * (Y_m1 - Y_m2)
    embedding_matrix_container = embedding_matrix_container.at[i, :, :].set(Y_new)
    return (Y_new, Y_m1, embedding_matrix_container, learning_rate, P)


def optimize_embeddings(max_iterations, initial_vals, learning_rate, P,
                            embedding_matrix_container):
    initial_state = (initial_vals, initial_vals, embedding_matrix_container, learning_rate, P)
    _, _, embedding_matrix_container, _, _ = jax.lax.fori_loop(2, max_iterations + 2, update_step, initial_state)
    return embedding_matrix_container


def run_embedding_process(P, num_dimensions, max_iterations, learning_rate, random_state):
    # Initialize the embedding matrix and initial values
    embedding_matrix_container, initial_vals = initialize_embedding(P, num_dimensions, max_iterations, random_state)

    # Optimize the embeddings
    Y = optimize_embeddings(max_iterations, initial_vals, learning_rate, P, embedding_matrix_container)

    return Y

def apply_scaling(affinity_matrix, scaling_factor):
    # Apply the scaling factor to the affinity matrix
    return affinity_matrix * scaling_factor


def calculate_scaled_affinities(data_mat, perplexity, perp_tol, scaling_factor, attempts=75, is_ntk=False):
    # Calculate the all symmetrical affinities
    affinity_matrix = all_sym_affinities(data_mat, perplexity, perp_tol, attempts, is_ntk)

    # Apply the scaling factor
    P = apply_scaling(affinity_matrix, scaling_factor)

    return P

def process_data_and_compute_matrix(data, is_ntk):

    # Setup device for JAX computations and move data if GPU is used
    if setup_device_for_jax():
        high_dimensional_data = put_data_on_gpu(data)
        print('Data is on GPU')

    print(f'NTK is: {is_ntk}')

    data = pca_jax(data) \
        if data.shape[1] > 30 else data

    # # Compute the data matrix based on the processed high-dimensional data and is_ntk flag
    # data_mat = compute_data_matrix(data, is_ntk)
    #
    # return data_mat
    n = data.shape[0]
    distance_matrix = jnp.zeros((n, n))
    # Determine whether to use chunking based on data size
    use_chunking = n > 200
    chunk_size = High_dimensional_chunking_limit_size if use_chunking else n  # If not using chunking, process all data at once

    # Chunking and computing distances
    for i in range(0, n, chunk_size):
        for j in range(0, n, chunk_size):
            end_i = min(i + chunk_size, n)  # Handle last chunk
            end_j = min(j + chunk_size, n)  # Handle last chunk

            chunk1 = data[i:end_i]
            chunk2 = data[j:end_j]
            distances = compute_data_matrix(chunk1, chunk2, is_ntk)

            # Update the matrix with the computed distances, accurately using end_i and end_j
            distance_matrix = distance_matrix.at[i:end_i, j:end_j].set(distances)

    return distance_matrix


def compute_low_dimensional_embedding(high_dimensional_data, num_dimensions,
                                      perplexity, max_iterations=100,
                                      learning_rate=10, scaling_factor=1.,
                                      random_state=42,
                                      perp_tol=1e-6,
                                       is_ntk = False):

    data_mat = process_data_and_compute_matrix(high_dimensional_data, is_ntk)

    # Compute pairwise affinities in high-dimensional space, scaled by a factor
    P = calculate_scaled_affinities(data_mat, perplexity, perp_tol, scaling_factor, attempts = 75, is_ntk = is_ntk)

    Y = run_embedding_process(P, num_dimensions, max_iterations, learning_rate, random_state)

    print(Y.shape)

    return Y[-1]
