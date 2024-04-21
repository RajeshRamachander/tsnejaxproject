
import jax
from jax import jit
from jax.nn import softmax
from jax.numpy.linalg import svd
import jax.numpy as jnp
from jax import random
from ntk import compute_ntk_matrix
from jax.experimental import host_callback as hcb


Low_dimensional_chunking_limit_size = 200
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

@jit
def get_eculidean_probability_at_ij(d, scale, i):
    d_scaled = -d / scale
    d_scaled -= jnp.max(d_scaled)
    exp_D = jnp.exp(d_scaled)
    exp_D = exp_D.at[i].set(0)
    epsilon = 1e-8  #to give numerical stability and avoid division by zero
    return exp_D / (jnp.sum(exp_D) + epsilon)

@jit
def get_ntk_probability_at_ij(d, sigma, i):
    # No negation for NTK values, as larger values indicate stronger influence.
    d_scaled = d / (sigma)
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


def perp_current_value(value):
  print(f"current_perp: {value}")

def perp_target_value(value):
  print(f"target_perp: {value}")


def perp_attempts(value):
  print(f"attempts: {value}")

def perp_condition(value):
  print(f"condition: {value}")

def perp_tol(value):
    print(f"tol: {value}")

@jit
def calculate_scaled_affinities_jit(data_mat, target_perp=30, tol = 1e-6,
                                sigma_low = 0.0, sigma_high = 1e4,
                                max_attempts=250, is_ntk=True):
    n = data_mat.shape[0]
    P = jnp.zeros(data_mat.shape)

    def body_fun_outer(i, P):
        LB_i = sigma_low
        UB_i = sigma_high
        d = data_mat[i, :]

        def cond_fun_inner(val):
            _, _, t, _ = val
            # hcb.call(perp_attempts, iterations)
            return t <= max_attempts

        def body_fun_inner(val):
            LB_i, UB_i, t, _ = val
            midpoint = (LB_i + UB_i) / 2
            scale = 2 * midpoint ** 2
            p_ij = get_probability_at_ij(d, scale, i, is_ntk)
            current_perp = get_shannon_entropy(p_ij)

            sigma_update_cond = current_perp < target_perp

            LB_i = jax.lax.cond(sigma_update_cond, lambda : midpoint, lambda : LB_i)
            UB_i = jax.lax.cond(sigma_update_cond, lambda : UB_i, lambda : midpoint)

            perp_cond_reached = jnp.isclose(current_perp, target_perp, atol=tol)

            hcb.call(perp_current_value, current_perp)
            hcb.call(perp_condition, perp_cond_reached)
            hcb.call(perp_attempts, t)
            t = jax.lax.cond(perp_cond_reached, lambda: max_attempts, lambda: t + 1  )

            return LB_i, UB_i, t, p_ij

        _, _, t, p_ij = jax.lax.while_loop(cond_fun_inner,body_fun_inner,(LB_i, UB_i, 0, P[i,:]))

        P = P.at[i, :].set(p_ij)
        return P

    P = jax.lax.fori_loop(0, n, body_fun_outer, P)
    return (P + P.T) / (2 * n)

def calculate_scaled_affinities_py(data_mat, target_perp=30, tol = 1e-6,
                                sigma_low = 0.0, sigma_high = 1e4,
                                max_attempts=250, is_ntk=True):
    n = data_mat.shape[0]
    P = jnp.zeros(data_mat.shape)

    for i in range(n):
        LB_i = sigma_low
        UB_i = sigma_high
        d = data_mat[i, :]

        for t in range(max_attempts):
            # Find the perplexity using sigma = midpoint.
            midpoint = (LB_i + UB_i) / 2
            scale = 2 * midpoint ** 2
            p_ij = get_probability_at_ij(d, scale, i, is_ntk)
            current_perp = get_shannon_entropy(p_ij)

            if current_perp < target_perp:
                LB_i = midpoint
            else:
                UB_i = midpoint


            perp_cond_reached = jnp.isclose(current_perp, target_perp, atol=tol)

            # hcb.call(perp_current_value,current_perp)
            # hcb.call(perp_condition, perp_cond_reached)
            # hcb.call(perp_attempts, t)

            if perp_cond_reached:
                break

        P = P.at[i, :].set(p_ij)

    return (P + P.T) / (2 * n)
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

def compute_grad_py(R, Y_dists, Y):

    dY = jnp.zeros_like(Y)

    for i in range(Y.shape[0]):
        grad_contribution = 4*jnp.dot(R[i,:]*Y_dists[i,:], Y[i,:] - Y)
        dY = dY.at[i,:].set(grad_contribution)

    return dY


@jit
def compute_data_matrix(chunk1, chunk2, is_ntk):
    # def compute_data_matrix(high_dimensional_data, is_ntk):
    def true_fun(_):
        return compute_ntk_matrix(chunk1, chunk2)

    def false_fun(_):
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
                # Determine the end indices for the chunk
                end_i = min(i + chunk_size, n)
                end_j = min(j + chunk_size, n)
                # Extract chunks
                chunk1 = Y[i:end_i]
                chunk2 = Y[j:end_j]

                # Compute distances for the current chunks
                distances_chunk = compute_pairwise_distances(chunk1, chunk2)

                # Update the corresponding part of the full distance matrix
                # Note: JAX arrays are immutable, so we need to use the .at[].set method for updates
                full_distance_matrix = full_distance_matrix.at[i:end_i, j:end_j].set(distances_chunk)

        return full_distance_matrix
    else:
        return compute_pairwise_distances(Y, Y)


def low_dim_affinities(Y):
    D = chunk_compute_assemble_distances(Y)
    Y_dists = jnp.power(1 + D , -1)
    # overall, Y_dists_no_diag will be the matrix Y_dists with all its diagonal elements set to zero.
    n = Y_dists.shape[0]
    Y_dists_no_diag = Y_dists.at[jnp.diag_indices(n)].set(0)
    # Ensure denominator is not too close to zero by adding a small constant epsilon
    # epsilon = 1e-8
    normalized_Y_dists_no_diag = Y_dists_no_diag / (jnp.sum(Y_dists_no_diag)) # + epsilon)

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
    Y_m2, Y_m1, embedding_matrix_container, learning_rate, P = state
    Q, Y_dists = low_dim_affinities(Y_m1)
    grad = compute_grad_py(P - Q, Y_dists, Y_m1)
    Y_new = Y_m1 - learning_rate * grad + momentum_func(i) * (Y_m1 - Y_m2)
    embedding_matrix_container = embedding_matrix_container.at[i, :, :].set(Y_new)
    return (Y_m1, Y_new, embedding_matrix_container, learning_rate, P)


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

# Define a function to normalize the data using Min-Max scaling
def min_max_scaling(data):
    min_val = jnp.min(data, axis=0)
    max_val = jnp.max(data, axis=0)
    scaled_data = (data - min_val) / (max_val - min_val)
    return scaled_data

# Define a function to normalize the data using Z-score normalization
def z_score_normalization(data):
    mean = jnp.mean(data, axis=0)
    std_dev = jnp.std(data, axis=0)
    normalized_data = (data - mean) / std_dev
    return normalized_data

def normalization(image_data):
    norm = jnp.linalg.norm(image_data)
    normalized_data = image_data / norm
    return normalized_data

def process_data_and_compute_matrix(data, is_ntk):

    # Setup device for JAX computations and move data if GPU is used
    if setup_device_for_jax():
        high_dimensional_data = put_data_on_gpu(data)
        print('Data is on GPU')
    else:
        print('Data is on CPU')

    # Update the global configuration to use float64 precision
    jax.config.update("jax_enable_x64",True)

    print(f'NTK is: {is_ntk}')

    data = pca_jax(data) \
        if data.shape[1] > 30 else data

    data = normalization(data)

    n = data.shape[0]
    distance_matrix = jnp.zeros((n, n))
    # Determine whether to use chunking based on data size
    use_chunking = n > High_dimensional_chunking_limit_size
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
                                      learning_rate=10, scaling_factor=2.,
                                      random_state=42,
                                      perp_tol=1e-6,
                                       is_ntk = False):

    data_mat = process_data_and_compute_matrix(high_dimensional_data, is_ntk)

    # Compute pairwise affinities in high-dimensional space, scaled by a factor
    P = scaling_factor * calculate_scaled_affinities_py(data_mat, target_perp = perplexity,
                                                     tol = perp_tol, sigma_low=0.0, sigma_high=1e4,
                                                     max_attempts= 250, is_ntk = is_ntk)

    Y = run_embedding_process(P, num_dimensions, max_iterations, learning_rate, random_state)

    print(Y.shape)

    # Check the current floating-point precision setting
    if jax.config.jax_enable_x64:
        print("JAX is using 64-bit floating-point precision (float64).")
    else:
        print("JAX is using 32-bit floating-point precision (float32).")


    return Y[-1]
