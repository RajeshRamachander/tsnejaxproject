import numpy as np
from tqdm import tqdm

EPSILON = 1e-12

def compute_pairwise_distances(high_dimensional_data):
    """
    Compute pairwise distances between data points.

    Args:
        high_dimensional_data (np.ndarray): High-dimensional input data.

    Returns:
        np.ndarray: Pairwise distances matrix.
    """
    # Compute pairwise squared Euclidean distances using the NumPy approach
    X_squared = np.square(high_dimensional_data)
    sum_X = np.sum(X_squared, axis=1)
    pairwise_distances = -2 * np.dot(high_dimensional_data, high_dimensional_data.T) + sum_X[:, np.newaxis] + sum_X
    return pairwise_distances

def calculate_row_wise_entropy(asym_affinities):
    """
    Row-wise Shannon entropy of pairwise affinity matrix P

    Parameters:
    asym_affinities: pairwise affinity matrix of shape (n_samples, n_samples)

    Returns:
    array-like row-wise Shannon entropy of shape (n_samples,)
    """
    asym_affinities = np.clip(
        asym_affinities, EPSILON, None
    )  # Some are so small that log2 fails.
    return -np.sum(asym_affinities * np.log2(asym_affinities), axis=1)

def calculate_row_wise_perplexities(asym_affinities):
    """
    Compute perplexities of pairwise affinity matrix P

    Parameters:
    asym_affinities: pairwise affinity matrix of shape (n_samples, n_samples)

    Returns:
    array-like row-wise perplexities of shape (n_samples,)
    """
    return 2 ** calculate_row_wise_entropy(asym_affinities)

def pairwise_affinities(data, sigmas, dist_mat):
    """Calculates the pairwise affinities p_{j|i} using the given values of sigma

    Parameters:
    data : ndarray of shape (n_samples, n_features)
    sigmas : column array of shape (n_samples, 1)
    dist_mat : data distance matrix; ndarray of shape (n_samples, n_samples)

    Returns:
    P: pairwise affinity matrix of size (n_samples, n_samples)

    """
    assert sigmas.shape == (data.shape[0], 1)
    inner = (-dist_mat) / (2 * (sigmas ** 2))
    numers = np.exp(inner)
    denoms = np.sum(numers, axis=1) - np.diag(numers)
    denoms = denoms.reshape(-1, 1)
    denoms += EPSILON  # Avoid div/0
    P = numers / denoms
    np.fill_diagonal(P, 0.0)
    return P

def all_sym_affinities(data, perp, tol, attempts=100):
    """
    Finds the data-specific sigma values and calculates the symmetric affinities matrix P
    Parameters:
    data : ndarray of shape (n_samples, n_features)
    perp : float, cost function parameter
    tol : float, tolerance of how close the current perplexity is to the target perplexity
    attempts : int, a maximum limit to the binary search attempts

    Returns:
    P: Symmetric affinities matrix of shape (n_samples, n_samples)

    """
    dist_mat = compute_pairwise_distances(data)  # mxm

    sigma_maxs = np.full(data.shape[0], 1e12)

    # Zero here causes div/0, /2sigma**2 in P calc
    sigma_mins = np.full(data.shape[0], 1e-12)

    current_perps = np.full(data.shape[0], np.inf)

    while (not np.allclose(current_perps, perp, atol=tol)) and attempts > 0:
        sigmas = (sigma_mins + sigma_maxs) / 2
        P = pairwise_affinities(data, sigmas.reshape(-1, 1), dist_mat)
        current_perps = calculate_row_wise_perplexities(P)
        attempts -= 1
        for i in range(len(current_perps)):
            current_perp = current_perps[i]
            if current_perp > perp:
                sigma_maxs[i] = sigmas[i]
            elif current_perp < perp:
                sigma_mins[i] = sigmas[i]

    if attempts == 0:
        print(
            "Warning: Ran out of attempts before converging, try a different perplexity?"
        )
    P = (P + P.T) / (2 * data.shape[0])
    return P


def low_dim_affinities(Y, Y_dist_mat):
    """
    computes the low dimensional affinities matrix Q
    Parameters:
    Y : low dimensional representation of the data, ndarray of shape (n_samples, n_components)
    Y_dist_mat : Y distance matrix; ndarray of shape (n_samples, n_samples)

    Returns:
    Q: Symmetric low dimensional affinities matrix of shape (n_samples, n_samples)

    """
    numers = (1 + Y_dist_mat) ** (-1)
    denom = np.sum(numers) - np.sum(np.diag(numers))
    denom += EPSILON  # Avoid div/0
    Q = numers / denom
    np.fill_diagonal(Q, 0.0)
    return Q

def compute_grad(P, Q, Y, Y_dist_mat):
    """
    computes the gradient vector needed to update the Y values
    Parameters:
    P: Symmetric affinities matrix of shape (n_samples, n_samples)
    Q: Symmetric low dimensional affinities matrix of shape (n_samples, n_samples)
    Y : low dimensional representation of the data, ndarray of shape (n_samples, n_components)
    Y_dist_mat : Y distance matrix; ndarray of shape (n_samples, n_samples)

    Returns:
    grad: the gradient vector, shape (n_samples, n_components)

    """
    Ydiff = Y[:, np.newaxis, :] - Y[np.newaxis, :, :]
    pq_factor = (P - Q)[:, :, np.newaxis]
    dist_factor = ((1 + Y_dist_mat) ** (-1))[:, :, np.newaxis]
    return np.sum(4 * pq_factor * Ydiff * dist_factor, axis=1)

def momentum_func(t):
    """returns optimization parameter

    Parameters:
    t: integer, iteration number

    Returns:
    float representing the momentum term added to the gradient
    """
    if t < 250:
        return 0.5
    else:
        return 0.8



def compute_low_dimensional_embedding(high_dimensional_data, num_dimensions,
                                      target_perplexity, max_iterations=100,
                                      learning_rate=100, scaling_factor = 4.,
                                      pbar=False, random_state=None,
                                      perp_tol=1e-8):
  """
  Compute the low-dimensional embedding using t-SNE algorithm.

  Args:
      high_dimensional_data (np.ndarray): High-dimensional input data.
      num_dimensions (int): Number of dimensions for the low-dimensional embedding.
      target_perplexity (float): Target perplexity value.
      max_iterations (int): Maximum number of iterations.
      initial_momentum (float): Initial momentum value.
      final_momentum (float): Final momentum value.
      learning_rate (float): Learning rate.

  Returns:
      np.ndarray: Low-dimensional embedding.
  """
  rand = np.random.RandomState(random_state)
  P = all_sym_affinities(data=high_dimensional_data,perp=target_perplexity,tol=perp_tol) * scaling_factor
  P = np.clip(P, EPSILON, None)


  init_mean = np.zeros(num_dimensions)
  init_cov = np.identity(num_dimensions) * 1e-4

  Y = rand.multivariate_normal(mean=init_mean, cov=init_cov, size=high_dimensional_data.shape[0])

  Y_old = np.zeros_like(Y)
  iter_range = range(max_iterations)
  if pbar:
      iter_range = tqdm(iter_range, "Iterations")
  for t in iter_range:
      Y_dist_mat = compute_pairwise_distances(Y)
      Q = low_dim_affinities(Y, Y_dist_mat)
      Q = np.clip(Q, EPSILON, None)
      grad = compute_grad(P, Q, Y, Y_dist_mat)
      Y = Y - learning_rate * grad + momentum_func(t) * (Y - Y_old)
      Y_old = Y.copy()
      if t == 100:
          P = P / scaling_factor
          pass
      pass

  return Y

