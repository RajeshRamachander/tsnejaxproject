import numpy as np
from scipy.spatial.distance import pdist, squareform

import matplotlib.pyplot as plt

def calculate_entropy_and_probabilities(pairwise_distances, beta=1.0):
    """
    Calculate entropy and probabilities from pairwise distances.

    Args:
        pairwise_distances (np.ndarray): Pairwise distances matrix.
        beta (float, optional): Beta parameter.

    Returns:
        tuple: Tuple containing entropy and probabilities.
    """
    probabilities = np.exp(-pairwise_distances * beta)
    sum_of_probabilities = np.sum(probabilities)
    entropy = np.log(sum_of_probabilities) + beta * np.sum(pairwise_distances * probabilities) / sum_of_probabilities
    probabilities /= sum_of_probabilities
    return entropy, probabilities


def initialize_beta_values(num_data_points):
  """
  Initialize beta values for each data point.

  Args:
      num_data_points (int): Number of data points.

  Returns:
      np.ndarray: Initialized beta values.
  """
  return np.ones((num_data_points, 1))


def update_beta(beta, entropy_difference, beta_min, beta_max):
  """
  Update the beta value based on entropy difference.

  Args:
      beta (float): Current beta value.
      entropy_difference (float): Current entropy difference.
      beta_min (float): Minimum beta value.
      beta_max (float): Maximum beta value.

  Returns:
      float: Updated beta value.
  """
  if entropy_difference > 0:
    beta_min = beta
    if beta_max == np.inf or beta_max == -np.inf:
      beta = beta * 2.
    else:
      beta = (beta + beta_max) / 2.
  else:
    beta_max = beta
    if beta_min == np.inf or beta_min == -np.inf:
      beta = beta / 2.
    else:
      beta = (beta + beta_min) / 2.
  return beta, beta_min, beta_max


def compute_pairwise_probabilities(high_dimensional_data, tolerance=1e-5,
                                   target_perplexity=30.0):
  """
  Compute pairwise probabilities using t-SNE algorithm.

  Args:
      high_dimensional_data (np.ndarray): High-dimensional input data.
      tolerance (float): Tolerance value for optimization.
      target_perplexity (float): Target perplexity value.

  Returns:
      np.ndarray: Pairwise probabilities matrix.
  """
  num_data_points = high_dimensional_data.shape[0]
  pairwise_distances = compute_pairwise_distances(high_dimensional_data)
  pairwise_probabilities = np.zeros((num_data_points, num_data_points))
  beta_values = initialize_beta_values(num_data_points)
  log_target_perplexity = np.log(target_perplexity)

  for i in range(num_data_points):
    beta_min = -np.inf
    beta_max = np.inf
    distances_i = pairwise_distances[i, compute_neighboring_indices(i, num_data_points)]
    entropy, this_probabilities = calculate_entropy_and_probabilities(distances_i, beta_values[i])
    entropy_difference = entropy - log_target_perplexity
    num_tries = 0
    while np.abs(entropy_difference) > tolerance and num_tries < 50:
      beta_values[i], beta_min, beta_max = update_beta(beta_values[i],entropy_difference,beta_min, beta_max)
      entropy, this_probabilities = calculate_entropy_and_probabilities(distances_i, beta_values[i])
      entropy_difference = entropy - log_target_perplexity
      num_tries += 1

    pairwise_probabilities[i, compute_neighboring_indices(i, num_data_points)] = this_probabilities

  return pairwise_probabilities


def compute_pairwise_distances(high_dimensional_data):
  """
  Compute pairwise distances between data points.

  Args:
      high_dimensional_data (np.ndarray): High-dimensional input data.

  Returns:
      np.ndarray: Pairwise distances matrix.
  """
  pairwise_distances = pdist(high_dimensional_data, "sqeuclidean")
  return squareform(pairwise_distances)


def compute_neighboring_indices(i, num_data_points):
  """
  Compute indices of neighboring data points.

  Args:
      i (int): Index of the current data point.
      num_data_points (int): Number of data points.

  Returns:
      np.ndarray: Indices of neighboring data points.
  """
  return np.concatenate((np.r_[0:i], np.r_[i + 1:num_data_points]))


def initialize_low_dimensional_embedding(num_data_points, num_dimensions):
    """
    Initialize the low-dimensional embedding.

    Args:
        num_data_points (int): Number of data points.
        num_dimensions (int): Number of dimensions for the low-dimensional embedding.

    Returns:
        np.ndarray: Initialized low-dimensional embedding.
    """
    return np.random.randn(num_data_points, num_dimensions)

def compute_initial_momentum(iteration, initial_momentum, final_momentum):
    """
    Compute the momentum for the current iteration.

    Args:
        iteration (int): Current iteration number.
        initial_momentum (float): Initial momentum value.
        final_momentum (float): Final momentum value.

    Returns:
        float: Computed momentum.
    """
    return initial_momentum if iteration < 20 else final_momentum

def update_gains(gains, gradient, previous_gradient):
    """
    Update the gains for gradient adjustments.

    Args:
        gains (np.ndarray): Current gains.
        gradient (np.ndarray): Gradient values.
        previous_gradient (np.ndarray): Previous gradient values.

    Returns:
        np.ndarray: Updated gains.
    """
    gains = (gains + 0.2) * ((gradient > 0.) != (previous_gradient > 0.)) + (gains * 0.8) * (
                (gradient > 0.) == (previous_gradient > 0.))
    gains[gains < 0.01] = 0.01
    return gains


def compute_low_dimensional_embedding(high_dimensional_data, num_dimensions,
                                      target_perplexity, max_iterations=1000,
                                      initial_momentum=0.5, final_momentum=0.8,
                                      learning_rate=500):
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
  num_data_points = high_dimensional_data.shape[0]

  high_dimensional_data -= np.mean(high_dimensional_data, axis=0)

  _, S, _ = np.linalg.svd(high_dimensional_data, full_matrices=False)
  high_dimensional_data /= np.sqrt(S[0])


  low_dimensional_embedding = np.random.randn(num_data_points, num_dimensions)
  gradient = np.zeros((num_data_points, num_dimensions))
  previous_gradient = np.zeros((num_data_points, num_dimensions))
  gains = np.ones((num_data_points, num_dimensions))

  pairwise_probabilities = compute_pairwise_probabilities(high_dimensional_data, 1e-5, target_perplexity)
  pairwise_probabilities += np.transpose(pairwise_probabilities)
  pairwise_probabilities /= np.sum(pairwise_probabilities)
  pairwise_probabilities *= 4.
  pairwise_probabilities = np.maximum(pairwise_probabilities, 1e-12)

  for iteration in range(max_iterations):
    sum_of_squared_low_dimensional_embedding = np.sum(
      np.square(low_dimensional_embedding), 1)
    num = -2. * np.dot(low_dimensional_embedding, low_dimensional_embedding.T)
    num = 1. / (
        1. + np.add(np.add(num, sum_of_squared_low_dimensional_embedding).T,
                    sum_of_squared_low_dimensional_embedding))
    num[range(num_data_points), range(num_data_points)] = 0.

    pairwise_similarity_q = num / np.sum(num)
    pairwise_similarity_q = np.maximum(pairwise_similarity_q, 1e-12)

    KL_divergence = np.sum(pairwise_probabilities * np.log(
      pairwise_probabilities / pairwise_similarity_q))
    if (iteration + 1) % 100 == 0:
      print(
        "Iteration %d: KL Divergence is %f" % (iteration + 1, KL_divergence))

    pairwise_similarity_pq = pairwise_probabilities - pairwise_similarity_q
    for i in range(num_data_points):
      gradient[i, :] = np.sum(
        np.tile(pairwise_similarity_pq[:, i] * num[:, i],
                (num_dimensions, 1)).T *
        (low_dimensional_embedding[i, :] - low_dimensional_embedding), 0)

    momentum = 0.5 if iteration < 20 else 0.8
    gains = (gains + 0.2) * ((gradient > 0.) != (previous_gradient > 0.)) + (
        gains * 0.8) * (
              (gradient > 0.) == (previous_gradient > 0.))
    gains[gains < 0.01] = 0.01
    previous_gradient = momentum * previous_gradient - learning_rate * (
        gains * gradient)
    low_dimensional_embedding += previous_gradient

    low_dimensional_embedding -= np.tile(np.mean(low_dimensional_embedding, 0),
                                         (num_data_points, 1))

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
