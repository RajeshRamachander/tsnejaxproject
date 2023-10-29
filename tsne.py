import numpy as np
from scipy.spatial.distance import pdist, squareform

def calculate_entropy_and_probabilities(pairwise_distances, beta=1.0):
    """
    Calculate entropy and probabilities from pairwise distances.

    Args:
        pairwise_distances (np.ndarray): Pairwise distances matrix.
        beta (float): Beta parameter.

    Returns:
        tuple: Tuple containing entropy and probabilities.
    """
    probabilities = np.exp(-pairwise_distances * beta)
    sum_of_probabilities = np.sum(probabilities)
    entropy = np.log(sum_of_probabilities) + beta * np.sum(pairwise_distances * probabilities) / sum_of_probabilities
    probabilities /= sum_of_probabilities
    return entropy, probabilities

def compute_pairwise_probabilities(high_dimensional_data, tolerance=1e-5, target_perplexity=30.0):
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
    pairwise_distances = pdist(high_dimensional_data, "sqeuclidean")
    pairwise_distances = squareform(pairwise_distances)
    pairwise_probabilities = np.zeros((num_data_points, num_data_points))
    beta_values = np.ones((num_data_points, 1))
    log_target_perplexity = np.log(target_perplexity)

    for i in range(num_data_points):
        beta_min = -np.inf
        beta_max = np.inf
        distances_i = pairwise_distances[i, np.concatenate((np.r_[0:i], np.r_[i + 1:num_data_points]))]
        entropy, this_probabilities = calculate_entropy_and_probabilities(distances_i, beta_values[i])
        entropy_difference = entropy - log_target_perplexity
        num_tries = 0

        while np.abs(entropy_difference) > tolerance and num_tries < 50:
            if entropy_difference > 0:
                beta_min = beta_values[i].copy()
                if beta_max == np.inf or beta_max == -np.inf:
                    beta_values[i] = beta_values[i] * 2.
                else:
                    beta_values[i] = (beta_values[i] + beta_max) / 2.
            else:
                beta_max = beta_values[i].copy()
                if beta_min == np.inf or beta_min == -np.inf:
                    beta_values[i] = beta_values[i] / 2.
                else:
                    beta_values[i] = (beta_values[i] + beta_min) / 2.
            entropy, this_probabilities = calculate_entropy_and_probabilities(distances_i, beta_values[i])
            entropy_difference = entropy - log_target_perplexity
            num_tries += 1

        pairwise_probabilities[i, np.concatenate((np.r_[0:i], np.r_[i + 1:num_data_points]))] = this_probabilities

    return pairwise_probabilities

def t_sne_dimensionality_reduction(high_dimensional_data, num_dimensions=2, initial_dimensions=50, target_perplexity=30.0):
    """
    Perform t-SNE dimensionality reduction.

    Args:
        high_dimensional_data (np.ndarray): High-dimensional input data.
        num_dimensions (int): Number of dimensions for the low-dimensional embedding.
        initial_dimensions (int): Number of dimensions to initialize the embedding.
        target_perplexity (float): Target perplexity value.

    Returns:
        np.ndarray: Low-dimensional embedding.
    """
    num_data_points = high_dimensional_data.shape[0]
    initial_momentum = 0.5
    final_momentum = 0.8
    learning_rate = 500
    max_iterations = 1000

    high_dimensional_data -= np.mean(high_dimensional_data, axis=0)
    eigenvalues = np.linalg.eigvals(np.dot(high_dimensional_data.T, high_dimensional_data))
    high_dimensional_data /= np.sqrt(eigenvalues.max())

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
        sum_of_squared_low_dimensional_embedding = np.sum(np.square(low_dimensional_embedding), 1)
        num = -2. * np.dot(low_dimensional_embedding, low_dimensional_embedding.T)
        num = 1. / (1. + np.add(np.add(num, sum_of_squared_low_dimensional_embedding).T,
                                sum_of_squared_low_dimensional_embedding))
        num[range(num_data_points), range(num_data_points)] = 0.
        pairwise_similarity_q = num / np.sum(num)
        pairwise_similarity_q = np.maximum(pairwise_similarity_q, 1e-12)

        pairwise_similarity_pq = pairwise_probabilities - pairwise_similarity_q
        for i in range(num_data_points):
            gradient[i, :] = np.sum(np.tile(pairwise_similarity_pq[:, i] * num[:, i], (num_dimensions, 1)).T * (
                    low_dimensional_embedding[i, :] - low_dimensional_embedding), 0)

        if iteration < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum

        gains = (gains + 0.2) * ((gradient > 0.) != (previous_gradient > 0.)) + (gains * 0.8) * (
                (gradient > 0.) == (previous_gradient > 0.))
        gains[gains < 0.01] = 0.01
        previous_gradient = momentum * previous_gradient - learning_rate * (gains * gradient)
        low_dimensional_embedding += previous_gradient

        low_dimensional_embedding -= np.tile(np.mean(low_dimensional_embedding, 0), (num_data_points, 1))

        if (iteration + 1) % 100 == 0:
            KL_divergence = np.sum(pairwise_probabilities * np.log(pairwise_probabilities / pairwise_similarity_q))
            print("Iteration %d: KL Divergence is %f" % (iteration + 1, KL_divergence))

    return low_dimensional_embedding

# Example usage:
# result = t_sne_dimensionality_reduction(high_dimensional_data, num_dimensions=2)
