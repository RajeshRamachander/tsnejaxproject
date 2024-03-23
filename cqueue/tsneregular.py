from tsne_common import run_tsne_algorithm


def compute_low_dimensional_embedding_regular_tsne(high_dimensional_data, num_dimensions,
                                      perplexity, max_iterations=100,
                                      learning_rate=100, scaling_factor=4.,
                                      random_state=42,
                                      perp_tol=.1):

    Y = run_tsne_algorithm(high_dimensional_data, perplexity, perp_tol, scaling_factor,
                           num_dimensions, max_iterations,
                           learning_rate, random_state)

    print(Y.shape)
    return Y[-1]

