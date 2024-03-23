
from tsne_common import run_tsne_algorithm

def compute_low_dimensional_embedding(high_dimensional_data, num_dimensions,
                                      perplexity, max_iterations=100,
                                      learning_rate=10, scaling_factor=1.,
                                      random_state=42,
                                      perp_tol=1e-6,
                                       is_ntk = False):


    Y = run_tsne_algorithm(high_dimensional_data, perplexity, perp_tol, scaling_factor,
                       num_dimensions, max_iterations,
                       learning_rate, random_state, is_ntk)

    print(Y.shape)
    return Y[-1]
