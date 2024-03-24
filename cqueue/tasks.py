import numpy as np
from tsne_common import (
    compute_low_dimensional_embedding,
    process_data_and_compute_matrix,
)
from sklearn.manifold import TSNE


from celery import Celery
import logging

def configure_logger(logger_name, log_file):
    """Configures a logger with a FileHandler and formatter."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]")
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger

# Configure Flask's app logger
task_logger = configure_logger("task", "task.log")

celery = Celery(
    'tasks',  # Use the app name from your Flask app
    broker="redis://127.0.0.1:6379/0",
    backend="redis://127.0.0.1:6379/0"
)

def compute_ntk_jax_tsne(data, num_dimensions, num_iterations, learning_rate, perplexity, is_ntk):
    """Compute low-dimensional embeddings using NTK or JAX t-SNE."""
    return compute_low_dimensional_embedding(
        data,
        num_dimensions=num_dimensions,
        max_iterations=num_iterations,
        learning_rate=learning_rate,
        perplexity=perplexity,
        is_ntk=is_ntk
    )

def compute_sklearn_tsne(data, num_dimensions, num_iterations, learning_rate, perplexity):
    """Compute low-dimensional embeddings using sklearn's t-SNE."""
    tsne_model = TSNE(
        n_components=num_dimensions,
        perplexity=perplexity,
        n_iter=num_iterations,
        learning_rate=learning_rate
    )
    return tsne_model.fit_transform(data)

def compute_tsne(transmit_data):

    data = np.array(transmit_data['data'])
    num_dimensions = transmit_data['num_dimensions']
    num_iterations = transmit_data['num_iterations']
    learning_rate = transmit_data['learning_rate']
    perplexity = transmit_data['perplexity']
    algorithm = transmit_data.get('algorithm', 'sklearn_tsne')  # Default to sklearn_tsne if not specified

    if algorithm in ['ntk', 'jax_tsne']:
        low_dim_embedding = compute_ntk_jax_tsne(
            data, num_dimensions, num_iterations, learning_rate, perplexity, is_ntk=(algorithm == 'ntk'))
    elif algorithm == 'sklearn_tsne':
        low_dim_embedding = compute_sklearn_tsne(data, num_dimensions, num_iterations, learning_rate, perplexity)
    else:
        raise ValueError(f"Unsupported algorithm specified: {algorithm}")

    # Convert the NumPy array result to a list for JSON serialization
    low_dim_embedding_list = low_dim_embedding.tolist()

    # Log the completion of the t-SNE task
    algorithm_name = 'NTK' if algorithm == 'ntk' else 'JAX' if algorithm == 'jax_tsne' else 'sklearn'
    task_logger.info(f"t-SNE task completed using {algorithm_name} approach.")

    return low_dim_embedding_list

def compute_matrix(transmit_data):
    data = np.array(transmit_data['data'])
    algorithm = transmit_data.get('algorithm', 'ntk')  # Default to ntk if not specified

    algorithm_functions = {
        'ntk': lambda data: process_data_and_compute_matrix(data, is_ntk=True),
        'jax_tsne': lambda data: process_data_and_compute_matrix(data, is_ntk=False)
    }

    process_function = algorithm_functions.get(algorithm)
    if process_function:
        return process_function(data).tolist()
    else:
        raise ValueError(f"Unsupported algorithm specified: {algorithm}")



# Define the strategy map outside of the function to avoid redefining it on each call
PACKER_STRATEGY_MAP = {
    "full": compute_tsne,
    "matrix": compute_matrix,
    # "some_other_value": some_other_function,  # Example for future extension
}

def handle_packer_action(packer_value, transmit_data):
    """Determines and executes the action based on the packer value using a strategy map."""
    action = PACKER_STRATEGY_MAP.get(packer_value)

    if action:
        return action(transmit_data)
    else:
        task_logger.info(f"No specific action defined for packer value '{packer_value}'. Skipping processing.")
        return None

@celery.task(name='tasks.tsne')
def tsne(transmit_data):
    """Processes data using t-SNE based on the 'packer' value."""
    packer = transmit_data.get('packer', '')
    result = handle_packer_action(packer, transmit_data)

    if result is None:
        # If no action was taken, you might want to return a default value or perform some other operation
        return []

    return result




