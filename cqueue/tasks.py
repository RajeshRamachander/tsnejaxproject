import numpy as np
from tsnentk import compute_low_dimensional_embedding_ntk
from tsneregular import compute_low_dimensional_embedding_regular_tsne
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

@celery.task(name='tasks.tsne')
def tsne(transmit_data):
    """Processes data using t-SNE and returns the low-dimensional embedding."""
    data = np.array(transmit_data['data'])
    num_dimensions = transmit_data['num_dimensions']
    num_iterations = transmit_data['num_iterations']
    learning_rate = transmit_data['learning_rate']
    perplexity = transmit_data['perplexity']
    algorithm = transmit_data.get('algorithm', 'sklearn_tsne')  # Default to sklearn_tsne if not specified

    # Initialize low_dim_embedding to None
    low_dim_embedding = None

    if algorithm == 'ntk':
        low_dim_embedding = compute_low_dimensional_embedding_ntk(
            data,
            num_dimensions=num_dimensions,
            max_iterations=num_iterations,
            learning_rate=learning_rate,
            perplexity=perplexity,
        )
    elif algorithm == 'jax_tsne':
        low_dim_embedding = compute_low_dimensional_embedding_regular_tsne(
            data,
            num_dimensions=num_dimensions,
            max_iterations=num_iterations,
            learning_rate=learning_rate,
            perplexity=perplexity,
        )
    elif algorithm == 'sklearn_tsne':
        tsne_model = TSNE(
            n_components=num_dimensions,
            perplexity=perplexity,
            n_iter=num_iterations,
            learning_rate=learning_rate
        )
        low_dim_embedding = tsne_model.fit_transform(data)
    else:
        raise ValueError(f"Unsupported algorithm specified: {algorithm}")

    # Convert the NumPy array result to a list for JSON serialization
    low_dim_embedding_list = low_dim_embedding.tolist()

    # Log the completion of the t-SNE task
    algorithm_name = 'NTK' if algorithm == 'ntk' else 'JAX' if algorithm == 'jax_tsne' else 'sklearn'
    task_logger.info(f"t-SNE task completed using {algorithm_name} approach.")

    # Return the low-dimensional embedding as a list
    return low_dim_embedding_list


