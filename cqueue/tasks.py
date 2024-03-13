import numpy as np
from tsnentk import compute_low_dimensional_embedding_ntk
from tsneregular import compute_low_dimensional_embedding_regular_tsne

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
def tsne(data_args):
    """Processes data using t-SNE and returns the low-dimensional embedding."""
    data_array = np.array(data_args['data'])
    num_dimensions = data_args['num_dimensions']
    num_iterations = data_args['num_iterations']
    learning_rate = data_args['learning_rate']
    perplexity = data_args['perplexity']
    use_ntk = data_args['use_ntk']

    # Determine the function to use based on `use_ntk`
    compute_embedding = compute_low_dimensional_embedding_ntk if use_ntk else compute_low_dimensional_embedding_regular_tsne

    # Compute the low-dimensional embedding
    low_dim_embedding = compute_embedding(
        data_array,
        num_dimensions=num_dimensions,
        max_iterations=num_iterations,
        learning_rate=learning_rate,
        perplexity=perplexity,
    )

    # Convert the NumPy array result to a list for JSON serialization
    low_dim_embedding_list = low_dim_embedding.tolist()

    # Log the completion of the t-SNE task
    task_logger.info(f"t-SNE task completed using {'NTK approach' if use_ntk else 'regular approach'}.")

    # Return the low-dimensional embedding as a list
    return low_dim_embedding_list


