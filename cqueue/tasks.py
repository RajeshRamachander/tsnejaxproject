import numpy as np
from tsnentk import compute_low_dimensional_embedding_ntk

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
    perplexity = data_args['perplexity']
    num_iterations = data_args['num_iterations']
    learning_rate = data_args['learning_rate']
    pbar = data_args['pbar']
    use_ntk = data_args['use_ntk']

    low_dim = compute_low_dimensional_embedding_ntk(
        data_array,
        num_dimensions=num_dimensions,
        target_perplexity=perplexity,
        max_iterations=num_iterations,
        learning_rate=learning_rate,
        pbar=pbar,
    )

    # Convert to list for serialization
    low_dim_list = low_dim.tolist()

    task_logger.info("t-SNE task completed.")

    return low_dim_list
