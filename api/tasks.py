from celery import shared_task
import numpy as np
from api.tsnejax import compute_low_dimensional_embedding  # Import only the needed function

from celery import Celery

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

    low_dim = compute_low_dimensional_embedding(
        data_array,
        num_dimensions=num_dimensions,
        target_perplexity=perplexity,
        max_iterations=num_iterations,
        learning_rate=learning_rate,
        pbar=pbar,
        use_ntk=use_ntk
    )

    # Convert to list for serialization
    low_dim_list = low_dim.tolist()

    return low_dim_list
