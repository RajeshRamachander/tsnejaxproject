from abc import ABC, abstractmethod
import tsnejax as tj
from sklearn.datasets import load_digits
import numpy as np

import logging

# Set up logging for the background worker
worker_log = logging.getLogger('worker')
worker_log.setLevel(logging.INFO)

# Create a file handler for worker logs (use celery.log)
worker_file_handler = logging.FileHandler('celery.log')
worker_file_handler.setLevel(logging.INFO)

# Create a formatter for the worker log messages
worker_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
worker_file_handler.setFormatter(worker_formatter)

# Add the file handler to the worker logger
worker_log.addHandler(worker_file_handler)


class DataProcessorStrategy(ABC):
    def __init__(self, worker):
        self.worker = worker

    @abstractmethod
    def process_data(self, data):
        pass

class WorkerDataProcessor(DataProcessorStrategy):

    def process_data(self, data):
        data_array = np.array(data['data'])
        num_dimensions = data['num_dimensions']
        perplexity = data['perplexity']
        num_iterations = data['num_iterations']
        learning_rate = data['learning_rate']
        pbar = data['pbar']
        use_ntk = data['use_ntk']

        low_dim = tj.compute_low_dimensional_embedding(
            data_array, num_dimensions, perplexity, num_iterations,
            learning_rate, pbar=pbar, use_ntk=use_ntk
        )

        # Convert to list for serialization
        low_dim_list = low_dim.tolist()

        return low_dim_list  # Return the serializable list


class CeleryTask:
    def __init__(self, strategy):
        self.strategy = strategy

    def process_data(self, data):
        return self.strategy.process_data(data)

if __name__ == "__main__":
    # Initialize logger and worker
    celery_log = logging.getLogger('celery')
    celery_log.setLevel(logging.INFO)
    file_handler = logging.FileHandler('celery.log')
    file_handler.setLevel(logging.INFO)
    celery_log.addHandler(file_handler)

    worker_strategy = WorkerDataProcessor(worker_log)

    task_worker = CeleryTask(worker_strategy)

    digits, digit_class = load_digits(return_X_y=True)
    rand_idx = np.random.choice(np.arange(digits.shape[0]), size=500, replace=False)
    data = digits[rand_idx, :]

    data_args = {
        'data': data.tolist(),
        'num_dimensions': 2,
        'perplexity': 30,
        'num_iterations': 500,
        'learning_rate': 100,
        'batch_size': 100,
        'pbar': True,
        'use_ntk': False
    }

    total2 = task_worker.process_data(data_args)

    print(f"Total (Worker Strategy): {total2}")

