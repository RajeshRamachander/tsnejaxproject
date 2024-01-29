import time
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
    def __init__(self, logger, worker):
        self.logger = logger
        self.worker = worker

    @abstractmethod
    def process_data(self, data):
        pass

class LoggingDataProcessor(DataProcessorStrategy):

    def process_data(self, data):
        total = 0
        for number in data:
            self.logger.info(f"Processing number: {number}")
            time.sleep(1)
            total += number
        return total

class WorkerDataProcessor(DataProcessorStrategy):

    def process_data(self, data):
        data_array = np.array(data_args['data'])
        num_dimensions = data_args['num_dimensions']
        perplexity = data_args['perplexity']
        num_iterations = data_args['num_iterations']
        learning_rate = data_args['learning_rate']
        batch_size = data_args['batch_size']
        pbar = data_args['pbar']
        use_ntk = data_args['use_ntk']

        low_dim = tj.compute_low_dimensional_embedding(
            data_array, num_dimensions, perplexity, num_iterations,
            learning_rate, pbar=pbar, use_ntk=use_ntk
        )

        # Convert to list for serialization
        low_dim_list = low_dim.tolist()

        return low_dim_list  # Return the serializable list

# Example usage:
class Worker:
    def add(self, number):
        # Simulate some worker operation
        return number * 2

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

    worker = Worker()
    logger_strategy = LoggingDataProcessor(worker_log, worker)
    worker_strategy = WorkerDataProcessor(worker_log, worker)

    # Example usage:
    task_logger = CeleryTask(logger_strategy)
    task_worker = CeleryTask(worker_strategy)

    data = [1, 2, 3, 4, 5]
    total1 = task_logger.process_data(data)

    digits, digit_class = load_digits(return_X_y=True)
    rand_idx = np.random.choice(np.arange(digits.shape[0]), size=500, replace=False)
    data = digits[rand_idx, :]

    data_args = {
        'data': data,
        'num_dimensions': 2,
        'perplexity': 30,
        'num_iterations': 500,
        'learning_rate': 100,
        'batch_size': 100,
        'pbar': True,
        'use_ntk': False
    }

    total2 = task_worker.process_data(data_args)

    print(f"Total (Logger Strategy): {total1}")
    print(f"Total (Worker Strategy): {total2}")

