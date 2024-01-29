import time
from abc import ABC, abstractmethod
import tsnejax as tj
from sklearn.datasets import load_digits
import numpy as np

# In some other module
import logging

logger = logging.getLogger('worker')
logger.setLevel(logging.INFO)
handler = logging.FileHandler('worker.log')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def some_function():
    logger.info("This is an info message")



class DataProcessorStrategy(ABC):
    @abstractmethod
    def process_data(self, data):
        pass

class LoggingDataProcessor(DataProcessorStrategy):
    def __init__(self, logger,worker):
        self.logger = logger
        self.worker = worker

    def process_data(self, data):
        total = 0
        for number in data:
            self.logger.info(f"Processing number: {number}")
            time.sleep(1)
            total += number
        return total

class WorkerDataProcessor(DataProcessorStrategy):
    def __init__(self, logger,worker):
        self.worker = worker
        self.logger = logger

    def process_data(self, data):
        self.logger.info(f"Type of low_dim: {type(data)}")
        self.logger.info(f"Low_dim data (partial view): {data[:10]}")

        if isinstance(data, list):
            data = np.array(data)

        digits, digit_class = load_digits(return_X_y=True)
        rand_idx = np.random.choice(np.arange(digits.shape[0]), size=500, replace=False)
        data1 = digits[rand_idx, :].copy()

        low_dim = tj.compute_low_dimensional_embedding(data, 2, 30, 500, \
                                                       100, pbar=True, use_ntk=False)

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
    logger_strategy = LoggingDataProcessor(celery_log)
    worker_strategy = WorkerDataProcessor(worker)

    # Example usage:
    task_logger = CeleryTask(logger_strategy)
    task_worker = CeleryTask(worker_strategy)

    data = [1, 2, 3, 4, 5]
    total1 = task_logger.process_data(data)
    total2 = task_worker.process_data(data)

    print(f"Total (Logger Strategy): {total1}")
    print(f"Total (Worker Strategy): {total2}")
