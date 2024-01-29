import logging
import time
from abc import ABC, abstractmethod

class DataProcessorStrategy(ABC):
    @abstractmethod
    def process_data(self, data):
        pass

class LoggingDataProcessor(DataProcessorStrategy):
    def __init__(self, logger):
        self.logger = logger

    def process_data(self, data):
        total = 0
        for number in data:
            self.logger.info(f"Processing number: {number}")
            time.sleep(1)
            total += number
        return total

class WorkerDataProcessor(DataProcessorStrategy):
    def __init__(self, worker):
        self.worker = worker

    def process_data(self, data):

        return data

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
