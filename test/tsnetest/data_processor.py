from abc import ABC, abstractmethod

class DataProcessor(ABC):
    def __init__(self, algorithm):
        self.algorithm = algorithm

    @abstractmethod
    def prepare_data(self):
        pass

    def wait_for_completion(self, task_id, server_communicator):
        # Implementation of the method
        pass

    @abstractmethod
    def output_data_processor(self, processed_result):
        pass
