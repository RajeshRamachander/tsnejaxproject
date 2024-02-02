import time
from abc import ABC, abstractmethod
import tsnejax as tj
from sklearn.datasets import load_digits
import numpy as np



class DataProcessorStrategy(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def process_data(self, data):
        pass

class WorkerDataProcessor(DataProcessorStrategy):

    def process_data(self, data_args):
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


class CeleryTask:
    def __init__(self, strategy):
        self.strategy = strategy

    def process_data(self, data):
        return self.strategy.process_data(data)

if __name__ == "__main__":



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

    total = CeleryTask(WorkerDataProcessor()).process_data(data_args)
    print(f"Total (Worker Strategy): {total}")

