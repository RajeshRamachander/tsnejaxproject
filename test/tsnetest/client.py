import time
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.datasets import load_digits
import numpy as np
import ast
import requests
# from tensorflow.keras.datasets import cifar10  # Import CIFAR-10 dataset from Keras
from tensorflow.keras.datasets import mnist



BASE_URL = 'http://127.0.0.1:7020'

SIZE = 1000

class ServerCommunicator:

    def start_task(self, data):
        response = requests.post(f'{BASE_URL}/start-task', json=data)
        return response

    def check_task_status(self, task_id):
        status_response = requests.get(f'{BASE_URL}/task-status/{task_id}')
        return status_response.json()


class DataProcessor(ABC):

    def __init__(self):
        self.algorithm = None  # Add an attribute to store the algorithm type

    @abstractmethod
    def prepare_data(self):
        pass


    def process_result(self, result):
        # Add your result processing logic here
        processed_data = f'Result received: {result}'
        return processed_data

    def wait_for_completion(self, task_id, server_communicator):
        while True:
            status_data = server_communicator.check_task_status(task_id)
            status = status_data['status']

            if status == 'success':
                result = status_data['result']
                return self.process_result(result)
            elif status == 'processing':
                print('Task is still processing. Waiting...')
                time.sleep(1)
            else:
                print(f'Unknown status: {status}')
                break

    def output_data_processor(self, processed_result):
    
        if processed_result is not None:

            # Remove the non-literal part of the string
            result_received = processed_result.replace("Result received: ", "")

            # Now convert the string back to a Python object (list of lists)
            low_dim = ast.literal_eval(result_received)

            low_dim = np.array(low_dim)


            rcParams["font.size"] = 18
            rcParams["figure.figsize"] = (12, 8)

            print(f"Using {self.algorithm} for t-SNE.")  # Print the algorithm type

            print(low_dim)

            scatter = plt.scatter(low_dim[:, 0], low_dim[:, 1], cmap="tab10", c=self.classes)
            plt.legend(*scatter.legend_elements(), fancybox=True, bbox_to_anchor=(1.05, 1))
            plt.show()



class SimpleDataProcessor(DataProcessor):
    def __init__(self, algorithm):
        super().__init__()  # Initialize the base class
        self.classes = None
        self.algorithm = algorithm  # Set the algorithm attribute based on the constructor argument

    def prepare_data(self):
        # Implement specific data processing logic
        digits, digit_class = load_digits(return_X_y=True)
        rand_idx = np.random.choice(np.arange(digits.shape[0]), size=SIZE, replace=False)
        data = digits[rand_idx, :]
        self.classes = digit_class[rand_idx]

        transmit_data = {
            'data': data.tolist(),  # Your high-dimensional data converted to a list
            'num_dimensions': 2,  # Target dimensionality for the embedding
            'perplexity': 30,  # Perplexity parameter for t-SNE
            'num_iterations': 10000,  # Number of iterations for optimization
            'learning_rate': 100,  # Learning rate for the optimization
            'algorithm': self.algorithm,  # Use the algorithm attribute
        }

        return transmit_data




if __name__ == '__main__':
    server_communicator = ServerCommunicator()

    # List of algorithms to use
    algorithms = ['ntk', 'jax_tsne', 'sklearn_tsne']

    for algorithm in algorithms[::-1]:
        print(f"Starting t-SNE with {algorithm} algorithm.")
        data_processor = SimpleDataProcessor(algorithm)  # Pass the algorithm to the constructor

        main_data = data_processor.prepare_data()

        response = server_communicator.start_task(main_data)

        if response.status_code == 202:
            task_id = response.json()['task_id']
            print(f'Task started with ID: {task_id}')

            start_time = time.time()  # Record start time

            processed_result = data_processor.wait_for_completion(task_id, server_communicator)

            end_time = time.time()  # Record end time
            execution_time = end_time - start_time
            print(f"Total execution time: {execution_time} seconds")

            data_processor.output_data_processor(processed_result)
        else:
            print(f'Error starting the task: {response.status_code}')
        print("\n")  # Print a newline for better separation between algorithm outputs




