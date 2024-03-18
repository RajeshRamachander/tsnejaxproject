from data_processor import DataProcessor
from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import ast
import time

class SimpleDataProcessor:
    def __init__(self, algorithm, size=1000, perplexity=30, num_iterations=10000, learning_rate=100):
        self.algorithm = algorithm
        self.size = size
        self.perplexity = perplexity
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.classes = None  # This should be set based on your dataset

    def prepare_data(self):
        try:
            digits, digit_class = load_digits(return_X_y=True)
            rand_idx = np.random.choice(np.arange(digits.shape[0]), size=self.size, replace=False)
            print(digits.shape[1])
            data = digits[rand_idx, :]
            self.classes = digit_class[rand_idx]

            transmit_data = {
                'data': data.tolist(),
                'num_dimensions': 2,
                'perplexity': self.perplexity,
                'num_iterations': self.num_iterations,
                'learning_rate': self.learning_rate,
                'algorithm': self.algorithm,
            }
            return transmit_data
        except Exception as e:
            print(f"Error preparing data: {e}")
            return None

    def output_data_processor(self, processed_result):
        
        if processed_result is None:
            print("No result to process.")
            return
        else:
            print(f'Result: {processed_result}')

        # Continue with the rest of the method as before
        if isinstance(processed_result, list):
            low_dim = np.array(processed_result)
        else:
            # If it's a string, perform the original processing
            result_received = processed_result.replace("Result received: ", "")
            low_dim = ast.literal_eval(result_received)
            low_dim = np.array(low_dim)

        rcParams["font.size"] = 18
        rcParams["figure.figsize"] = (12, 8)

        print(f"Using {self.algorithm} for t-SNE.")
        scatter = plt.scatter(low_dim[:, 0], low_dim[:, 1], cmap="tab10", c=self.classes)
        plt.legend(*scatter.legend_elements(), fancybox=True, bbox_to_anchor=(1.05, 1))
        plt.show()




    def wait_for_completion(self, task_id, server_communicator):
        """Polls the server for the status of the task until it is completed.

        Args:
            task_id (str): The ID of the task to check.
            server_communicator (ServerCommunicator): The communicator object to interact with the server.

        Returns:
            The result of the task if successful, None otherwise.
        """
        while True:
            status_data = server_communicator.check_task_status(task_id)
            status = status_data['status']

            if status == 'success':
                print('Task completed successfully.')
                return status_data['result']
            elif status == 'processing':
                print('Task is still processing. Waiting...')
                time.sleep(1)  # Wait for a bit before checking again
            else:
                print(f'Error or unknown status received: {status}')
                return None

