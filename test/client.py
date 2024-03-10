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
        pass

    @abstractmethod
    def prepare_data(self):
        # Sample data to send in the POST request
        data = {
            'data': [i for i in range(1, 10)]  # Replace with your own data
        }
        return data


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

            print(low_dim)

            scatter = plt.scatter(low_dim[:, 0], low_dim[:, 1], cmap="tab10", c=self.classes)
            plt.legend(*scatter.legend_elements(), fancybox=True, bbox_to_anchor=(1.05, 1))
            plt.show()



class SimpleDataProcessor(DataProcessor):

    def __init__(self):
        self.classes = None
    def prepare_data(self):
        # Implement specific data processing logic
        digits, digit_class = load_digits(return_X_y=True)
        rand_idx = np.random.choice(np.arange(digits.shape[0]), size=SIZE, replace=False)
        data = digits[rand_idx, :]
        self.classes = digit_class[rand_idx]

        transmit_data = {
            'data': data.tolist(),
            'num_dimensions': 2,
            'perplexity': 30,
            'num_iterations': 1000,
            'learning_rate': 100,
            'use_ntk': False
        }

        return transmit_data  

    

# class CIFAR10DataProcessor(DataProcessor):
#
#     def __init__(self):
#         self.classes = None
#
#     def prepare_data(self):
#         # Load CIFAR-10 data
#         (x_train, y_train), (_, _) = cifar10.load_data()
#
#         # Select a random subset of the data
#         rand_idx = np.random.choice(np.arange(x_train.shape[0]), size=SIZE, replace=False)
#         data = x_train[rand_idx]
#         self.classes = y_train[rand_idx].flatten()  # Flatten the class labels array
#
#         # Assuming the server expects data in a flattened form
#         # Flatten the images: (SIZE, 32, 32, 3) -> (SIZE, 3072)
#         data = data.reshape(SIZE, -1)
#
#         transmit_data = {
#             'data': data.tolist(),  # Convert the numpy array to a list for JSON serialization
#             'num_dimensions': 2,
#             'perplexity': 30,
#             'num_iterations': 500,
#             'learning_rate': 100,
#             'batch_size': 100,
#             'pbar': True,
#             'use_ntk': False
#         }
#
#         return transmit_data
#
#     def output_data_processor(self, processed_result):
#         if processed_result is not None:
#             result_received = processed_result.replace("Result received: ", "")
#             low_dim = ast.literal_eval(result_received)
#             low_dim = np.array(low_dim)
#
#             rcParams["font.size"] = 18
#             rcParams["figure.figsize"] = (12, 8)
#
#             scatter = plt.scatter(low_dim[:, 0], low_dim[:, 1], c=self.classes, cmap='viridis', alpha=0.6)
#             plt.colorbar(scatter, label='Classes')
#             plt.title("CIFAR-10 Data Visualized")
#             plt.xlabel("Dimension 1")
#             plt.ylabel("Dimension 2")
#             plt.show()


if __name__ == '__main__':
    server_communicator = ServerCommunicator()
    data_processor = SimpleDataProcessor()

    main_data = data_processor.prepare_data()

    response = server_communicator.start_task(main_data)

    if response.status_code == 202:
        task_id = response.json()['task_id']
        print(f'Task started with ID: {task_id}')

        start_time = time.time()  # Record start time

        processed_result = data_processor.wait_for_completion(task_id, server_communicator)
        # print(processed_result)

        end_time = time.time()  # Record end time
        execution_time = end_time - start_time
        print(f"Total execution time: {execution_time} seconds")

        data_processor.output_data_processor(processed_result)
    else:
        print(f'Error starting the task: {response.status_code}')



