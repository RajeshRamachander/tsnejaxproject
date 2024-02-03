import time
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.datasets import load_digits
import numpy as np
import ast
import requests

BASE_URL = 'http://localhost:7020'

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

    @abstractmethod
    def output_data_processor(self, processed_result):

        pass



class SimpleDataProcessor(DataProcessor):

    def __init__(self):
        self.classes = None
    def prepare_data(self):
        # Implement specific data processing logic
        digits, digit_class = load_digits(return_X_y=True)
        rand_idx = np.random.choice(np.arange(digits.shape[0]), size=500, replace=False)
        data = digits[rand_idx, :]
        self.classes = digit_class[rand_idx]

        transmit_data = {
            'data': data.tolist(),
            'num_dimensions': 2,
            'perplexity': 30,
            'num_iterations': 500,
            'learning_rate': 100,
            'batch_size': 100,
            'pbar': True,
            'use_ntk': False
        }

        return transmit_data

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


if __name__ == '__main__':
    server_communicator = ServerCommunicator()
    data_processor = SimpleDataProcessor()

    main_data = data_processor.prepare_data()

    response = server_communicator.start_task(main_data)

    if response.status_code == 202:
        task_id = response.json()['task_id']
        print(f'Task started with ID: {task_id}')

        processed_result = data_processor.wait_for_completion(task_id, server_communicator)
        # print(processed_result)

        data_processor.output_data_processor(processed_result)
    else:
        print(f'Error starting the task: {response.status_code}')