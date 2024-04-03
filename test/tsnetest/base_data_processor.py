
import time
from datetime import datetime
from server_communicator import ServerCommunicator

class BaseDataProcessor:
    def __init__(self, algorithm, preparation_method=None,
                 size=None, filename_to_save_output = None,
                 perplexity=30, num_iterations=10000, learning_rate=100):
        self.algorithm = algorithm
        self.size = size
        self.perplexity = perplexity
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.data = None
        self.classes = None
        self.filename_to_save_output = filename_to_save_output
        if preparation_method is None or preparation_method == 'full':
            self.preparation_method = 'full'
        elif preparation_method == 'matrix':
            self.preparation_method = 'matrix'
            if not self.filename_to_save_output:
                self.filename_to_save_output = self.generate_file_name_with_timestamp()
        else:
            print(f"Unknown preparation method: {self.preparation_method}")
            raise ValueError(f"Unknown preparation method: {preparation_method}")
        self.data, self.classes, self.size = self.load_data()

        self.pay_load_data = self.prepare_data()
        self.server_communicator = ServerCommunicator()


    def load_data(self):
        raise NotImplementedError("Subclasses must implement")

    def prepare_data(self):
        raise NotImplementedError("Subclasses must implement prepare_data method")

    def start_task(self):
        start_time = time.time()
        response = self.server_communicator.start_task(self.pay_load_data)

        if response.status_code == 202:
            task_id = response.json()['task_id']
            print(f'Task started with ID: {task_id}')
            processed_result = self.wait_for_completion(task_id)
            execution_time = time.time() - start_time
            if response.status_code == 202:
                self.output_data_processor(processed_result)
            else:
                print(f'Error starting the task: {response.status_code}')
            print("\n")
            return execution_time
        else:
            execution_time = time.time() - start_time
            return execution_time



    def output_data_processor(self, processed_result):
        raise NotImplementedError("Subclasses must implement output_data_processor method")

    def output_data_processor_full(self, processed_result):
        raise NotImplementedError("Subclasses must implement output_data_processor_full method")

    def output_data_processor_matrix(self, processed_result):
        raise NotImplementedError("Subclasses must implement output_data_processor_matrix method")

    def generate_file_name_with_timestamp(self):
        timestamp = datetime.now()
        return f'Load_digits_matrix_{self.size}_{timestamp.strftime("%Y%m%d-%H-%M-%S")}.csv'

    def wait_for_completion(self, task_id):
        """Polls the server for the status of the task until it is completed.

        Args:
            task_id (str): The ID of the task to check.
            server_communicator (ServerCommunicator): The communicator object to interact with the server.

        Returns:
            The result of the task if successful, None otherwise.
        """
        while True:
            status_data = self.server_communicator.check_task_status(task_id)
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
