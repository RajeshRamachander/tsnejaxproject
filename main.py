from data_processor import DataProcessor, SimpleDataProcessor

import requests

BASE_URL = 'http://localhost:7020'

class ServerCommunicator:

    def start_task(self, data):
        response = requests.post(f'{BASE_URL}/start-task', json=data)
        return response

    def check_task_status(self, task_id):
        status_response = requests.get(f'{BASE_URL}/task-status/{task_id}')
        return status_response.json()


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
