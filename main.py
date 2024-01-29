from server_communication import ServerCommunicator
from data_processor import DataProcessor

if __name__ == '__main__':
    server_communicator = ServerCommunicator()
    data_processor = DataProcessor()

    main_data = data_processor.prepare_data()

    response = server_communicator.start_task(main_data)

    if response.status_code == 202:
        task_id = response.json()['task_id']
        print(f'Task started with ID: {task_id}')

        processed_result = data_processor.wait_for_completion(task_id, server_communicator)
        print(processed_result)
    else:
        print(f'Error starting the task: {response.status_code}')
