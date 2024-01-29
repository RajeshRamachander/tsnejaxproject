import time

class DataProcessor:
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

            if status == 'completed':
                result = status_data['result']
                return self.process_result(result)
            elif status == 'processing':
                print('Task is still processing. Waiting...')
                time.sleep(1)
            else:
                print(f'Unknown status: {status}')
                break
