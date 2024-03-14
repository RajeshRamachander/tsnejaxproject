import requests

BASE_URL = 'http://127.0.0.1:7020'

class ServerCommunicator:
    def start_task(self, data):
        response = requests.post(f'{BASE_URL}/start-task', json=data)
        return response

    def check_task_status(self, task_id):
        status_response = requests.get(f'{BASE_URL}/task-status/{task_id}')
        return status_response.json()
