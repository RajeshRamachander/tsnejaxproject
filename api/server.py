from flask import Flask, request, jsonify
from celery import Celery
import logging
from abc import ABC, abstractmethod
import tsnejax as tj
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

app = Flask(__name__)


celery = Celery(
    app.name,  # Use app.name for consistency
    broker="redis://127.0.0.1:6379/0",
    backend="redis://127.0.0.1:6379/0"
)

def configure_logger(logger_name, log_file):
    """Configures a logger with a FileHandler and formatter."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]")
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger

# Configure Celery's task logger
task_logger = configure_logger("celery_tasks", "celery_tasks.log")

# Configure Flask's app logger
app_logger = configure_logger("flask_app", "flask_app.log")


@celery.task(name='app.server.process_data',bind=True)
def process_data(self, data):


    task_logger.info('Calling app.process_data')

    return CeleryTask(WorkerDataProcessor()).process_data(data)



@app.route('/')
def hello_world():
    app_logger.info('Info level log')
    app_logger.warning('Warning level log')
    return 'Hello, World!'

@app.route('/start-task', methods=['POST'])
def start_task():
    # Log a message when the route is accessed
    app_logger.info('Received POST request to start a task.')
    json_data = request.get_json()
    data = json_data['data']

    if data is None:
        return jsonify({'error': 'Data not provided in the request'}), 400

    app_logger.info(f"Type of low_dim: {type(data)}")
    app_logger.info(f"Low_dim data (partial view): {data[:10]}")


    task = process_data.delay(json_data)
    return jsonify({'task_id': task.id}), 202

@app.route('/task-status/<task_id>', methods=['GET'])
def task_status(task_id):
    result = process_data.AsyncResult(task_id)
    if result.state == 'SUCCESS':
        return jsonify({'status': 'success', 'result': result.get()})
    elif result.status == 'FAILURE':
        return jsonify({'status': 'failure'}), 202

    return jsonify({'status': 'processing'}), 202



if __name__ == '__main__':
    print("Flask app is starting...")
    app.run(debug=True, port=7020)

