from flask import Flask, request, jsonify
from celery import Celery
import logging
import worker as wk

from celery.utils.log import get_task_logger
from celery.signals import after_setup_logger
import os

logger = get_task_logger("tasks")


app = Flask(__name__)


log_file = 'app.log'


# Set up basic logging for the Flask app
logging.basicConfig(filename=log_file, level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]')


app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'  # Use your Redis server details here for the result backend
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 3600  # Set an appropriate caching time (e.g., 1 hour)

celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

# Set up Celery logger

celery_log = logging.getLogger('celery')
celery_log.setLevel(logging.INFO)

# Create a file handler for Celery logs
celery_file_handler = logging.FileHandler('./celery.log')
celery_file_handler.setLevel(logging.INFO)

# Create a formatter for the Celery log messages
celery_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
celery_file_handler.setFormatter(celery_formatter)

# Add the file handler to the Celery logger
celery_log.addHandler(celery_file_handler)

# Add the Celery logger to the worker's logger handlers
worker_logger = logging.getLogger('worker')
worker_logger.addHandler(celery_file_handler)

@after_setup_logger.connect
def setup_celery_logger(logger, *args, **kwargs):
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("tasks")
    fh = logging.FileHandler('tasks.log')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

@celery.task(bind=True)
def process_data(self, data):


    logger.info('Calling app.process_data')

    task_worker = wk.CeleryTask(wk.WorkerDataProcessor(celery_log))

    return task_worker.process_data(data)


@app.route('/')
def hello_world():
    app.logger.info('Info level log')
    app.logger.warning('Warning level log')
    return 'Hello, World!'

@app.route('/start-task', methods=['POST'])
def start_task():
    # Log a message when the route is accessed
    app.logger.info('Received POST request to start a task.')
    json_data = request.get_json()
    data = json_data['data']

    app.logger.info(f"Type of low_dim: {type(data)}")
    app.logger.info(f"Low_dim data (partial view): {data[:10]}")
    task = process_data.delay(json_data)
    return jsonify({'task_id': task.id}), 202

@app.route('/task-status/<task_id>', methods=['GET'])
def task_status(task_id):
    result = process_data.AsyncResult(task_id)
    if result.ready():
        return jsonify({'status': 'completed', 'result': result.get()})
    return jsonify({'status': 'processing'}), 202



if __name__ == '__main__':
    print("Flask app is starting...")
    app.run(debug=True, port=7020)

