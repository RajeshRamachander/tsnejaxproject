from flask import Flask, request, jsonify
from celery import Celery
import time
import logging
import worker as wk

app = Flask(__name__)

# Set up basic logging
logging.basicConfig(filename='app.log', level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]')

# app.config['CELERY_BROKER_URL'] = 'pyamqp://guest@localhost//'
# app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/3'  # Use your Redis server details here

app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'  # Use your Redis server details here for the result backend
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 3600  # Set an appropriate caching time (e.g., 1 hour)

celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)
# Configure logging for Celery tasks
celery_log = logging.getLogger('celery')
celery_log.setLevel(logging.INFO)  # Set the desired log level

# Configure a file handler for Celery logs
file_handler = logging.FileHandler('celery.log')  # Specify the log file name

file_handler.setLevel(logging.INFO)  # Set the desired log level
celery_log.addHandler(file_handler)

@celery.task(bind=True)
def process_data(self, data):
    # total = 0
    # for number in data:
    #     # Log messages using Python's logging module
    #     celery_log.info(f"Processing number: {number}")
    #
    #     # Simulate a time-consuming operation
    #     time.sleep(1)
    #     total += number
    # return total

    task_worker = wk.CeleryTask(wk.WorkerDataProcessor(celery_log,wk.Worker()))

    return task_worker.process_data(data)


@app.route('/')
def hello_world():
    app.logger.info('Info level log')
    app.logger.warning('Warning level log')
    return 'Hello, World!'

@app.route('/start-task', methods=['POST'])
def start_task():
    json_data = request.get_json()
    data = json_data['data']

    task = process_data.delay(data)
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

