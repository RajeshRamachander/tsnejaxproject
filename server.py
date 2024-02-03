from flask import Flask, request, jsonify
from celery import Celery
import logging
import worker as wk


app = Flask(__name__)

def configure_logger(logger_name, log_file):
    """Configures a logger with a FileHandler and formatter."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]")
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

# Configure Flask's app logger
app_logger = configure_logger("flask_app", "flask_app.log")

# Configure Celery's task logger
task_logger = configure_logger("celery_tasks", "celery_tasks.log")

app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'  # Use your Redis server details here for the result backend
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 3600  # Set an appropriate caching time (e.g., 1 hour)

celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

@celery.task(bind=True)
def process_data(self, data):


    # task_logger.info('Calling app.process_data')

    return wk.CeleryTask(wk.WorkerDataProcessor()).process_data(data)



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

    if data is None:
        return jsonify({'error': 'Data not provided in the request'}), 400

    app.logger.info(f"Type of low_dim: {type(data)}")
    app.logger.info(f"Low_dim data (partial view): {data[:10]}")


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

