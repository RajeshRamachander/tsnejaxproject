from flask import Flask, request, jsonify
import logging
from celery import Celery

celery = Celery(
    'tasks',  # Use the app name from your Flask app
    broker="redis://127.0.0.1:6379/0",
    backend="redis://127.0.0.1:6379/0"
)

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

    return logger

# Configure Celery's task logger
task_logger = configure_logger("celery_tasks", "celery_tasks.log")

# Configure Flask's app logger
app_logger = configure_logger("flask_app", "flask_app.log")




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


    task = celery.send_task('tasks.tsne', args=[json_data])  # Use `app.send_task`
    return jsonify({'task_id': task.id}), 202

@app.route('/task-status/<task_id>', methods=['GET'])
def task_status(task_id):
    result = celery.AsyncResult(task_id)
    if result.state == 'SUCCESS':
        return jsonify({'status': 'success', 'result': result.get()})
    elif result.status == 'FAILURE':
        return jsonify({'status': 'failure'}), 202

    return jsonify({'status': 'processing'}), 202



if __name__ == '__main__':
    print("Flask app is starting...")
    app.run(debug=True, port=7020)