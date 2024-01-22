from flask import Flask, request, jsonify
from celery import Celery
import numpy as np
from tsnejax import compute_low_dimensional_embedding 

app = Flask(__name__)

# Configure Celery
celery = Celery(app.name, broker='redis://localhost:6379/0', backend='redis://localhost:6379/0')

@celery.task
def compute_low_dimensional_embedding_async(data, num_dimensions, target_perplexity, max_iterations):
    # Convert data back to numpy array
    data = np.array(data)
    # Call the actual computation function
    result = compute_low_dimensional_embedding(data, num_dimensions, target_perplexity, max_iterations)
    return result.tolist()  # Convert numpy array to list for serialization

@app.route('/compute_embedding', methods=['POST'])
def compute_embedding():
    try:
        content = request.json
        if 'data' not in content:
            return jsonify({"error": "Missing 'data' in request"}), 400
        data = content['data']
        num_dimensions = content.get('num_dimensions', 2)
        target_perplexity = content.get('target_perplexity', 30)
        max_iterations = content.get('max_iterations', 1000)

        # Start the Celery task
        task = compute_low_dimensional_embedding_async.delay(data, num_dimensions, target_perplexity, max_iterations)
        return jsonify({"task_id": task.id}), 202
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3000)
