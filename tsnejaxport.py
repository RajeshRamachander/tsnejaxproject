from flask import Flask, request, jsonify
import numpy as np
import tsnejax

app = Flask(__name__)

@app.route('/compute_tsne', methods=['POST'])
def compute_tsne():
    try:
        # Extract parameters from the request
        content = request.get_json()
        high_dimensional_data = np.array(content['data'])
        num_dimensions = content.get('num_dimensions', 2)
        target_perplexity = content.get('target_perplexity', 30)
        max_iterations = content.get('max_iterations', 100)
        learning_rate = content.get('learning_rate', 100)
        scaling_factor = content.get('scaling_factor', 4.0)
        pbar = content.get('pbar', False)
        random_state = content.get('random_state', 42)
        perp_tol = content.get('perp_tol', 1e-8)
        use_ntk = content.get('use_ntk', True)

        # Call the tsnejax function
        low_dim_data = tsnejax.compute_low_dimensional_embedding(
            high_dimensional_data, num_dimensions, target_perplexity,
            max_iterations, learning_rate, scaling_factor, pbar,
            random_state, perp_tol, use_ntk
        )

        # Convert the result to a list and return it
        return jsonify(low_dim_data.tolist())

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
