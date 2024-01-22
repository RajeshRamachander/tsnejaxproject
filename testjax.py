
import requests
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.datasets import load_digits
import numpy as np

rcParams["font.size"] = 18
rcParams["figure.figsize"] = (12, 8)

# Load data
digits, digit_class = load_digits(return_X_y=True)
rand_idx = np.random.choice(np.arange(digits.shape[0]), size=500, replace=False)
data = digits[rand_idx, :].copy()
classes = digit_class[rand_idx]

# Prepare data and parameters for the POST request
payload = {
    'data': data.tolist(),
    'n_components': 2,
    'perplexity': 30,
    'learning_rate': 500,
    'n_iter': 100,
    'pbar': True,
    'use_ntk': False
}

# URL of the Flask app
url = 'http://localhost:3000/compute_embedding'

# Send POST request
response = requests.post(url, json=payload)

# Check if the request was successful
if response.status_code == 200:
    # Parse the JSON response
    low_dim = np.array(response.json())

    # Plotting
    scatter = plt.scatter(low_dim[:, 0], low_dim[:, 1], cmap="tab10", c=classes)
    plt.legend(*scatter.legend_elements(), fancybox=True, bbox_to_anchor=(1.05, 1))
    plt.show()
else:
    print("Failed to get a successful response:", response.status_code)
    print(response.text)








# app = Flask(__name__)

# @app.route('/compute_embedding', methods=['POST'])
# def compute_embedding():
#     content = request.json
#     if not content or 'data' not in content:
#         return jsonify({"error": "Missing 'data' in request"}), 400

#     data = np.array(content['data'])  # Convert list to NumPy array
#     # Add validation for 'data' if necessary, e.g., check type, format

#     # Default values are used if parameters are not provided
#     n_components = content.get('n_components', 2)
#     perplexity = content.get('perplexity', 30)
#     learning_rate = content.get('learning_rate', 500)
#     n_iter = content.get('n_iter', 100)
#     pbar = content.get('pbar', True)
#     use_ntk = content.get('use_ntk', False)

#     try:
#         # Call your processing function
#         result = compute_low_dimensional_embedding(data, n_components, perplexity, learning_rate, n_iter, pbar, use_ntk)
#         return jsonify(result.tolist())
#     except Exception as e:
#         # Handle exceptions and return an error message
#         return jsonify({"error": str(e)}), 500



# if __name__ == "__main__":
#     app.run(host='0.0.0.0', port=3000)