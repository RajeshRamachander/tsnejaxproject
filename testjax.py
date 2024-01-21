import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.datasets import load_digits
import numpy as np
import tsnejax as tj


rcParams["font.size"] = 18
rcParams["figure.figsize"] = (12, 8)


digits, digit_class = load_digits(return_X_y=True)
rand_idx = np.random.choice(np.arange(digits.shape[0]), size=500, replace=False)
data = digits[rand_idx, :].copy()
classes = digit_class[rand_idx]

low_dim = tj.compute_low_dimensional_embedding(data, 2, 30, 500, \
                                            100, pbar=True, use_ntk=False)


scatter = plt.scatter(low_dim[:, 0], low_dim[:, 1], cmap="tab10", c=classes)
plt.legend(*scatter.legend_elements(), fancybox=True, bbox_to_anchor=(1.05, 1))
plt.show()

import requests
import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Set plotting parameters
rcParams["font.size"] = 18
rcParams["figure.figsize"] = (12, 8)

# Load data from load_digits
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
    'n_iter': 1000,
    'pbar': True,
    'use_ntk': False
}

# URL of the Flask app
url = 'http://localhost:3000/compute_embedding'

try:
    # Send POST request to the Flask app with a timeout
    response = requests.post(url, json=payload, timeout=10)

    # Process the response
    if response.status_code == 200:
        low_dim = np.array(response.json())
        # Plotting
        scatter = plt.scatter(low_dim[:, 0], low_dim[:, 1], cmap="tab10", c=classes)
        plt.legend(*scatter.legend_elements(), fancybox=True, bbox_to_anchor=(1.05, 1))
        plt.show()
    else:
        print("Failed to get response from the server:", response.status_code)
        # Optionally print more details from the response
        print(response.text)

except requests.exceptions.ConnectionError:
    print("Failed to connect to the server. Please check the URL and ensure the server is running.")
except requests.exceptions.Timeout:
    print("Request timed out. The server might be too slow or unresponsive.")
except requests.exceptions.RequestException as e:
    # For any other exceptions that requests might raise
    print(f"An error occurred: {e}")


# import requests

# response = requests.get('http://localhost:3000/')
# print(response.text)