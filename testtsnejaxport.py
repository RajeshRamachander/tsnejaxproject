import requests
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

# Configuration for matplotlib
plt.rcParams["font.size"] = 18
plt.rcParams["figure.figsize"] = (12, 8)

def send_data_to_server(url, payload, timeout=60):
    """
    Send data to the server and return the response.

    :param url: URL of the server endpoint.
    :param payload: Payload to be sent as JSON. Should be a dictionary.
    :param timeout: Timeout for the request in seconds.
    :return: Server response or None if an error occurred.
    """
    print("Sending data to server...")
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        if response.status_code == 200:
            print("Received response from server.")
            return np.array(response.json())  # Convert response back to numpy array
        else:
            print("Error:", response.status_code, response.text)
            return None
    except requests.exceptions.Timeout:
        print("Request timed out")
        return None
    except requests.exceptions.RequestException as e:
        print("Error during request:", e)
        return None

def prepare_payload(data, num_dimensions=2, target_perplexity=30,
                    max_iterations=100, learning_rate=100, scaling_factor=4.0,
                    pbar=False, random_state=42, perp_tol=1e-8, use_ntk=True):
    """
    Prepare the payload for sending to the server.

    :param data: Data to be sent. Can be a NumPy array.
    :return: Payload as a dictionary.
    """
    return {
        'data': data.tolist(),  # Convert numpy array to list for JSON serialization
        'num_dimensions': num_dimensions,
        'target_perplexity': target_perplexity,
        'max_iterations': max_iterations,
        'learning_rate': learning_rate,
        'scaling_factor': scaling_factor,
        'pbar': pbar,
        'random_state': random_state,
        'perp_tol': perp_tol,
        'use_ntk': use_ntk
    }

def prepare_digits_data():
    """
    Prepare digit dataset for sending to the server.

    :return: Tuple of (data, classes)
    """
    digits, digit_classes = load_digits(return_X_y=True)
    rand_idx = np.random.choice(len(digits), size=500, replace=False)
    return digits[rand_idx], digit_classes[rand_idx]

def plot_embedding(low_dim, classes):
    """
    Plot the low-dimensional embedding.

    :param low_dim: Low-dimensional data.
    :param classes: Corresponding classes for the data points.
    """
    scatter = plt.scatter(low_dim[:, 0], low_dim[:, 1], cmap="tab10", c=classes)
    plt.legend(*scatter.legend_elements(), fancybox=True, bbox_to_anchor=(1.05, 1))
    plt.show()

if __name__ == '__main__':
    data, classes = prepare_digits_data()

    # URL of the server endpoint
    url = 'http://127.0.0.1:5000/compute_tsne'

    # Prepare payload
    payload = prepare_payload(data)

    # Send data to server and receive processed data
    print("Payload prepared, sending to server...")
    processed_data = send_data_to_server(url, payload)
    if processed_data is not None:
        print("Data received from server, plotting...")
        processed_data = np.array(processed_data)  # Convert back to numpy array if needed
        plot_embedding(processed_data, classes)
    else:
        print("No data received from server.")
