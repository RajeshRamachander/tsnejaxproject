import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
import numpy as np
from sklearn.datasets import load_digits

SIZE = 500

(x_train, y_train), (_, _) = cifar10.load_data()
print(x_train.shape)


# Select a random subset of the data
rand_idx = np.random.choice(np.arange(x_train.shape[0]), size=SIZE, replace=False)
data = x_train[rand_idx]


# Assuming the server expects data in a flattened form
# Flatten the images: (SIZE, 32, 32, 3) -> (SIZE, 3072)
data = data.reshape(SIZE, -1)
print(data.shape)
#
transmit_data = {
    'data': data.tolist(),  # Convert the numpy array to a list for JSON serialization
    'num_dimensions': 2,
    'perplexity': 30,
    'num_iterations': 500,
    'learning_rate': 100,
    'batch_size': 100,
    'pbar': True,
    'use_ntk': False
}

print(len(transmit_data['data']))

digits, digit_class = load_digits(return_X_y=True)
print(digits.shape)
rand_idx = np.random.choice(np.arange(digits.shape[0]), size=SIZE, replace=False)
data = digits[rand_idx, :]

print(data.shape)

transmit_data = {
    'data': data.tolist(),
    'num_dimensions': 2,
    'perplexity': 30,
    'num_iterations': 500,
    'learning_rate': 100,
    'batch_size': 100,
    'pbar': True,
    'use_ntk': False
}

print(len(transmit_data['data']))