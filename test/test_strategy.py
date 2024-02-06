from sklearn.datasets import load_digits
import numpy as np
from api import server as srv



digits, digit_class = load_digits(return_X_y=True)
rand_idx = np.random.choice(np.arange(digits.shape[0]), size=500, replace=False)
data = digits[rand_idx, :]

data_args = {
    'data': data.tolist(),
    'num_dimensions': 2,
    'perplexity': 30,
    'num_iterations': 500,
    'learning_rate': 100,
    'batch_size': 100,
    'pbar': True,
    'use_ntk': False
}

total = srv.CeleryTask(srv.WorkerDataProcessor()).process_data(data_args)
print(f"Total (Worker Strategy): {total}")


