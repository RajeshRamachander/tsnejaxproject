
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.datasets import load_digits
import numpy as np
from app import tsnejax as tj

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
