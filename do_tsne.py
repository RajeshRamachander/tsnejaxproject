import numpy as np
import tsne as my_tsne
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Generate a synthetic dataset with three clusters
np.random.seed(42)

# Cluster 1: centered at (0, 0)
cluster_1 = np.random.randn(100, 2)

# Cluster 2: centered at (10, 10)
cluster_2 = np.random.randn(100, 2) + 10

# Cluster 3: centered at (-10, 10)
cluster_3 = np.random.randn(100, 2) + np.array([-10, 10])

X = np.vstack([cluster_1, cluster_2, cluster_3])

# Plotting
plt.figure(figsize=(18, 6))

# Original Data
plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], 20, range(len(X)))
plt.title('Original Data')
plt.grid(True)

# Custom t-SNE transformation
perplexity_value = 30
Y_custom = my_tsne.t_sne_dimensionality_reduction(X, 2, 2, perplexity_value)
plt.subplot(1, 3, 2)
plt.scatter(Y_custom[:, 0], Y_custom[:, 1], 20, range(len(Y_custom)))
plt.title('Custom t-SNE transformed Data')
plt.grid(True)

# scikit-learn t-SNE transformation
tsne_sklearn = TSNE(n_components=2, perplexity=perplexity_value, random_state=42)
Y_sklearn = tsne_sklearn.fit_transform(X)
plt.subplot(1, 3, 3)
plt.scatter(Y_sklearn[:, 0], Y_sklearn[:, 1], 20, range(len(Y_sklearn)))
plt.title('sklearn t-SNE transformed Data')
plt.grid(True)

plt.tight_layout()
plt.show()
