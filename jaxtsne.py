import numpy as np

# Sample data matrix (mean-centered)
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Compute the covariance matrix
cov_matrix = np.cov(X, rowvar=False)

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort eigenvalues and eigenvectors (optional)
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Print results
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:")
print(eigenvectors)