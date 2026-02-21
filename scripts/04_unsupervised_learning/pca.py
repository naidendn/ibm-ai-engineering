import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# Fix the random seed for reproducibility
np.random.seed(42)

# Generate 200 samples from a 2D multivariate normal distribution
# The covariance matrix [[3,2],[2,2]] creates correlated data (positive covariance)
# This is an ideal case for PCA: the data has a clear principal direction
mean = [0, 0]
cov = [[3, 2], [2, 2]]
X = np.random.multivariate_normal(mean=mean, cov=cov, size=200)

# Fit PCA and immediately transform the data
# n_components=2 retains both principal components (no dimensionality reduction here,
# we're using it to find the axes of maximum variance)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# pca.components_ is a (2, 2) matrix where each row is a principal component vector
# These vectors are orthogonal and point in the directions of maximum variance
components = pca.components_

# Print the fraction of total variance explained by each principal component
# PC1 explains the most variance, PC2 explains the remainder
print(pca.explained_variance_ratio_)

# Project each data point onto PC1 and PC2 to get scalar scores along each axis
# np.dot(X, components[0]) gives the coordinate of each point along PC1
projection_pc1 = np.dot(X, components[0])
projection_pc2 = np.dot(X, components[1])

# Convert scalar projections back to 2D Cartesian coordinates for plotting
# This places the projected points on the PC1 and PC2 lines in the original space
x_pc1 = projection_pc1 * components[0][0]
y_pc1 = projection_pc1 * components[0][1]
x_pc2 = projection_pc2 * components[1][0]
y_pc2 = projection_pc2 * components[1][1]

# Plot the original data and the orthogonal projections onto each principal component
plt.figure()
plt.scatter(X[:, 0], X[:, 1], label='Original Data', ec='k', s=50, alpha=0.6)

# Red X markers: where each point projects onto PC1 (the axis of most variance)
plt.scatter(x_pc1, y_pc1, c='r', ec='k', marker='X', s=70, alpha=0.5, label='Projection onto PC 1')

# Blue X markers: where each point projects onto PC2 (the orthogonal axis)
plt.scatter(x_pc2, y_pc2, c='b', ec='k', marker='X', s=70, alpha=0.5, label='Projection onto PC 2')

plt.title('Linearly Correlated Data Projected onto Principal Components')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
