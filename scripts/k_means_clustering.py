import warnings

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

warnings.filterwarnings('ignore')

# Fix the random seed so the blob layout is the same every run
np.random.seed(0)

# Generate 5000 synthetic 2D data points arranged in 4 clusters
# cluster_std=0.9 controls how spread out each cluster is
X, y = make_blobs(n_samples=5000, centers=[[4, 4], [-2, -1], [2, -3], [1, 1]], cluster_std=0.9)

# Visualise the raw data before clustering (no labels shown)
plt.scatter(X[:, 0], X[:, 1], marker='.', alpha=0.3, ec='k', s=80)
# plt.show()

# Fit K-Means with k=3 clusters
# init='k-means++': smarter centroid initialisation that spreads starting centroids apart,
#   reducing the chance of converging to a poor local minimum vs. random init
# n_init=12: run the algorithm 12 times with different seeds and keep the best result
k_means = KMeans(init="k-means++", n_clusters=3, n_init=12)
k_means.fit(X)

# labels_: integer cluster assignment (0, 1, or 2) for every data point
k_means_labels = k_means.labels_
print(k_means_labels)

# cluster_centers_: (x, y) coordinates of each centroid
k_means_cluster_centers = k_means.cluster_centers_
print(k_means_cluster_centers)

# Set up the figure
fig = plt.figure(figsize=(6, 4))

# Generate a distinct colour for each cluster from the tab10 colour map
colors = plt.cm.tab10(np.linspace(0, 1, len(set(k_means_labels))))

ax = fig.add_subplot(1, 1, 1)

# Plot each cluster's data points and its centroid in the matching colour
# k iterates over cluster indices 0, 1, 2; col is the associated colour
for k, col in zip(range(len(k_means_cluster_centers)), colors):
    # Boolean mask: True for every point assigned to cluster k
    my_members = (k_means_labels == k)

    # Coordinates of this cluster's centroid
    cluster_center = k_means_cluster_centers[k]

    # Plot data points belonging to this cluster
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.', ms=10)

    # Plot the centroid with a black edge so it stands out from the data points
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)

ax.set_title('KMeans')
ax.set_xticks(())
ax.set_yticks(())

# plt.show()
