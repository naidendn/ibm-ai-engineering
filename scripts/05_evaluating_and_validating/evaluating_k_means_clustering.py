import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score


def evaluate_clustering(X, labels, n_clusters, ax=None, title_suffix=''):
    """
    Evaluate a clustering model using silhouette scores and the Davies-Bouldin index.

    Parameters:
    X (ndarray): Feature matrix.
    labels (array-like): Cluster labels assigned to each sample.
    n_clusters (int): The number of clusters in the model.
    ax: The subplot axes to plot on.
    title_suffix (str): Optional suffix for plot title.

    Returns:
    None: Displays silhouette scores and a silhouette plot.
    """
    if ax is None:
        ax = plt.gca()  # fall back to the current active axis

    # Silhouette score per sample: ranges from -1 (wrong cluster) to +1 (well-matched)
    # silhouette_avg summarises the whole clustering in a single number
    silhouette_avg = silhouette_score(X, labels)
    sample_silhouette_values = silhouette_samples(X, labels)

    # --- Silhouette plot ---
    # Each horizontal bar represents one cluster; bar width = per-sample silhouette coefficient
    # Bars extending past the red dashed line (= average) indicate above-average cohesion
    unique_labels = np.unique(labels)
    colormap = cm.tab10
    color_dict = {label: colormap(float(label) / n_clusters) for label in unique_labels}
    y_lower = 10
    for i in unique_labels:
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = color_dict[i]
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, ith_cluster_silhouette_values,
                         facecolor=color, edgecolor=color, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax.set_title(f'Silhouette Score for {title_suffix} \n' +
                 f'Average Silhouette: {silhouette_avg:.2f}')
    ax.set_xlabel('Silhouette Coefficient')
    ax.set_ylabel('Cluster')
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax.set_xlim([-0.25, 1])  # silhouette range is [-1, 1]; -0.25 gives padding on the left
    ax.set_yticks([])  # cluster labels are already rendered as text; tick marks aren't needed


# --- Synthetic dataset ---
# make_blobs creates Gaussian clusters; cluster_std controls how spread-out each blob is
# Using different stddevs per cluster simulates real-world unevenly sized / shaped groups
X, y = make_blobs(n_samples=500, n_features=2, centers=4, cluster_std=[1.0, 3, 5, 2], random_state=42)

# --- Fit K-Means with the known number of clusters (k=4) ---
# KMeans assigns each sample to its nearest centroid, then recomputes centroids;
# this repeats until convergence (centroids stop moving)
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
y_kmeans = kmeans.fit_predict(X)  # returns cluster label for every sample

colormap = cm.tab10

# Panel 1: raw data with centroids overlaid (no colour by cluster yet)
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], s=50, alpha=0.6, edgecolor='k')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X', alpha=0.9, label='Centroids')
plt.title(f'Synthetic Blobs with {n_clusters} Clusters')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

# Panel 2: same data coloured by the cluster assignment predicted by K-Means
colors = colormap(y_kmeans.astype(float) / n_clusters)

plt.subplot(1, 3, 2)
plt.scatter(X[:, 0], X[:, 1], c=colors, s=50, alpha=0.6, edgecolor='k')

# Draw a white circle at each centroid, then overlay the cluster index as a marker
centers = kmeans.cluster_centers_
plt.scatter(
    centers[:, 0],
    centers[:, 1],
    marker="o",
    c="white",
    alpha=1,
    s=200,
    edgecolor="k",
    label='Centroids'
)
for i, c in enumerate(centers):
    plt.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

plt.title(f'KMeans Clustering with {n_clusters} Clusters')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

# Panel 3: silhouette plot — shows how well each sample fits its assigned cluster
plt.subplot(1, 3, 3)
evaluate_clustering(X, y_kmeans, n_clusters, title_suffix=' k-Means Clustering')
# plt.show()

# --- Effect of random initialisation ---
# K-Means is sensitive to the initial centroid positions (random_state=None → different seed each run)
# Running it multiple times shows how much the result varies; inertia measures total within-cluster
# sum of squared distances — lower is better, but depends on k
n_runs = 8
inertia_values = []

n_cols = 2
n_rows = -(-n_runs // n_cols)  # ceiling division: ensures enough rows for all runs
plt.figure(figsize=(16, 16))

for i in range(n_runs):
    kmeans = KMeans(n_clusters=4, random_state=None)  # random init each run
    kmeans.fit(X)
    inertia_values.append(kmeans.inertia_)

    plt.subplot(n_rows, n_cols, i + 1)
    plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='tab10', alpha=0.6, edgecolor='k')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', s=200, marker='x',
                label='Centroids')
    plt.title(f'K-Means Run {i + 1}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(loc='upper right', fontsize='small')

plt.tight_layout()
# plt.show()

# Range of k values to test for the elbow / metric sweep below
k_values = range(2, 11)

# Print the inertia from the random-init runs above
for i, inertia in enumerate(inertia_values, start=1):
    print(f'Run {i}: Inertia={inertia:.2f}')

# --- Sweep k from 2 to 10 and compute three complementary metrics ---
inertia_values = []
silhouette_scores = []
davies_bouldin_indices = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    y_kmeans = kmeans.fit_predict(X)

    # Inertia: total within-cluster variance — always decreases as k grows,
    # so look for the "elbow" where the rate of improvement flattens out
    inertia_values.append(kmeans.inertia_)

    # Silhouette score: higher is better (max 1.0); peak indicates the optimal k
    silhouette_scores.append(silhouette_score(X, y_kmeans))

    # Davies-Bouldin index: ratio of within-cluster scatter to between-cluster distance
    # lower is better (min 0); a trough in the plot suggests the best k
    davies_bouldin_indices.append(davies_bouldin_score(X, y_kmeans))

# --- Plot all three metrics side-by-side ---
# Using all three together reduces the risk of choosing the wrong k from any single metric
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.plot(k_values, inertia_values, marker='o')
plt.title('Elbow Method: Inertia vs. k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')

plt.subplot(1, 3, 2)
plt.plot(k_values, silhouette_scores, marker='o')
plt.title('Silhouette Score vs. k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')

plt.subplot(1, 3, 3)
plt.plot(k_values, davies_bouldin_indices, marker='o')
plt.title('Davies-Bouldin Index vs. k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Davies-Bouldin Index')

plt.tight_layout()
plt.show()
