import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

import umap.umap_ as UMAP   # imported as UMAP to avoid collision with the module name
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import plotly.express as px

# --- Generate synthetic 3D data with 4 clusters ---
# Each center defines the mean location of a cluster in 3D space
# cluster_std controls how spread out each cluster is (higher = more overlap)
centers = [[ 2, -6, -6],
           [-1,  9,  4],
           [-8,  7,  2],
           [ 4,  7,  9]]

cluster_std = [1, 1, 2, 3.5]

# make_blobs returns X (the data points) and labels_ (the ground-truth cluster index per point)
X, labels_ = make_blobs(n_samples=500, centers=centers, n_features=3, cluster_std=cluster_std, random_state=42)

# --- Interactive 3D scatter plot of the original data ---
df = pd.DataFrame(X, columns=['X', 'Y', 'Z'])
fig = px.scatter_3d(df, x='X', y='Y', z='Z', color=labels_.astype(str),
                    opacity=0.7, color_discrete_sequence=px.colors.qualitative.G10,
                    title="3D Scatter Plot of Four Blobs")
fig.update_traces(marker=dict(size=5, line=dict(width=1, color='black')), showlegend=False)
fig.update_layout(coloraxis_showscale=False, width=1000, height=800)
#fig.show()

# Standardize the 3D data before applying any dimensionality reduction
# All three methods are sensitive to feature scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- t-SNE: t-Distributed Stochastic Neighbor Embedding ---
# t-SNE preserves local neighbourhood structure — nearby points in high-D stay nearby in 2D
# perplexity: loosely the expected number of neighbours per point (typical range: 5–50)
# max_iter: number of optimisation iterations
tsne = TSNE(n_components=2, random_state=42, perplexity=40, max_iter=1000)
X_tsne = tsne.fit_transform(X_scaled)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels_, cmap='viridis', s=50, alpha=0.7, edgecolor='k')
ax.set_title("2D t-SNE Projection of 3D Data")
ax.set_xlabel("t-SNE Component 1")
ax.set_ylabel("t-SNE Component 2")
ax.set_xticks([])
ax.set_yticks([])
#plt.show()

# --- UMAP: Uniform Manifold Approximation and Projection ---
# UMAP also preserves local structure but is generally faster and better preserves global structure
# min_dist: controls how tightly points are packed together in the 2D embedding
# spread: scale of the embedded space (higher = more spread out clusters)
# n_jobs=1: single-threaded for reproducibility
umap_model = UMAP.UMAP(n_components=2, random_state=42, min_dist=0.5, spread=1, n_jobs=1)
X_umap = umap_model.fit_transform(X_scaled)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
ax.scatter(X_umap[:, 0], X_umap[:, 1], c=labels_, cmap='viridis', s=50, alpha=0.7, edgecolor='k')
ax.set_title("2D UMAP Projection of 3D Data")
ax.set_xlabel("UMAP Component 1")
ax.set_ylabel("UMAP Component 2")
ax.set_xticks([])
ax.set_yticks([])
#plt.show()

# --- PCA: Principal Component Analysis ---
# PCA is a linear method — it projects onto the directions of maximum variance
# Unlike t-SNE and UMAP, PCA cannot capture non-linear structure in the data
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fig = plt.figure(figsize=(8, 6))
ax2 = fig.add_subplot(111)
scatter2 = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_, cmap='viridis', s=50, alpha=0.7, edgecolor='k')
ax2.set_title("2D PCA Projection of 3-D Data")
ax2.set_xlabel("PCA 1")
ax2.set_ylabel("PCA 2")
ax2.set_xticks([])
ax2.set_yticks([])
plt.show()
