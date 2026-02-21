import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the classic Iris dataset: 150 samples, 4 features (sepal/petal length & width), 3 species
iris = datasets.load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# Standardize features before PCA: subtract the mean and scale to unit variance
# PCA is sensitive to scale — without this, features measured in larger units dominate
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reduce 4 features down to 2 principal components for 2D visualisation
# PCA finds the 2 orthogonal directions of maximum variance in the 4D feature space
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))

colors = ['navy', 'turquoise', 'darkorange']
lw = 1

# Plot each species as a separate scatter series so they appear in the legend
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], color=color, s=50, ec='k', alpha=0.7, lw=lw,
                label=target_name)

plt.title('PCA 2-dimensional reduction of IRIS dataset')
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(loc='best', shadow=False, scatterpoints=1)
# plt.grid(True)
# plt.show()

# Total variance retained by the 2 components (closer to 1.0 means less information lost)
percentage = sum(pca.explained_variance_ratio_)
print(f"Variance explained by 2 PCs: {percentage:.2%}")
