import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# Load the customer segmentation dataset from IBM Cloud Object Storage
cust_df = pd.read_csv(
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%204/data/Cust_Segmentation.csv")

# Drop the 'Address' column — it is a non-numeric identifier, not a feature
cust_df = cust_df.drop('Address', axis=1)

# Remove rows with missing values — K-Means cannot handle NaNs
cust_df = cust_df.dropna()

# Build the feature matrix, skipping the first column (Customer ID is not a feature)
X = cust_df.values[:, 1:]

# Standardize features so that variables measured in different units (e.g. age vs. income)
# contribute equally to the distance calculation
Clus_dataSet = StandardScaler().fit_transform(X)

# Fit K-Means with k=3 customer segments
# k-means++ initialization spreads starting centroids to avoid poor local minima
clusterNum = 3
k_means = KMeans(init="k-means++", n_clusters=clusterNum, n_init=12)
k_means.fit(Clus_dataSet)

# Retrieve the cluster assignment (0, 1, or 2) for each customer
labels = k_means.labels_

# Append cluster labels back to the original dataframe for analysis
cust_df["Clus_km"] = labels

# Print the mean of each feature per cluster to characterise each customer segment
cust_df.groupby('Clus_km').mean()

# 2D scatter plot: Age vs Income, with bubble size proportional to Education level
# Color indicates the assigned cluster
area = np.pi * (X[:, 1]) ** 2
plt.scatter(X[:, 0], X[:, 3], s=area, c=labels.astype(float), cmap='tab10', ec='k', alpha=0.5)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)
#plt.show()

# Interactive 3D scatter plot: Education (x), Age (y), Income (z), coloured by cluster
fig = px.scatter_3d(x=X[:, 1], y=X[:, 0], z=X[:, 3],
                    color=labels.astype(float), opacity=0.7)
px.scatter_3d(X, x=1, y=0, z=3, opacity=0.7, color=labels.astype(float))

fig.update_traces(marker=dict(size=5, line=dict(width=.25)), showlegend=False)
fig.update_layout(coloraxis_showscale=False, width=1000, height=800, scene=dict(
    xaxis=dict(title='Education'),
    yaxis=dict(title='Age'),
    zaxis=dict(title='Income')
))
#fig.show()

# Pairwise scatter plot matrix of Age, Education, and Income coloured by cluster
# KDE plots on the diagonal show the distribution of each variable per segment
cust_df_sub = cust_df[['Age', 'Edu', 'Income', 'Clus_km']].copy()
sns.pairplot(cust_df_sub, hue='Clus_km', palette='viridis', diag_kind='kde')
plt.suptitle('Pairwise Scatter Plot with K-means Clusters', y=1.02)
plt.show()
