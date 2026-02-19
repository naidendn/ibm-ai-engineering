import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the telecom customer dataset from IBM Cloud Object Storage
# Target column 'custcat' contains 4 customer categories (service tiers)
df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv')

# Separate features from the target label
X = df.drop('custcat', axis=1)
y = df['custcat']

# Standardize features: KNN is distance-based, so features with larger ranges
# would dominate the distance calculation without scaling
X_norm = StandardScaler().fit_transform(X)

# Split into 80% training and 20% test data
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=4)

# Evaluate every K from 1 to 100 to find the optimal number of neighbours
Ks = 100
acc = np.zeros((Ks))      # stores accuracy for each K
std_acc = np.zeros((Ks))  # stores standard deviation of accuracy for each K

for n in range(1, Ks + 1):
    # Train a KNN model with n neighbours and predict on the test set
    knn_model_n = KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train)
    yhat = knn_model_n.predict(X_test)

    # Record accuracy and the standard error of the accuracy estimate
    acc[n - 1] = accuracy_score(y_test, yhat)
    std_acc[n - 1] = np.std(yhat == y_test) / np.sqrt(yhat.shape[0])

# Plot accuracy vs K with a shaded ±1 standard deviation band
plt.plot(range(1, Ks + 1), acc, 'g')
plt.fill_between(range(1, Ks + 1), acc - 1 * std_acc, acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy value', 'Standard Deviation'))
plt.ylabel('Model Accuracy')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()

# Report the K value that produced the highest test accuracy
print("The best accuracy was with", acc.max(), "with k =", acc.argmax() + 1)
