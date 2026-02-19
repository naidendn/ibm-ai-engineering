import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Load the breast cancer dataset (built into scikit-learn)
# 569 samples, 30 numeric features derived from cell nucleus measurements
# Binary target: 0 = malignant, 1 = benign
data = load_breast_cancer()
X, y = data.data, data.target
labels = data.target_names        # ['malignant', 'benign']
feature_names = data.feature_names

# Print the full dataset description and class names for reference
print(data.DESCR)
print(data.target_names)

# Standardize all 30 features to mean=0, std=1
# Both KNN and SVM are sensitive to feature scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Add Gaussian noise to simulate a real-world noisy dataset ---
# This tests how robust each model is when the input data is imperfect
np.random.seed(42)
noise_factor = 0.5  # controls the strength of the noise (higher = noisier)
X_noisy = X_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X.shape)

# Load both versions into DataFrames for easy comparison and visualisation
df = pd.DataFrame(X_scaled, columns=feature_names)
df_noisy = pd.DataFrame(X_noisy, columns=feature_names)

print("Original Data (First 5 rows):")
df.head()

# Scatter plot comparing one feature before and after adding noise
# Points scattered away from the diagonal show the effect of noise on that feature
plt.figure(figsize=(12, 6))
plt.scatter(df[feature_names[5]], df_noisy[feature_names[5]], lw=5)
plt.title('Scaled feature comparison with and without noise')
plt.xlabel('Original Feature')
plt.ylabel('Noisy Feature')
plt.tight_layout()
plt.show()

# Split the noisy dataset into 70% training and 30% test data
X_train, X_test, y_train, y_test = train_test_split(X_noisy, y, test_size=0.3, random_state=42)

# --- Model initialisation ---
# KNN with k=5: classifies each point by majority vote among its 5 nearest neighbours
knn = KNeighborsClassifier(n_neighbors=5)

# SVM with a linear kernel and regularisation parameter C=1
# C controls the trade-off between maximising the margin and minimising misclassifications
svm = SVC(kernel='linear', C=1, random_state=42)

# Train both models on the noisy training data
knn.fit(X_train, y_train)
svm.fit(X_train, y_train)

# --- Test set predictions ---
y_pred_knn = knn.predict(X_test)
y_pred_svm = svm.predict(X_test)

# Overall fraction of test samples correctly classified
print(f"KNN Testing Accuracy: {accuracy_score(y_test, y_pred_knn):.3f}")
print(f"SVM Testing Accuracy: {accuracy_score(y_test, y_pred_svm):.3f}")

# Per-class precision, recall, and F1-score on the test set
# Precision: of all predicted positive, how many were actually positive?
# Recall:    of all actual positives, how many did the model catch?
# F1-score:  harmonic mean of precision and recall
print("\nKNN Testing Data Classification Report:")
print(classification_report(y_test, y_pred_knn))

print("\nSVM Testing Data Classification Report:")
print(classification_report(y_test, y_pred_svm))

# --- Test set confusion matrices ---
# Rows = actual class, Columns = predicted class
# Diagonal cells = correct predictions; off-diagonal = errors
conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(conf_matrix_knn, annot=True, cmap='Blues', fmt='d', ax=axes[0],
            xticklabels=labels, yticklabels=labels)
axes[0].set_title('KNN Testing Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

sns.heatmap(conf_matrix_svm, annot=True, cmap='Blues', fmt='d', ax=axes[1],
            xticklabels=labels, yticklabels=labels)
axes[1].set_title('SVM Testing Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.show()

# --- Training set predictions (to check for overfitting) ---
# If training accuracy is much higher than test accuracy, the model is overfitting
y_pred_train_knn = knn.predict(X_train)
y_pred_train_svm = svm.predict(X_train)

print(f"KNN Training Accuracy: {accuracy_score(y_train, y_pred_train_knn):.3f}")
print(f"SVM Training Accuracy: {accuracy_score(y_train, y_pred_train_svm):.3f}")

print("\nKNN Training Classification Report:")
print(classification_report(y_train, y_pred_train_knn))

print("\nSVM Training Classification Report:")
print(classification_report(y_train, y_pred_train_svm))

# --- Training set confusion matrices ---
# Comparing these with the test matrices reveals how well each model generalises
conf_matrix_knn = confusion_matrix(y_train, y_pred_train_knn)
conf_matrix_svm = confusion_matrix(y_train, y_pred_train_svm)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(conf_matrix_knn, annot=True, cmap='Blues', fmt='d', ax=axes[0],
            xticklabels=labels, yticklabels=labels)
axes[0].set_title('KNN Training Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

sns.heatmap(conf_matrix_svm, annot=True, cmap='Blues', fmt='d', ax=axes[1],
            xticklabels=labels, yticklabels=labels)
axes[1].set_title('SVM Training Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.show()
