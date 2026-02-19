import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')

# Load the obesity level prediction dataset from IBM Cloud Object Storage
file_path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/GkDzb7bWrtvGXdPOfk6CIg/Obesity-level-prediction-dataset.csv"
data = pd.read_csv(file_path)
data.head()

# Visualise the distribution of obesity levels to understand class balance
sns.countplot(y='NObeyesdad', data=data)
plt.title('Distribution of Obesity Levels')
#plt.show()

# Verify there are no missing values before preprocessing
print(data.isnull().sum())

# Explore the data types and basic statistics
print(data.info())
print(data.describe())

# --- Step 1: Standardize continuous (float) features ---
# StandardScaler: (x - mean) / std  →  each feature has mean=0, std=1
# This prevents features with large ranges from dominating the model
continuous_columns = data.select_dtypes(include=['float64']).columns.tolist()
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[continuous_columns])
scaled_df = pd.DataFrame(scaled_features, columns=scaler.get_feature_names_out(continuous_columns))

# Replace original continuous columns with their standardized versions
scaled_data = pd.concat([data.drop(columns=continuous_columns), scaled_df], axis=1)

# --- Step 2: One-hot encode categorical features (excluding the target) ---
# Identify string columns and remove the target so it is not encoded as a feature
categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
if 'NObeyesdad' in categorical_columns:
    categorical_columns.remove('NObeyesdad')

# drop='first' drops one dummy per category to avoid multicollinearity
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_features = encoder.fit_transform(data[categorical_columns])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))

# --- Step 3: Build the feature matrix from processed columns only ---
# CRITICAL: target string values (e.g. 'Normal_Weight') must never appear in X
X = pd.concat([scaled_df, encoded_df], axis=1)

# --- Step 4: Encode the target as integer class codes ---
# pandas .cat.codes assigns a unique integer to each obesity level string
y = data['NObeyesdad'].astype('category').cat.codes

# Split into 80% training and 20% test data; stratify preserves class ratios
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- One-vs-All (OvA) classifier ---
# Trains one binary "this class vs. rest" classifier per class
# The class with the highest decision score wins
model_ova = LogisticRegression(multi_class='ovr', max_iter=1000)
model_ova.fit(X_train, y_train)

# Predict the winning class for each test sample
y_pred_ova = model_ova.predict(X_test)

print("One-vs-All (OvA) Strategy")
print(f"Accuracy: {np.round(100*accuracy_score(y_test, y_pred_ova),2)}%")

# --- One-vs-One (OvO) classifier ---
# Trains one binary classifier for every pair of classes: C(n, 2) classifiers in total
# The class that wins the most pairwise duels is the final prediction
model_ovo = OneVsOneClassifier(LogisticRegression(max_iter=1000))
model_ovo.fit(X_train, y_train)

y_pred_ovo = model_ovo.predict(X_test)

print("One-vs-One (OvO) Strategy")
print(f"Accuracy: {np.round(100*accuracy_score(y_test, y_pred_ovo),2)}%")

# --- Feature importance for OvA ---
# Average the absolute coefficient magnitudes across all class-specific classifiers
# Higher mean absolute coefficient → feature has a stronger effect on classification
feature_importance = np.mean(np.abs(model_ova.coef_), axis=0)
plt.barh(X.columns, feature_importance)
plt.title("Feature Importance")
plt.xlabel("Importance")
#plt.show()

# --- Feature importance for OvO ---
# Each estimator in OvO is a binary classifier with its own coef_ vector
# Collect all coefficients, then average their absolute values across all binary classifiers
coefs = np.array([est.coef_[0] for est in model_ovo.estimators_])
feature_importance = np.mean(np.abs(coefs), axis=0)

plt.barh(X.columns, feature_importance)
plt.title("Feature Importance (One-vs-One)")
plt.xlabel("Importance")
#plt.show()
