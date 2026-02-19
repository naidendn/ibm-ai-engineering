from __future__ import print_function

import warnings

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings('ignore')

# Load the credit card fraud dataset from IBM Cloud Object Storage
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/creditcard.csv"
raw_data = pd.read_csv(url)

# Inspect the class distribution — fraud datasets are heavily imbalanced
# (very few fraudulent transactions compared to legitimate ones)
labels = raw_data.Class.unique()
sizes = raw_data.Class.value_counts().values

# Pie chart to visualise the class imbalance
fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct='%1.3f%%')
ax.set_title('Target Variable Value Counts')
# plt.show()

# Plot the correlation of each feature with the Class label to identify useful predictors
correlation_values = raw_data.corr()['Class'].drop('Class')
correlation_values.plot(kind='barh', figsize=(10, 6))

# Standardize columns 1–29 (the anonymized V1–V28 features plus Amount)
# This removes the mean and scales to unit variance so the SVM is not skewed by scale
raw_data.iloc[:, 1:30] = StandardScaler().fit_transform(raw_data.iloc[:, 1:30])
data_matrix = raw_data.values

# Select the 6 features most correlated with Class (identified earlier via correlation analysis)
# Excluding the Time column (index 0) as it is not predictive of fraud
X = data_matrix[:, [3, 10, 12, 14, 16, 17]]

# Target: 0 = legitimate transaction, 1 = fraudulent transaction
y = data_matrix[:, 30]

# Apply L1 normalisation so each sample vector sums to 1
# This further standardizes each row and can help with imbalanced classes
X = normalize(X, norm="l1")

# Split into 70% training and 30% test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Compute per-sample weights to counteract class imbalance:
# minority class (fraud) samples receive a higher weight during training
w_train = compute_sample_weight('balanced', y_train)

# Train a Decision Tree baseline model with sample weights to handle imbalance
# max_depth=4 limits complexity; random_state ensures reproducibility
dt = DecisionTreeClassifier(max_depth=4, random_state=35)
dt.fit(X_train, y_train, sample_weight=w_train)

# Train a Linear SVC (Support Vector Classifier) with balanced class weights
# loss='hinge' is the standard SVM margin loss
# fit_intercept=False is appropriate after normalisation
# class_weight='balanced' automatically adjusts weights inversely proportional to class frequencies
svm = LinearSVC(class_weight='balanced', random_state=31, loss="hinge", fit_intercept=False)
svm.fit(X_train, y_train)

# Get fraud probability scores from the decision tree (column 1 = P(fraud))
y_pred_dt = dt.predict_proba(X_test)[:, 1]

# ROC-AUC measures how well the model separates fraud from legitimate transactions
# 0.5 = random guessing, 1.0 = perfect separation
roc_auc_dt = roc_auc_score(y_test, y_pred_dt)
print('Decision Tree ROC-AUC score : {0:.3f}'.format(roc_auc_dt))

# LinearSVC does not output probabilities; use the raw decision function score instead
# The decision function gives the signed distance to the separating hyperplane
y_pred_svm = svm.decision_function(X_test)

roc_auc_svm = roc_auc_score(y_test, y_pred_svm)
print("SVM ROC-AUC score: {0:.3f}".format(roc_auc_svm))

# Retrieve the top 6 features by absolute correlation for reference
correlation_values = abs(raw_data.corr()['Class']).drop('Class')
correlation_values = correlation_values.sort_values(ascending=False)[0:6]
