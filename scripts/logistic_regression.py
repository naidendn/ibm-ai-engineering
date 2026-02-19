import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import log_loss
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# Load the customer churn dataset from IBM Cloud Object Storage
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv"
churn_df = pd.read_csv(url)

# Keep only the features we want to train on, plus the target column
churn_df = churn_df[['tenure', 'age', 'address', 'ed', 'equip', 'churn']]

# Ensure the target is an integer (0 = did not churn, 1 = churned)
churn_df['churn'] = churn_df['churn'].astype('int')

# Separate features (X) from the target label (y)
X = np.asarray(churn_df[['tenure', 'age', 'address', 'ed', 'equip']])
y = np.asarray(churn_df['churn'])

# Standardize features: subtract the mean and scale to unit variance
# This prevents features with larger ranges from dominating the model
X_norm = StandardScaler().fit(X).transform(X)

# Split into 80% training and 20% test data
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=4)

# Train the logistic regression model on the training set
LR = LogisticRegression().fit(X_train, y_train)

# Predict binary class labels for the test set
yhat = LR.predict(X_test)

# Predict class probabilities — needed for log loss calculation
# Shape is (n_samples, 2): column 0 = P(not churn), column 1 = P(churn)
yhat_prob = LR.predict_proba(X_test)

# Extract the learned coefficient for each feature
# A positive coefficient means the feature increases the probability of churn
coefficients = pd.Series(LR.coef_[0], index=churn_df.columns[:-1])
coefficients.sort_values().plot(kind='barh')
plt.title("Feature Coefficients in Logistic Regression Churn Model")
plt.xlabel("Coefficient Value")
# plt.show()

# Log loss measures how well the predicted probabilities match the true labels
# Lower is better; 0 would be a perfect probabilistic classifier
print(log_loss(y_test, yhat_prob))
