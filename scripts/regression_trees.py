from __future__ import print_function

import warnings

import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

warnings.filterwarnings('ignore')

# Load the NYC taxi trip dataset from IBM Cloud Object Storage
url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/pu9kbeSaAtRZ7RxdJKX9_A/yellow-tripdata.csv'
raw_data = pd.read_csv(url)

# Plot correlation of each feature with the tip amount to identify useful predictors
correlation_values = raw_data.corr()['tip_amount'].drop('tip_amount')
correlation_values.plot(kind='barh', figsize=(10, 6))

# Extract the target variable (tip amount) as a float32 array
y = raw_data[['tip_amount']].values.astype('float32')

# Drop the target and columns that are not useful predictors
# (payment_type and VendorID are categorical IDs; improvement_surcharge and store_and_fwd_flag are admin fields)
proc_data = raw_data.drop(['tip_amount', 'improvement_surcharge', 'payment_type', 'VendorID', 'store_and_fwd_flag'],
                          axis=1)

# Build the feature matrix
X = proc_data.values

# Apply L1 normalisation row-wise so each sample vector sums to 1
# This makes samples with very different total amounts comparable
X = normalize(X, axis=1, norm='l1', copy=False)

# Split into 70% training and 30% test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.tree import DecisionTreeRegressor

# Create a decision tree regressor limited to depth 4 to avoid overfitting
# criterion='squared_error' minimises MSE at each split (equivalent to variance reduction)
# random_state ensures reproducible tree splits
dt_reg = DecisionTreeRegressor(criterion='squared_error',
                               max_depth=4,
                               random_state=35)

# Fit the tree to the training data
dt_reg.fit(X_train, y_train)

# Predict tip amounts on the unseen test set
y_pred = dt_reg.predict(X_test)

# MSE: average squared difference between predicted and actual tip amounts
mse_score = mean_squared_error(y_test, y_pred)
print('MSE score : {0:.3f}'.format(mse_score))

# R²: proportion of variance in tip_amount explained by the model (1.0 = perfect)
r2_score = dt_reg.score(X_test, y_test)
print('R^2 score : {0:.3f}'.format(r2_score))
