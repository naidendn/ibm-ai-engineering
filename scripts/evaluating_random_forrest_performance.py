import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Load the California housing dataset (built into scikit-learn)
# ~20,640 samples; 8 numeric features derived from 1990 US Census block groups
# Target: median house value in units of $100,000 (e.g. 5.0 = $500k)
data = fetch_california_housing()
X, y = data.data, data.target

print(data.DESCR)

# Split data into 80% training / 20% test before fitting the model
# random_state=42 ensures the same split every run for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Plot the target distribution on the training set
# Multiply by 1e5 to convert from $100k units back to full dollar values
# Skewness > 0 means a long right tail: a few very expensive houses pull the mean up
plt.hist(1e5 * y_train, bins=30, color='lightblue', edgecolor='black')
plt.title(f'Median House Value Distribution\nSkewness: {skew(y_train):.2f}')
plt.xlabel('Median House Value')
plt.ylabel('Frequency')
plt.show()

# --- Initialise and fit the Random Forest Regressor ---
# n_estimators=100: an ensemble of 100 decision trees
# Each tree is trained on a random bootstrap sample of the training data
# At every split only a random subset of features is considered, which
# decorrelates the trees and reduces the overall variance of the ensemble
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)

# Generate predictions on the held-out test set
y_pred_test = rf_regressor.predict(X_test)

# --- Regression evaluation metrics ---
# MAE:  average absolute error — easy to interpret, in the same units as the target
# MSE:  penalises large errors more heavily than MAE (squared units)
# RMSE: square root of MSE — back in the original units, more interpretable than MSE
# R²:   proportion of variance in y explained by the model
#       (1.0 = perfect fit, 0.0 = predicting the mean every time)
mae = mean_absolute_error(y_test, y_pred_test)
mse = mean_squared_error(y_test, y_pred_test)
rmse = root_mean_squared_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# Scatter plot of actual vs predicted values
# A perfect model would place every point on the dashed diagonal (y = x)
# Spread around the line and skew at the extremes reveal systematic errors
plt.scatter(y_test, y_pred_test, alpha=0.5, color="blue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Random Forest Regression - Actual vs Predicted")
plt.show()

# --- Feature importances ---
# Importance = mean decrease in impurity (MSE for regression) across all splits
# in all trees where that feature was used; higher = more predictive power
importances = rf_regressor.feature_importances_
indices = np.argsort(importances)[::-1]  # sort from most to least important
features = data.feature_names

# Bar chart ordered from the most to the least important feature
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=45)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Feature Importances in Random Forest Regression")
plt.show()
