import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from sklearn import preprocessing

# Load the fuel consumption dataset from IBM Cloud Object Storage
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
df = pd.read_csv(url)

# Drop non-numeric and categorical columns that won't be used as features
df = df.drop(['MODELYEAR', 'MAKE', 'MODEL', 'VEHICLECLASS', 'TRANSMISSION', 'FUELTYPE'], axis=1)

# Print the correlation matrix to see how strongly each feature relates to the target
print(df.corr())

# Drop highly correlated or redundant fuel consumption columns
# Keeping only ENGINESIZE and FUELCONSUMPTION_COMB_MPG as features, and CO2EMISSIONS as the target
df = df.drop(['CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB'], axis=1)

# Visualise pairwise relationships between the remaining columns
axes = pd.plotting.scatter_matrix(df, alpha=0.2)
for ax in axes.flatten():
    ax.xaxis.label.set_rotation(90)
    ax.yaxis.label.set_rotation(0)
    ax.yaxis.label.set_ha('right')
plt.tight_layout()
plt.gcf().subplots_adjust(wspace=0, hspace=0)
plt.show()

# Separate features (ENGINESIZE and FUELCONSUMPTION_COMB_MPG) from the target (CO2EMISSIONS)
X = df.iloc[:, [0, 1]].to_numpy()
y = df.iloc[:, [2]].to_numpy()

# Standardize features so both are on the same scale before fitting
# StandardScaler: (x - mean) / std  →  mean=0, std=1
std_scaler = preprocessing.StandardScaler()
X_std = std_scaler.fit_transform(X)

pd.DataFrame(X_std).describe().round(2)

# Split into 80% training and 20% test data
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.2, random_state=42)

from sklearn import linear_model

# Create and train the multiple linear regression model on the two standardized features
regressor = linear_model.LinearRegression()
regressor.fit(X_train, y_train)

# coef_ now has two values, one per feature
coef_ = regressor.coef_
intercept_ = regressor.intercept_

print('Coefficients: ', coef_)
print('Intercept: ', intercept_)

# Convert the standardized coefficients back to the original feature scale
# This lets us interpret the effect of each feature in its original units
means_ = std_scaler.mean_
std_devs_ = np.sqrt(std_scaler.var_)

coef_original = coef_ / std_devs_
intercept_original = intercept_ - np.sum((means_ * coef_) / std_devs_)

print('Coefficients (original scale): ', coef_original)
print('Intercept (original scale): ', intercept_original)

# Extract the two test-set feature columns for 3D plotting
X1 = X_test[:, 0] if X_test.ndim > 1 else X_test
X2 = X_test[:, 1] if X_test.ndim > 1 else np.zeros_like(X1)

# Build a mesh grid over the feature space to draw the regression plane
x1_surf, x2_surf = np.meshgrid(np.linspace(X1.min(), X1.max(), 100),
                               np.linspace(X2.min(), X2.max(), 100))
y_surf = intercept_ + coef_[0, 0] * x1_surf + coef_[0, 1] * x2_surf

# Colour test points by whether they fall above or below the regression plane
y_pred = regressor.predict(X_test.reshape(-1, 1)) if X_test.ndim == 1 else regressor.predict(X_test)
above_plane = y_test >= y_pred
below_plane = y_test < y_pred
above_plane = above_plane[:, 0]
below_plane = below_plane[:, 0]

# Plot the test data points and the fitted regression plane in 3D
fig = plt.figure(figsize=(20, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X1[above_plane], X2[above_plane], y_test[above_plane], label="Above Plane", s=70, alpha=.7, ec='k')
ax.scatter(X1[below_plane], X2[below_plane], y_test[below_plane], label="Below Plane", s=50, alpha=.3, ec='k')
ax.plot_surface(x1_surf, x2_surf, y_surf, color='k', alpha=0.21, label='plane')

ax.view_init(elev=10)
ax.legend(fontsize='x-large', loc='upper center')
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_box_aspect(None, zoom=0.75)
ax.set_xlabel('ENGINESIZE', fontsize='xx-large')
ax.set_ylabel('FUELCONSUMPTION', fontsize='xx-large')
ax.set_zlabel('CO2 Emissions', fontsize='xx-large')
ax.set_title('Multiple Linear Regression of CO2 Emissions', fontsize='xx-large')
plt.tight_layout()
plt.show()

# Plot ENGINESIZE vs CO2EMISSIONS with the fitted line for that feature alone
plt.scatter(X_train[:, 0], y_train, color='blue')
plt.plot(X_train[:, 0], coef_[0, 0] * X_train[:, 0] + intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

# Plot FUELCONSUMPTION_COMB_MPG vs CO2EMISSIONS with the fitted line for that feature alone
plt.scatter(X_train[:, 1], y_train, color='blue')
plt.plot(X_train[:, 1], coef_[0, 1] * X_train[:, 1] + intercept_[0], '-r')
plt.xlabel("FUELCONSUMPTION_COMB_MPG")
plt.ylabel("Emission")
plt.show()

# Fit a simple (single-feature) regression on ENGINESIZE only for comparison
X_train_1 = X_train[:, 0]
regressor_1 = linear_model.LinearRegression()
regressor_1.fit(X_train_1.reshape(-1, 1), y_train)
coef_1 = regressor_1.coef_[0]
intercept_1 = regressor_1.intercept_

print('Single-feature Coefficients: ', coef_1)
print('Single-feature Intercept: ', intercept_1)
