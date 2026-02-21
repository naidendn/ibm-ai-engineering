import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load the fuel consumption dataset from IBM Cloud Object Storage
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
df = pd.read_csv(url)

# Keep only the relevant columns for this exercise
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]

# Define the single feature (X) and the target variable (y)
# We predict CO2 emissions from combined fuel consumption
X = cdf.FUELCONSUMPTION_COMB.to_numpy()
y = cdf.CO2EMISSIONS.to_numpy()

# Split into 80% training and 20% test data
# random_state=42 ensures the same split every run (reproducibility)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
# reshape(-1, 1) converts a 1D array to a 2D column vector, which sklearn requires for a single feature
regressor = linear_model.LinearRegression()
regressor.fit(X_train.reshape(-1, 1), y_train)

# coef_ is the slope (how much CO2 increases per unit of fuel consumption)
# intercept_ is the y-intercept (predicted CO2 when fuel consumption is 0)
print('Coefficients: ', regressor.coef_[0])
print('Intercept: ', regressor.intercept_)

# Plot the regression line over the training data to visually check the fit
plt.scatter(X_train, y_train, color='blue')
plt.plot(X_train, regressor.coef_ * X_train + regressor.intercept_, '-r')
plt.xlabel("Fuel Consumption (Comb)")
plt.ylabel("Emission")
plt.show()

# Plot the regression line over the test data to check generalisation
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, regressor.coef_ * X_test + regressor.intercept_, '-r')
plt.xlabel("Fuel Consumption (Comb)")
plt.ylabel("Emission")
plt.show()

# Generate predictions on the unseen test set
prediction = regressor.predict(X_test.reshape(-1, 1))

# Evaluate model performance with several regression metrics:
# MAE  - average absolute difference between predictions and actual values
# MSE  - average squared difference (penalises large errors more heavily)
# RMSE - square root of MSE, expressed in the same units as the target
# R²   - proportion of variance explained by the model (1.0 is perfect)
print("Mean absolute error: %.2f" % mean_absolute_error(y_test, prediction))
print("Mean squared error: %.2f" % mean_squared_error(y_test, prediction))
print("Root mean squared error: %.2f" % np.sqrt(mean_squared_error(y_test, prediction)))
print("R2-score: %.2f" % r2_score(y_test, prediction))
