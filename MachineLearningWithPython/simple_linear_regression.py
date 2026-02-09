import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Read the CSV with pandas
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
df = pd.read_csv(url)

# Get only 4 columns
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]

# Get engine size and co2 emissions as numpy arrays
X = cdf.FUELCONSUMPTION_COMB.to_numpy()
y = cdf.CO2EMISSIONS.to_numpy()

# Split training to test data 80/20. Random state keeps the distribution always the same
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define linear regression model. Reshape X array because .fit() expects a 2d array for features. Fit features (X) and target (y) into the model
regressor = linear_model.LinearRegression()
regressor.fit(X_train.reshape(-1, 1), y_train)

# Print coefficient - it's only 1 because we have 1 feature (engine size), and intercept - where do we start from (emissions when engine size is 0)
print('Coefficients: ', regressor.coef_[0])
print('Intercept: ', regressor.intercept_)

# Plot our regression line against the training data
plt.scatter(X_train, y_train, color='blue')
plt.plot(X_train, regressor.coef_ * X_train + regressor.intercept_, '-r')
plt.xlabel("Fuel Consumption (Comb)")
plt.ylabel("Emission")
plt.show()

# Plotting the regression model result over the test data
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, regressor.coef_ * X_test + regressor.intercept_, '-r')
plt.xlabel("Fuel Consumption (Comb)")
plt.ylabel("Emission")
plt.show()

# Use the predict method to make test predictions
prediction = regressor.predict(X_test.reshape(-1, 1))

# Evaluation
print("Mean absolute error: %.2f" % mean_absolute_error(y_test, prediction))
print("Mean squared error: %.2f" % mean_squared_error(y_test, prediction))
print("Root mean squared error: %.2f" % np.sqrt(mean_squared_error(y_test, prediction)))
print("R2-score: %.2f" % r2_score(y_test, prediction))

