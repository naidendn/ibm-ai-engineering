import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def regression_results(y_true, y_pred, regr_type):
    # Explained variance: proportion of variance in y accounted for by the model
    # Similar to R² but does not penalise a systematic bias in the predictions
    ev = explained_variance_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print('Evaluation metrics for ' + regr_type + ' Linear Regression')
    print('explained_variance: ', round(ev, 4))
    print('r2: ', round(r2, 4))
    print('MAE: ', round(mae, 4))
    print('MSE: ', round(mse, 4))
    print('RMSE: ', round(np.sqrt(mse), 4))
    print()


# =============================================================================
# Part 1: outlier sensitivity — 1 feature, 1,000 samples
# =============================================================================

# Generate a clean 1-D linear dataset: y = 4 + 3x + noise
# random_state=42 makes the simulation reproducible
noise = 1
np.random.seed(42)
X = 2 * np.random.rand(1000, 1)
y = 4 + 3 * X + noise * np.random.randn(1000, 1)  # noisy observations
y_ideal = 4 + 3 * X                                # noise-free ground truth

# Inject a small number of extreme outliers only in the high-X region
# to see how each regularisation method handles them
threshold = 1.5           # only modify samples where X > 1.5
num_outliers = 5          # how many points to corrupt
y_outlier = pd.Series(y.reshape(-1).copy())
outlier_indices = np.where(X.flatten() > threshold)[0]
selected_indices = np.random.choice(outlier_indices, num_outliers, replace=False)

# Shift the chosen targets up by 50–100 units — far outside the normal range
y_outlier[selected_indices] += np.random.uniform(50, 100, num_outliers)

# Visualise the clean data alongside the noise-free ideal line
# (outliers are not shown here — this is the baseline before corruption)
plt.figure(figsize=(12, 6))
plt.scatter(X, y, alpha=0.4, ec='k', label='Original Data without Outliers')
plt.plot(X, y_ideal, linewidth=3, color='g', label='Ideal, noise free data')
plt.xlabel('Feature (X)')
plt.ylabel('Target (y)')
plt.title('')
plt.legend()
plt.show()

# --- Fit all three models on the outlier-corrupted target ---
# OLS has no penalty, so its coefficients are pulled heavily toward the outliers
lin_reg = LinearRegression()
lin_reg.fit(X, y_outlier)
y_outlier_pred_lin = lin_reg.predict(X)

# Ridge (L2 penalty): adds λ * Σwᵢ² to the loss — shrinks all coefficients toward
# zero proportionally, reducing the influence of outliers without zeroing any feature
ridge_reg = Ridge(alpha=1)
ridge_reg.fit(X, y_outlier)
y_outlier_pred_ridge = ridge_reg.predict(X)

# Lasso (L1 penalty): adds λ * Σ|wᵢ| to the loss — can shrink coefficients to exactly
# zero, effectively performing feature selection; also more robust to outliers than OLS
lasso_reg = Lasso(alpha=.2)
lasso_reg.fit(X, y_outlier)
y_outlier_pred_lasso = lasso_reg.predict(X)

# Evaluate against the clean target y (not the corrupted one) to measure true error
regression_results(y, y_outlier_pred_lin, 'Ordinary')
regression_results(y, y_outlier_pred_ridge, 'Ridge')
regression_results(y, y_outlier_pred_lasso, 'Lasso')

# Overlay all three regression lines to see how much each was pulled by the outliers
plt.figure(figsize=(12, 6))
plt.scatter(X, y, alpha=0.4, ec='k', label='Original Data')
plt.plot(X, y_ideal, linewidth=2, color='k', label='Ideal, noise free data')
plt.plot(X, y_outlier_pred_lin, linewidth=5, label='Linear Regression')
plt.plot(X, y_outlier_pred_ridge, linestyle='--', linewidth=2, label='Ridge Regression')
plt.plot(X, y_outlier_pred_lasso, linewidth=2, label='Lasso Regression')
plt.xlabel('Feature (X)')
plt.ylabel('Target (y)')
plt.title('Comparison of Predictions with Outliers')
plt.legend()
plt.show()


# =============================================================================
# Part 2: high-dimensional sparse setting — 100 features, only 10 informative
# =============================================================================

# make_regression builds a controlled sparse problem:
# only 10 of the 100 features actually contribute to y;
# the rest have ideal coefficients of 0 — perfect for demonstrating Lasso's
# ability to recover sparsity by driving irrelevant coefficients to zero
X, y, ideal_coef = make_regression(n_samples=100, n_features=100, n_informative=10, noise=10, random_state=42,
                                   coef=True)

# ideal_predictions uses only the true generative coefficients (no noise)
# — the ceiling a model could theoretically achieve
ideal_predictions = X @ ideal_coef

X_train, X_test, y_train, y_test, ideal_train, ideal_test = train_test_split(X, y, ideal_predictions, test_size=0.3,
                                                                             random_state=42)

# alpha controls regularisation strength:
# Lasso alpha=0.1 — light penalty, still aggressive enough to zero out most irrelevant features
# Ridge alpha=1.0 — moderate L2 shrinkage; cannot zero features but reduces their magnitude
lasso = Lasso(alpha=0.1)
ridge = Ridge(alpha=1.0)
linear = LinearRegression()

lasso.fit(X_train, y_train)
ridge.fit(X_train, y_train)
linear.fit(X_train, y_train)

y_pred_linear = linear.predict(X_test)
y_pred_ridge = ridge.predict(X_test)
y_pred_lasso = lasso.predict(X_test)

regression_results(y_test, y_pred_linear, 'Ordinary')
regression_results(y_test, y_pred_ridge, 'Ridge')
regression_results(y_test, y_pred_lasso, 'Lasso')

# --- Actual vs Predicted scatter plots (top row) ---
# Points on the dashed diagonal = perfect predictions;
# scatter around it = prediction error
fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)

axes[0, 0].scatter(y_test, y_pred_linear, color="red", label="Linear")
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
axes[0, 0].set_title("Linear Regression")
axes[0, 0].set_xlabel("Actual", )
axes[0, 0].set_ylabel("Predicted", )

axes[0, 2].scatter(y_test, y_pred_lasso, color="blue", label="Lasso")
axes[0, 2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
axes[0, 2].set_title("Lasso Regression", )
axes[0, 2].set_xlabel("Actual", )

axes[0, 1].scatter(y_test, y_pred_ridge, color="green", label="Ridge")
axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
axes[0, 1].set_title("Ridge Regression", )
axes[0, 1].set_xlabel("Actual", )

axes[0, 2].scatter(y_test, y_pred_lasso, color="blue", label="Lasso")
axes[0, 2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
axes[0, 2].set_title("Lasso Regression", )
axes[0, 2].set_xlabel("Actual", )

# --- Line plots: model predictions vs actual test values (bottom row) ---
# Closer overlap between the dashed prediction line and the solid actual line = better fit
axes[1, 0].plot(y_test, label="Actual", lw=2)
axes[1, 0].plot(y_pred_linear, '--', lw=2, color='red', label="Linear")
axes[1, 0].set_title("Linear vs Ideal", )
axes[1, 0].legend()

axes[1, 1].plot(y_test, label="Actual", lw=2)
# axes[1,1].plot(ideal_test, '--', label="Ideal", lw=2, color="purple")
axes[1, 1].plot(y_pred_ridge, '--', lw=2, color='green', label="Ridge")
axes[1, 1].set_title("Ridge vs Ideal", )
axes[1, 1].legend()

axes[1, 2].plot(y_test, label="Actual", lw=2)
axes[1, 2].plot(y_pred_lasso, '--', lw=2, color='blue', label="Lasso")
axes[1, 2].set_title("Lasso vs Ideal", )
axes[1, 2].legend()

plt.tight_layout()
plt.show()

# --- Learned coefficients vs the ideal (true) coefficients ---
# OLS: unrestricted, can wildly overfit the 90 noise features
# Ridge: all 100 coefficients are shrunk but remain non-zero
# Lasso: most coefficients are driven to exactly 0 — ideally keeping only the 10 informative ones
linear_coeff = linear.coef_
ridge_coeff = ridge.coef_
lasso_coeff = lasso.coef_

x_axis = np.arange(len(linear_coeff))
x_labels = np.arange(min(x_axis), max(x_axis), 10)
plt.figure(figsize=(12, 6))

plt.scatter(x_axis, ideal_coef, label='Ideal', color='blue', ec='k', alpha=0.4)
plt.bar(x_axis - 0.25, linear_coeff, width=0.25, label='Linear Regression', color='blue')
plt.bar(x_axis, ridge_coeff, width=0.25, label='Ridge Regression', color='green')
plt.bar(x_axis + 0.25, lasso_coeff, width=0.25, label='Lasso Regression', color='red')

plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')
plt.title('Comparison of Model Coefficients')
plt.xticks(x_labels)
plt.legend()
plt.show()

# --- Coefficient residuals: ideal_coef − learned_coef ---
# A bar near zero means the model recovered the true coefficient accurately
# Lasso is plotted as a line to distinguish it visually; its residuals
# are near-zero for noise features (coefficient = 0 ≈ ideal) and
# small for informative features (mild shrinkage bias)
x_axis = np.arange(len(linear_coeff))

plt.figure(figsize=(12, 6))

plt.bar(x_axis - 0.25, ideal_coef - linear_coeff, width=0.25, label='Linear Regression', color='blue')
plt.bar(x_axis, ideal_coef - ridge_coeff, width=0.25, label='Ridge Regression', color='green')
# plt.bar(x_axis + 0.25, ideal_coef - lasso_coeff, width=0.25, label='Lasso Regression', color='red')
plt.plot(x_axis, ideal_coef - lasso_coeff, label='Lasso Regression', color='red')

plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')
plt.title('Comparison of Model Coefficient Residuals')
plt.xticks(x_labels)
plt.legend()
plt.show()
