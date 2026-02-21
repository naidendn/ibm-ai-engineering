import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load the Titanic dataset from seaborn (891 passengers, mix of numeric and categorical features)
# Target: survived (0 = died, 1 = survived)
titanic = sns.load_dataset('titanic')
titanic.head()

# Select a subset of features — a mix of numeric and categorical columns
# sibsp = number of siblings/spouses aboard; parch = number of parents/children aboard
features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'class', 'who', 'adult_male', 'alone']
target = 'survived'

X = titanic[features]
y = titanic[target]

# stratify=y preserves the survived/died ratio in both train and test splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Detect numeric and categorical columns automatically from the training set dtypes
# (never from the full dataset, to avoid leakage)
numerical_features = X_train.select_dtypes(include=['number']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

# --- Preprocessing sub-pipelines ---
# Numerical: fill missing values with the column median, then standardise to mean=0, std=1
# Median imputation is preferred over mean when outliers are present (e.g. extreme fares)
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical: fill missing values with the most frequent category, then one-hot encode
# handle_unknown='ignore' silently drops unseen categories at inference time
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# ColumnTransformer applies each sub-pipeline to its respective columns in parallel
# and concatenates the results into a single feature matrix
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Full pipeline: preprocessing → Random Forest classifier
# Wrapping the classifier in the pipeline ensures the preprocessor is
# refitted only on training folds during cross-validation
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# =============================================================================
# Part 1: Random Forest with GridSearchCV
# =============================================================================

# Hyperparameter grid for the Random Forest step (prefix 'classifier__')
# n_estimators: number of trees; more trees reduce variance but cost more compute
# max_depth: limits tree depth to prevent overfitting (None = grow until pure leaves)
# min_samples_split: minimum samples required to split a node — higher values regularise more
param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5]
}

# StratifiedKFold ensures each fold has the same survived/died ratio as the full training set
cv = StratifiedKFold(n_splits=5, shuffle=True)

# GridSearchCV tries all 2×3×2=12 combinations, evaluating each with 5-fold CV
model = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cv, scoring='accuracy', verbose=2)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# classification_report shows per-class precision, recall, and F1 for the best model
print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')
plt.title('Titanic Classification Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# Retrieve the one-hot encoded feature names from the fitted preprocessor
# needed to map each importance value back to a human-readable feature name
model.best_estimator_['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)

# feature_importances_ from Random Forest = mean decrease in impurity across all trees
feature_importances = model.best_estimator_['classifier'].feature_importances_

# Concatenate numeric feature names with the expanded one-hot encoded categorical names
feature_names = numerical_features + list(model.best_estimator_['preprocessor']
                                        .named_transformers_['cat']
                                        .named_steps['onehot']
                                        .get_feature_names_out(categorical_features))

importance_df = pd.DataFrame({'Feature': feature_names,
                              'Importance': feature_importances
                             }).sort_values(by='Importance', ascending=False)

# Horizontal bar chart: longer bar = stronger predictor of survival
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.gca().invert_yaxis()
plt.title('Most Important Features in predicting whether a passenger survived')
plt.xlabel('Importance Score')
plt.show()

test_score = model.score(X_test, y_test)
print(f"\nTest set accuracy: {test_score:.2%}")

# =============================================================================
# Part 2: Swap in Logistic Regression and re-run GridSearchCV
# =============================================================================

# Replace the classifier inside the existing pipeline object
# set_params() modifies the named step in-place without rebuilding the full pipeline
pipeline.set_params(classifier=LogisticRegression(random_state=42))

# Point the GridSearchCV object at the updated pipeline
model.estimator = pipeline

# Logistic Regression hyperparameters:
# solver='liblinear' supports both L1 and L2 penalties on small datasets
# penalty: L1 (Lasso-like, can zero coefficients) vs L2 (Ridge-like, shrinks all coefficients)
# class_weight='balanced' up-weights the minority class (died vs survived) to reduce bias
param_grid = {
    # 'classifier__n_estimators': [50, 100],
    # 'classifier__max_depth': [None, 10, 20],
    # 'classifier__min_samples_split': [2, 5],
    'classifier__solver' : ['liblinear'],
    'classifier__penalty': ['l1', 'l2'],
    'classifier__class_weight' : [None, 'balanced']
}

model.param_grid = param_grid

# Fit the updated pipeline with Logistic Regression
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Confusion matrix for the Logistic Regression model — compare with the Random Forest above
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(conf_matrix , annot=True, cmap='Blues', fmt='d')
plt.title('Titanic Classification Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# coef_[0] gives one coefficient per feature after one-hot expansion
# positive coefficient → increases odds of survival; negative → decreases odds
coefficients = model.best_estimator_.named_steps['classifier'].coef_[0]

# Reconstruct the full feature name list (same order as the preprocessor output)
numerical_feature_names = numerical_features
categorical_feature_names = (model.best_estimator_.named_steps['preprocessor']
                                     .named_transformers_['cat']
                                     .named_steps['onehot']
                                     .get_feature_names_out(categorical_features)
                            )
feature_names = numerical_feature_names + list(categorical_feature_names)

# Sort by absolute coefficient magnitude — large absolute value = strong effect on survival odds
# Note: feature_importances from the Random Forest is reused here for comparison purposes
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': feature_importances
}).sort_values(by='Coefficient', ascending=False, key=abs)  # Sort by absolute values

# Horizontal bar chart of coefficient magnitudes (absolute value strips the sign)
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Coefficient'].abs(), color='skyblue')
plt.gca().invert_yaxis()
plt.title('Feature Coefficient magnitudes for Logistic Regression model')
plt.xlabel('Coefficient Magnitude')
plt.show()

# Print test score
test_score = model.best_estimator_.score(..., ...)
print(f"\nTest set accuracy: {test_score:.2%}")
