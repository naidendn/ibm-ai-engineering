import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load the Australian weather dataset from IBM Cloud Object Storage
# Contains daily weather observations across many Australian weather stations
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/_0eYOqji3unP1tDNKWZMjg/weatherAUS-2.csv"
df = pd.read_csv(url)
df.head()

# Drop rows with any missing values — keeps preprocessing simple for this exercise
# In production you would impute instead, to avoid discarding potentially useful data
df = df.dropna()
df.info()

# Rename columns to reflect what they actually mean at prediction time:
# 'RainToday' in the raw data = did it rain on this observation day (i.e. yesterday relative to tomorrow)
# 'RainTomorrow' = did it rain the following day — this becomes our target 'RainToday'
df = df.rename(columns={'RainToday': 'RainYesterday',
                        'RainTomorrow': 'RainToday'
                        })

# Restrict to Melbourne-area stations to reduce geographic variance in the model
# and keep the dataset manageable for a single-region classifier
df = df[df.Location.isin(['Melbourne', 'MelbourneAirport', 'Watsonia', ])]
df.info()


def date_to_season(date):
    # Map calendar month to Southern Hemisphere seasons
    # Australia's seasons are reversed relative to the Northern Hemisphere:
    # Summer = Dec–Feb, Autumn = Mar–May, Winter = Jun–Aug, Spring = Sep–Nov
    month = date.month
    if (month == 12) or (month == 1) or (month == 2):
        return 'Summer'
    elif (month == 3) or (month == 4) or (month == 5):
        return 'Autumn'
    elif (month == 6) or (month == 7) or (month == 8):
        return 'Winter'
    elif (month == 9) or (month == 10) or (month == 11):
        return 'Spring'


# Parse dates and derive a Season feature — seasonality is a strong predictor of rainfall
df['Date'] = pd.to_datetime(df['Date'])
df['Season'] = df['Date'].apply(date_to_season)

# Drop the raw Date column; the model uses Season instead (a categorical signal)
df = df.drop(columns=["Date"])

X = df.drop('RainToday', axis=1)
y = df['RainToday']  # binary: 'Yes' / 'No'

# stratify=y maintains the rain/no-rain class balance in both splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Detect column types from the training set only (avoids leakage from test set)
numeric_features = X_train.select_dtypes(include=['number']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

# Standardise numeric features — not strictly required for Random Forest (tree-based),
# but needed later when we swap in Logistic Regression
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

# One-hot encode categoricals; handle_unknown='ignore' silently drops unseen categories at inference
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

# ColumnTransformer applies each sub-pipeline to its own set of columns in parallel
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Full pipeline: preprocessing → Random Forest
# The pipeline ensures the preprocessor is fitted only on training folds during CV
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# =============================================================================
# Part 1: Random Forest with GridSearchCV
# =============================================================================

# Hyperparameter grid (2×3×2 = 12 combinations total)
# n_estimators: more trees → lower variance, higher compute cost
# max_depth: None = full-depth trees; limiting depth regularises and speeds up training
# min_samples_split: higher values prevent the tree from learning noise in small leaf nodes
param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5]
}

# StratifiedKFold preserves the rainfall class ratio across all 5 folds
cv = StratifiedKFold(n_splits=5, shuffle=True)

grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cv, scoring='accuracy', verbose=2)
grid_search.fit(X_train, y_train)

print("\nBest parameters found: ", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

# Evaluate the best model on the held-out test set
test_score = grid_search.score(X_test, y_test)
print("Test set score: {:.2f}".format(test_score))

y_pred = grid_search.predict(X_test)

# Per-class precision, recall, and F1 — important here because rainfall is imbalanced
# (more 'No' days than 'Yes' days), so accuracy alone can be misleading
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ConfusionMatrixDisplay is a convenience wrapper that labels axes automatically
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# Reconstruct the full feature name list after one-hot expansion
# so each importance value maps to a named feature
feature_names = numeric_features + list(grid_search.best_estimator_['preprocessor']
                                        .named_transformers_['cat']
                                        .named_steps['onehot']
                                        .get_feature_names_out(categorical_features))

# feature_importances_ = mean decrease in impurity (Gini) across all trees for each feature
feature_importances = grid_search.best_estimator_['classifier'].feature_importances_

importance_df = pd.DataFrame({'Feature': feature_names,
                              'Importance': feature_importances
                             }).sort_values(by='Importance', ascending=False)

# Display only the top N features to keep the chart readable
N = 20
top_features = importance_df.head(N)

plt.figure(figsize=(10, 6))
plt.barh(top_features['Feature'], top_features['Importance'], color='skyblue')
plt.gca().invert_yaxis()  # most important feature at the top
plt.title(f'Top {N} Most Important Features in predicting whether it will rain today')
plt.xlabel('Importance Score')
plt.show()

# =============================================================================
# Part 2: Swap in Logistic Regression and re-run GridSearchCV
# =============================================================================

# Replace the classifier in the existing pipeline in-place
pipeline.set_params(classifier=LogisticRegression(random_state=42))

# Point the GridSearchCV object at the updated pipeline
grid_search.estimator = pipeline

# solver='liblinear' is efficient for small-to-medium datasets and supports both L1 and L2
# penalty: L1 can zero out irrelevant feature coefficients; L2 shrinks all coefficients smoothly
# class_weight='balanced' up-weights the minority class ('Yes' rain days) automatically
param_grid = {
    # 'classifier__n_estimators': [50, 100],
    # 'classifier__max_depth': [None, 10, 20],
    # 'classifier__min_samples_split': [2, 5],
    'classifier__solver' : ['liblinear'],
    'classifier__penalty': ['l1', 'l2'],
    'classifier__class_weight' : [None, 'balanced']
}

grid_search.param_grid = param_grid

# Fit the updated pipeline with LogisticRegression
grid_search.fit(X_train, y_train)

y_pred = grid_search.predict(X_test)

print(classification_report(y_test, y_pred))

# Confusion matrix for the Logistic Regression model — compare with the Random Forest above
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')
plt.title('Titanic Classification Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()
