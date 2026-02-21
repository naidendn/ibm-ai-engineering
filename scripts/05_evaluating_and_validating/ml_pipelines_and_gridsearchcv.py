import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset: 150 samples, 4 features (sepal/petal length & width), 3 classes
data = load_iris()
X, y = data.data, data.target
labels = data.target_names  # ['setosa', 'versicolor', 'virginica']

# --- Build a Pipeline with fixed hyperparameters ---
# A Pipeline chains preprocessing and modelling steps so that:
#   1. fit() on training data applies each step in order
#   2. predict() / score() applies the same fitted transformations to new data automatically
#      — this prevents data leakage because the scaler never sees the test set during fitting
pipeline = Pipeline([
    ('scaler', StandardScaler()),        # Step 1: standardise features to mean=0, std=1
    ('pca', PCA(n_components=2),),       # Step 2: reduce 4 features → 2 principal components
    ('knn', KNeighborsClassifier(n_neighbors=5, ))  # Step 3: classify using 5 nearest neighbours
])

# stratify=y ensures each split preserves the original class proportions
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# fit() applies StandardScaler → PCA on X_train, then trains KNN on the transformed data
pipeline.fit(X_train, y_train)

# score() transforms X_test through the same fitted scaler and PCA before classifying
test_score = pipeline.score(X_test, y_test)
print(f"{test_score:.3f}")

y_pred = pipeline.predict(X_test)

conf_matrix = confusion_matrix(y_test, y_pred)

# Confusion matrix: rows = actual class, columns = predicted class
# Diagonal = correct predictions; off-diagonal = misclassifications
plt.figure()
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d',
            xticklabels=labels, yticklabels=labels)
plt.title('Classification Pipeline Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# --- GridSearchCV: find the best hyperparameters automatically ---
# Rebuild the pipeline without fixing any hyperparameters yet;
# GridSearchCV will try every combination in param_grid
pipeline = Pipeline(
    [('scaler', StandardScaler()),
     ('pca', PCA()),
     ('knn', KNeighborsClassifier())
     ]
)

# param_grid uses the step name as a prefix (e.g. 'pca__n_components')
# to target hyperparameters inside specific pipeline steps
param_grid = {'pca__n_components': [2, 3],
              'knn__n_neighbors': [3, 5, 7]
              }

# StratifiedKFold with 5 splits: each fold preserves class proportions,
# which is important for balanced evaluation on small or imbalanced datasets
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# GridSearchCV exhaustively tries all 2×3=6 parameter combinations,
# evaluating each with 5-fold cross-validation on the training set
# verbose=2 prints progress for each combination during fitting
best_model = GridSearchCV(estimator=pipeline,
                          param_grid=param_grid,
                          cv=cv,
                          scoring='accuracy',
                          verbose=2
                          )

best_model.fit(X_train, y_train)

# Evaluate the best found model on the held-out test set
test_score = best_model.score(X_test, y_test)
print(test_score)

# best_params_ shows which combination of hyperparameters performed best in cross-validation
print(best_model.best_params_)

y_pred = best_model.predict(X_test)

# Confusion matrix for the tuned model — compare with the fixed-params version above
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d',
            xticklabels=labels, yticklabels=labels)
plt.title('KNN Classification Testing Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()
