import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')

# Load the drug prescription dataset from IBM Cloud Object Storage
path = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv'
my_data = pd.read_csv(path)

# Print column types and non-null counts to understand the data shape
print(my_data.info())

# Encode categorical string columns as integers so the tree can split on them
# LabelEncoder maps each unique string to an integer (e.g. 'F'→0, 'M'→1)
label_encoder = LabelEncoder()
my_data['Sex'] = label_encoder.fit_transform(my_data['Sex'])
my_data['BP'] = label_encoder.fit_transform(my_data['BP'])
my_data['Cholesterol'] = label_encoder.fit_transform(my_data['Cholesterol'])

print(my_data)

# Check for missing values — decision trees can't handle NaNs
print(my_data.isnull().sum())

# Create a numeric version of the target for potential correlation analysis
custom_map = {'drugA': 0, 'drugB': 1, 'drugC': 2, 'drugX': 3, 'drugY': 4}
my_data['Drug_num'] = my_data['Drug'].map(custom_map)

#my_data = my_data.drop("Drug", axis=1)
#print(my_data)

#correlations = my_data.corr(method='pearson')
#print(correlations)

# Count how many patients were prescribed each drug
category_counts = my_data['Drug'].value_counts()

# Bar chart of class distribution — helps spot imbalanced classes
plt.bar(category_counts.index, category_counts.values, color='blue')
plt.xlabel('Drug')
plt.ylabel('Count')
plt.title('Category Distribution')
plt.xticks(rotation=45)
#plt.show()

# Define features (X) and target label (y)
# Drop both the string drug name and its numeric equivalent from features
y = my_data['Drug']
X = my_data.drop(['Drug', 'Drug_num'], axis=1)

# Split into 70% training and 30% test data
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=32)

# Train a decision tree using information gain (entropy) as the split criterion
# max_depth=4 limits tree size to prevent overfitting
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth=4)
drugTree.fit(X_trainset, y_trainset)

# Predict drug labels for the test set
tree_predictions = drugTree.predict(X_testset)

# Accuracy: fraction of test samples correctly classified
print("Decision Trees's Accuracy: ", metrics.accuracy_score(y_testset, tree_predictions))

# Quick tree visualisation without feature labels
plot_tree(drugTree)
#plt.show()

# Full tree visualisation with feature names, class names, colour-coded nodes,
# and rounded boxes — each node shows the split condition and class distribution
plt.figure(figsize=(20, 10))
plot_tree(
    drugTree,
    feature_names=X.columns,
    class_names=drugTree.classes_,
    filled=True,
    rounded=True
)
plt.show()
