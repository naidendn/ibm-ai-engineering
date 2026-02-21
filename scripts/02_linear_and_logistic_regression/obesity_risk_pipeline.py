import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def obesity_risk_pipeline(data_path, test_size):
    # Load the obesity level prediction dataset
    data = pd.read_csv(data_path)

    # --- Feature scaling: standardize continuous (float) columns ---
    # StandardScaler brings each continuous feature to mean=0 and std=1,
    # which prevents features with large ranges from dominating the model
    continuous_columns = data.select_dtypes(include=['float64']).columns.tolist()
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data[continuous_columns])
    scaled_df = pd.DataFrame(scaled_features, columns=scaler.get_feature_names_out(continuous_columns))

    # Rebuild the dataframe with scaled continuous columns replacing the originals
    scaled_data = pd.concat([data.drop(columns=continuous_columns), scaled_df], axis=1)

    # --- Feature encoding: one-hot encode categorical (string) columns ---
    # Identify categorical columns but exclude the target variable 'NObeyesdad'
    categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
    if 'NObeyesdad' in categorical_columns:
        categorical_columns.remove('NObeyesdad')

    # drop='first' removes one dummy per category to avoid multicollinearity
    # (known as the "dummy variable trap")
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_features = encoder.fit_transform(data[categorical_columns])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))

    # Build the final feature matrix from scaled continuous + encoded categorical features
    # The target column is never included in X
    X = pd.concat([scaled_df, encoded_df], axis=1)

    # Encode the target labels as integers (e.g. 'Obesity_Type_I' → 0, 'Normal_Weight' → 1, …)
    y = data['NObeyesdad'].astype('category').cat.codes

    # Split into training and test sets; stratify=y preserves class proportions in both splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    # Train a One-vs-All (OvA) logistic regression classifier
    # OvA trains one binary classifier per class: "this class vs. all others"
    model_ova = LogisticRegression(multi_class='ovr', max_iter=1000)
    model_ova.fit(X_train, y_train)

    # Predict the obesity level class for each test sample
    y_pred_ova = model_ova.predict(X_test)

    # Report the percentage of correctly classified test samples
    print("One-vs-All (OvA) Strategy")
    print(f"Accuracy: {np.round(100 * accuracy_score(y_test, y_pred_ova), 2)}%")


# Dataset URL hosted on IBM Cloud Object Storage
file_path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/GkDzb7bWrtvGXdPOfk6CIg/Obesity-level-prediction-dataset.csv"

# Run the pipeline with a 20% test split
obesity_risk_pipeline(file_path, 0.2)
