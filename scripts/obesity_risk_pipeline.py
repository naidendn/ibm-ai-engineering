import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def obesity_risk_pipeline(data_path, test_size):
    data = pd.read_csv(data_path)

    continuous_columns = data.select_dtypes(include=['float64']).columns.tolist()
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data[continuous_columns])
    scaled_df = pd.DataFrame(scaled_features, columns=scaler.get_feature_names_out(continuous_columns))
    scaled_data = pd.concat([data.drop(columns=continuous_columns), scaled_df], axis=1)

    categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
    if 'NObeyesdad' in categorical_columns:
        categorical_columns.remove('NObeyesdad')

    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_features = encoder.fit_transform(data[categorical_columns])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))

    X = pd.concat([scaled_df, encoded_df], axis=1)
    y = data['NObeyesdad'].astype('category').cat.codes

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    model_ova = LogisticRegression(multi_class='ovr', max_iter=1000)
    model_ova.fit(X_train, y_train)

    y_pred_ova = model_ova.predict(X_test)

    print("One-vs-All (OvA) Strategy")
    print(f"Accuracy: {np.round(100 * accuracy_score(y_test, y_pred_ova), 2)}%")

file_path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/GkDzb7bWrtvGXdPOfk6CIg/Obesity-level-prediction-dataset.csv"
obesity_risk_pipeline(file_path, 0.2)






