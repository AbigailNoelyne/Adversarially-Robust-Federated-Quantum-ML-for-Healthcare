import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch

def preprocess_data(data):
    data = data.dropna()
    target_column = 'Stage'

    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Encode target labels (1, 2, 3) to (0, 1, 2) - PyTorch's CrossEntropyLoss expects class labels to start from 0
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    X = pd.get_dummies(X, drop_first=True)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y

def split_data_among_clients(X, y, num_clients):
    client_data = []
    data_per_client = len(X) // num_clients
    for i in range(num_clients):
        start_idx = i * data_per_client
        end_idx = (i + 1) * data_per_client if i != num_clients - 1 else len(X)
        client_data.append((X[start_idx:end_idx], y[start_idx:end_idx]))
    return client_data

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor