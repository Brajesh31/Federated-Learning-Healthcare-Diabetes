# utils.py
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and split the diabetes dataset for a specific client
def load_data(client_id, num_clients=2):
    # Read dataset
    df = pd.read_csv("diabetes.csv")

    # Features and labels
    X = df.drop("Outcome", axis=1).values
    y = df["Outcome"].values.reshape(-1, 1)

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train-test split (same for all clients)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Divide training data among clients
    chunk_size = len(X_train) // num_clients
    start = client_id * chunk_size
    end = (client_id + 1) * chunk_size if client_id < num_clients - 1 else len(X_train)

    # Each client gets a portion of the data
    X_client = X_train[start:end]
    y_client = y_train[start:end]

    # Convert to PyTorch DataLoader
    trainset = TensorDataset(torch.tensor(X_client, dtype=torch.float32),
                             torch.tensor(y_client, dtype=torch.float32))
    testset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                            torch.tensor(y_test, dtype=torch.float32))

    return DataLoader(trainset, batch_size=16, shuffle=True), DataLoader(testset, batch_size=16)
