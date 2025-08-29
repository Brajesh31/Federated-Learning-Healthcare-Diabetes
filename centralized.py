# centralized.py
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from model import DiabetesNet

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and preprocess the full dataset
def load_full_data():
    df = pd.read_csv("diabetes.csv")

    # Separate features and labels
    X = df.drop("Outcome", axis=1).values
    y = df["Outcome"].values.reshape(-1, 1)

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create PyTorch datasets
    trainset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                             torch.tensor(y_train, dtype=torch.float32))
    testset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                            torch.tensor(y_test, dtype=torch.float32))

    # Return dataloaders
    return DataLoader(trainset, batch_size=16, shuffle=True), DataLoader(testset, batch_size=16)

# Train the model on full data
def train(model, trainloader, epochs=5):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.BCELoss()

    for epoch in range(epochs):
        running_loss = 0.0
        for X, y in trainloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            output = model(X).view(-1)
            loss = criterion(output, y.view(-1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(trainloader):.4f}")

# Evaluate the model
def test(model, testloader):
    model.eval()
    correct, total, loss = 0, 0, 0.0
    criterion = torch.nn.BCELoss()

    with torch.no_grad():
        for X, y in testloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            output = model(X).view(-1)
            preds = output > 0.5
            correct += (preds == y.view(-1)).sum().item()
            loss += criterion(output, y.view(-1)).item()
            total += y.size(0)

    accuracy = correct / total
    print(f"Centralized Test Accuracy: {accuracy:.4f}, Loss: {loss/total:.4f}")

# Main block
if __name__ == "__main__":
    # Load full dataset
    trainloader, testloader = load_full_data()

    # Load model
    model = DiabetesNet().to(DEVICE)

    # Train and evaluate
    train(model, trainloader, epochs=5)
    test(model, testloader)
