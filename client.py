# client.py
import flwr as fl
import torch
from collections import OrderedDict
from model import DiabetesNet
from utils import load_data

# Set device (GPU if available)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configuration
NUM_CLIENTS = 2  # Total number of simulated clients
CLIENT_ID = int(input("Enter client ID (0 or 1): "))  # Ask user to specify client index

# Initialize model and data
model = DiabetesNet().to(DEVICE)
trainloader, testloader = load_data(CLIENT_ID, NUM_CLIENTS)

# Get model parameters as NumPy arrays
def get_parameters(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

# Set model parameters from NumPy arrays
def set_parameters(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

# Local training function
def train(net, trainloader, epochs=1):
    net.train()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    loss_fn = torch.nn.BCELoss()  # Binary Cross Entropy Loss

    for _ in range(epochs):
        for X, y in trainloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            output = net(X).view(-1)
            loss = loss_fn(output, y.view(-1))
            loss.backward()
            optimizer.step()

# Evaluation function
def test(net, testloader):
    net.eval()
    correct, total, loss = 0, 0, 0.0
    loss_fn = torch.nn.BCELoss()

    with torch.no_grad():
        for X, y in testloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            outputs = net(X).view(-1)
            preds = outputs > 0.5  # Threshold for binary classification
            correct += (preds == y.view(-1)).sum().item()
            loss += loss_fn(outputs, y.view(-1)).item()
            total += y.size(0)

    return loss / total, correct / total

# Define Flower client class
class DiabetesClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return get_parameters(model)

    def fit(self, parameters, config):
        set_parameters(model, parameters)
        train(model, trainloader, epochs=1)
        return get_parameters(model), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        set_parameters(model, parameters)
        loss, accuracy = test(model, testloader)
        print(f"[Client {CLIENT_ID}] Eval -> Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        return loss, len(testloader.dataset), {"accuracy": accuracy}

# Start the client
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=DiabetesClient())
