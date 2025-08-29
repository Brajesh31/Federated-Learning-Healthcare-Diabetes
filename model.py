# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a simple feedforward neural network for binary classification
class DiabetesNet(nn.Module):
    def __init__(self):
        super(DiabetesNet, self).__init__()
        self.fc1 = nn.Linear(8, 16)    # Input layer (8 features) -> Hidden layer 1
        self.fc2 = nn.Linear(16, 12)   # Hidden layer 1 -> Hidden layer 2
        self.fc3 = nn.Linear(12, 1)    # Hidden layer 2 -> Output layer (1 neuron for binary)

    def forward(self, x):
        x = F.relu(self.fc1(x))        # Activation function: ReLU
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x)) # Output layer with sigmoid for binary classification
        return x
