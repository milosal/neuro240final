import torch
import torch.nn as nn
import torch.optim as optim
from hfunction import H
from dfunction import D
import matplotlib.pyplot as plt
import random

hidden_layers = [8, 16, 8]

class ResidualBlock(nn.Module):
    def __init__(self, input_size, output_size, dropout_probability=0.5):
        super(ResidualBlock, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.dropout = nn.Dropout(p=dropout_probability)
        self.relu = nn.ReLU(inplace=False)  # Ensure ReLU is not in-place
        # Ensure input_size == output_size for residual connection
        if input_size != output_size:
            self.adjust_dimensions = nn.Linear(input_size, output_size)
        else:
            self.adjust_dimensions = None

    def forward(self, x):
        identity = x
        
        out = self.linear(x)
        out = self.dropout(out)
        out = self.relu(out)

        if self.adjust_dimensions is not None:
            identity = self.adjust_dimensions(identity)

        out = out + identity  # Ensure this addition isn't marked as in-place
        out = self.relu(out)  # Make sure ReLU here isn't in-place
        return out

class EnhancedNN(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, dropout_probability=0.5):
        super(EnhancedNN, self).__init__()
        self.layers = nn.ModuleList()
        # Create the first residual block separately to handle input size
        self.layers.append(ResidualBlock(input_size, hidden_layers[0], dropout_probability))

        # Create subsequent residual blocks based on hidden layers
        for i in range(1, len(hidden_layers)):
            self.layers.append(ResidualBlock(hidden_layers[i-1], hidden_layers[i], dropout_probability))

        # Final layer - no dropout here
        self.final = nn.Linear(hidden_layers[-1], output_size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.final(x)
        return x

model = EnhancedNN(1, hidden_layers, 1, dropout_probability=0.5)
model.load_state_dict(torch.load('models/model_3_50.pth'))
model.eval()

n = random.randint(1, 1000)
input_tensor = torch.tensor([n], dtype=torch.float32)

with torch.no_grad():
    predicted_output = model(input_tensor)
    print(f"Predicted output for input {n}: {predicted_output.item()}")

    print("Real answer: ", H(n))