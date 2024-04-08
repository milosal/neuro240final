import torch
import torch.nn as nn
import torch.optim as optim
from hfunction import H
from dfunction import D
import matplotlib.pyplot as plt
import random

hidden_layers = [100, 100]

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size):
        super(SimpleNN, self).__init__()
        layers = [nn.Linear(input_size, hidden_layers[0]), nn.ReLU()]
        for i in range(len(hidden_layers)-1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_layers[-1], output_size))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

model = SimpleNN(1, hidden_layers, 1)
model.load_state_dict(torch.load('models/model_4_25.pth'))
model.eval()

n = random.randint(1, 1000)
input_tensor = torch.tensor([n], dtype=torch.float32)

with torch.no_grad():
    predicted_output = model(input_tensor)
    print(f"Predicted output for input {n}: {predicted_output.item()}")

    print("Real answer: ", H(n))