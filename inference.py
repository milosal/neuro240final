import torch
import torch.nn as nn
import torch.optim as optim
from hfunction import H
from dfunction import D
import matplotlib.pyplot as plt
import random
import math

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
    
def step(n):
    return 2 * math.floor(n / 100)

def sin(n):
    return math.sin(n)

def pi(n):
    return math.pi * n

def rat(n):
    return (pow(n, 3) / 2) / (pow(n, 2) - 1000001)

def pointil(n):
    if n % 240 == 0:
        return 1
    else:
        return 0


model = SimpleNN(1, hidden_layers, 1)
model.load_state_dict(torch.load('models/model_pointil_100.pth'))
model.eval()

inputs = torch.tensor([[n] for n in range(1, 100001, 3333)], dtype=torch.float32)
predicted_outputs = []
actual_outputs = []

with torch.no_grad():
    for input_tensor in inputs:
        predicted_output = model(input_tensor)
        predicted_outputs.append(predicted_output.item())
        actual_outputs.append(pointil(input_tensor.item()))

plt.figure(figsize=(10, 5))


plt.plot(inputs.numpy(), predicted_outputs, label='Predicted', color='red')
plt.plot(inputs.numpy(), actual_outputs, label='Actual', color='blue')

plt.title('Comparison of Predicted and Actual Values')
plt.xlabel('Input Value')
plt.ylabel('Output Value')
plt.legend()
plt.show()

'''
n = random.randint(240, 240)
input_tensor = torch.tensor([n], dtype=torch.float32)

with torch.no_grad():
    predicted_output = model(input_tensor)
    print(f"Predicted output for input {n}: {predicted_output.item()}")

    print("Real answer: ", pointil(n))
'''