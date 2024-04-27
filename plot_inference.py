import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math


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


def plot_model_predictions(model, function, range_min, range_max, step_size):
    model.eval()  
    inputs = torch.tensor([[n] for n in range(range_min, range_max + 1, step_size)], dtype=torch.float32)
    predicted_outputs = []
    actual_outputs = []

    with torch.no_grad():
        for input_tensor in inputs:
            predicted_output = model(input_tensor)
            predicted_outputs.append(predicted_output.item())
            actual_outputs.append(function(input_tensor.item()))

    plt.figure(figsize=(10, 5))
    plt.plot(inputs.numpy(), predicted_outputs, label='Predicted', color='red')
    plt.plot(inputs.numpy(), actual_outputs, label='Actual', color='blue')
    plt.title('Comparison of Predicted and Actual Values')
    plt.xlabel('Input Value')
    plt.ylabel('Output Value')
    plt.legend()
    plt.show()

def pointil(n):
    if n % 240 == 0:
        return 1
    else:
        return 0


model = SimpleNN(1, [100, 100, 100], 1)
model.load_state_dict(torch.load('models/model_pointil_100.pth'))

plot_model_predictions(model, pointil, 1, 100000, 3333)