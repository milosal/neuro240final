import torch
import torch.nn as nn
import torch.optim as optim
from hfunction import H
from dfunction import D
import matplotlib.pyplot as plt

model = torch.load("model_4_40.pth")
model.eval()

n = 150
input_tensor = torch.tensor([n], dtype=torch.float32)

with torch.no_grad():
    predicted_output = model(input_tensor)
    print(f"Predicted output for input {n}: {predicted_output.item()}")