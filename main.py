import torch
import math
import numpy as np
import torch.nn as nn
import torch.optim as optim
from dfunction import D
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split

FN_USED = "wonky50"

LR = 0.001

START_TRAIN = 1
END_TRAIN = 10000
START_TEST = 10001
END_TEST = 12000

EPOCHS = 50
PRINT_EVERY = 1
PRINTS_PER_EPOCH = 1

GRAPH_THRESHOLD = 20

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


class HnDataset(Dataset):
    def __init__(self, file_path):
        self.data = torch.load(file_path)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        n, hn = self.data[idx]
        return torch.tensor([n], dtype=torch.float32), torch.tensor([hn], dtype=torch.float32)

def get_dataloaders(batch_size=64, file_path=f'data/{FN_USED}_dataset.pt', split_ratio=0.8):
    dataset = HnDataset(file_path=file_path)
    train_size = int(len(dataset) * split_ratio)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataloader, test_dataloader

train_dataloader, test_dataloader = get_dataloaders(batch_size=64, file_path=f'data/{FN_USED}_dataset.pt', split_ratio=0.8)

def calculate_accuracy(predictions, true_outputs, thresh_percent=0.01):
    thresh = true_outputs * (thresh_percent / 100)
    absolute_errors = torch.abs(predictions - true_outputs)
    accurate_predictions = (absolute_errors <= thresh)
    accuracy = accurate_predictions.float().mean()
    return accuracy

input_size = 1
hidden_layers = [100, 100, 100] 
output_size = 1
model = SimpleNN(input_size, hidden_layers, output_size)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)



epochs = EPOCHS
test_losses = []
train_losses = []
train_maes = []
test_maes = []
log_interval = math.ceil(len(train_dataloader) / PRINTS_PER_EPOCH)

for epoch in range(epochs):
    model.train()
    total_loss = 0
    total_mae = 0
    batch_count = 0

    for inputs, true_outputs in train_dataloader:
        optimizer.zero_grad()
        predicted_outputs = model(inputs)
        loss = criterion(predicted_outputs, true_outputs)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        mae = torch.abs(predicted_outputs - true_outputs).mean() 
        total_mae += mae.item()
        batch_count += 1

        
        if batch_count % log_interval == 0 or batch_count == len(train_dataloader):
            avg_train_loss = round(total_loss / batch_count, 8)
            avg_train_mae = round(total_mae / batch_count, 8)
            train_losses.append(avg_train_loss)
            train_maes.append(avg_train_mae)
            
            model.eval()
            test_loss = 0
            test_mae = 0
            with torch.no_grad():
                for test_inputs, test_true_outputs in test_dataloader:
                    test_outputs = model(test_inputs)
                    test_loss += criterion(test_outputs, test_true_outputs).item()
                    test_mae += torch.abs(test_outputs - test_true_outputs).mean().item()
            avg_test_loss = round(test_loss / len(test_dataloader), 8)
            avg_test_mae = round(test_mae / len(test_dataloader), 8)
            test_losses.append(avg_test_loss)
            test_maes.append(avg_test_mae)
            print(f"After {batch_count/len(train_dataloader)*100:.2f}% of epoch {epoch}, Train Loss: {avg_train_loss}, Train MAE: {avg_train_mae}, Test Loss: {avg_test_loss}, Test MAE: {avg_test_mae}")

save_file_name = f"models/model__{len(hidden_layers)}_{FN_USED}_{EPOCHS}.pth"
torch.save(model.state_dict(), save_file_name)

plt.figure(figsize=(20, 6)) 
plt.subplot(1, 2, 1)  

x_axis = np.linspace(0, epochs, num=len(train_losses))

plt.plot(x_axis, train_losses, label='Training Loss', color='blue')
plt.plot(x_axis, test_losses, label='Testing Loss', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'{FN_USED} Loss Over Epochs')
plt.ylim([0, 1.1 * max(train_losses)])  
plt.legend()

plt.subplot(1, 2, 2) 
plt.plot(x_axis, train_maes, label='Training MAE', color='blue')
plt.plot(x_axis, test_maes, label='Testing MAE', color='orange') 
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.title(f'{FN_USED} MAE Over Epochs')
plt.ylim([0, 1.1 * max(train_maes)])  
plt.legend()

plt.show()