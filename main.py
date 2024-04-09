import torch
import torch.nn as nn
import torch.optim as optim
from dfunction import D
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split

FN_USED = "pointil"

LR = 0.001

START_TRAIN = 1
END_TRAIN = 10000
START_TEST = 10001
END_TEST = 12000

EPOCHS = 100
PRINT_EVERY = 5

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
hidden_layers = [100, 100] 
output_size = 1
model = SimpleNN(input_size, hidden_layers, output_size)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)


epochs = EPOCHS
test_losses = []
train_losses = []
train_maes = []
test_maes = []
for epoch in range(epochs):
    model.train()
    total_loss = 0
    total_mae = 0 
    for inputs, true_outputs in train_dataloader:
        optimizer.zero_grad()
        predicted_outputs = model(inputs)
        loss = criterion(predicted_outputs, true_outputs)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        mae = torch.abs(predicted_outputs - true_outputs).mean() 
        total_mae += mae.item()

    avg_train_loss = total_loss / len(train_dataloader)
    avg_train_mae = total_mae / len(train_dataloader) 
    train_losses.append(avg_train_loss)
    train_maes.append(avg_train_mae) 

    #test
    model.eval()
    test_loss = 0
    total_mae = 0
    with torch.no_grad():
        for inputs, true_outputs in test_dataloader:
            predicted_outputs = model(inputs)
            loss = criterion(predicted_outputs, true_outputs)
            test_loss += loss.item()
            mae = torch.abs(predicted_outputs - true_outputs).mean()
            total_mae += mae.item()

    avg_test_loss = test_loss / len(test_dataloader)
    avg_test_mae = total_mae / len(test_dataloader)  
    test_losses.append(avg_test_loss)
    test_maes.append(avg_test_mae)
    
    if epoch % PRINT_EVERY == 0:
        print(f"Epoch {epoch}, Train Loss: {avg_train_loss}, Train MAE: {avg_train_mae}\nTest Loss: {avg_test_loss}, Test MAE: {avg_test_mae}")

save_file_name = f"models/model_{FN_USED}_{EPOCHS}.pth"
torch.save(model.state_dict(), save_file_name)

plt.figure(figsize=(20, 6)) 
plt.subplot(1, 2, 1)  
plt.plot(range(epochs), train_losses, label='Training Loss', color='blue')
plt.plot(range(epochs), test_losses, label='Testing Loss', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'{FN_USED} Loss Over Epochs')
plt.ylim([0, 1.3 * train_losses[4]])  
plt.legend()

plt.subplot(1, 2, 2) 
plt.plot(range(epochs), train_maes, label='Training MAE', color='blue')
plt.plot(range(epochs), test_maes, label='Testing MAE', color='orange') 
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.title(f'{FN_USED} MAE Over Epochs')
plt.ylim([0, 1.3 * train_maes[3]])
plt.legend()

plt.show()