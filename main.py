import torch
import torch.nn as nn
import torch.optim as optim
from hfunction import H
from dfunction import D
import matplotlib.pyplot as plt

START_TRAIN = 1
END_TRAIN = 10000
START_TEST = 10001
END_TEST = 12000

EPOCHS = 40
PRINT_EVERY = 4

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


def generate_data(n):
    input = torch.tensor([n], dtype=torch.float32)
    output = torch.tensor([H(n)], dtype=torch.float32)
    return input, output

input_size = 1
hidden_layers = [64, 256, 256, 64] 
output_size = 1
model = SimpleNN(input_size, hidden_layers, output_size)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# This is the loop to train the model
epochs = EPOCHS
test_losses = []
train_losses = []
for epoch in range(epochs):
    total_loss = 0
    for n in range(START_TRAIN, END_TRAIN + 1):  
        input, true_output = generate_data(n)
        optimizer.zero_grad()
        predicted_output = model(input)
        loss = criterion(predicted_output, true_output)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    train_losses.append(total_loss / (END_TRAIN - START_TRAIN + 1))
    # Test the model every epoch!
    with torch.no_grad():
        test_loss = 0
        for n in range(START_TEST, END_TEST + 1): 
            input, true_output = generate_data(n)
            predicted_output = model(input)
            loss = criterion(predicted_output, true_output)
            test_loss += loss.item()
        test_losses.append(test_loss / (END_TEST - START_TEST + 1))
    if epoch % PRINT_EVERY == 0:
        print(f"Epoch {epoch}, Train Loss: {total_loss / (END_TRAIN - START_TRAIN + 1)}")
        print(f"Test Loss: {test_loss / (END_TEST - START_TEST + 1)}")

save_file_name = f"model_{len(hidden_layers)}_{EPOCHS}.pth"
torch.save(model, save_file_name)

plt.figure(figsize=(10, 6))
plt.plot(range(epochs), train_losses, label='Training Loss')
plt.plot(range(epochs), test_losses, label='Testing Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss Over Epochs')
plt.legend()
plt.show()
