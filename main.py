import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from hfunction import H
from dfunction import D
import matplotlib.pyplot as plt

LR = 0.005

START_TRAIN = 1
END_TRAIN = 10000
START_TEST = 10001
END_TEST = 12000

EPOCHS = 200
PRINT_EVERY = 5

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

def generate_data(n):
    input = torch.tensor([n], dtype=torch.float32)
    output = torch.tensor([H(n)], dtype=torch.float32)
    return input, output

input_size = 1
hidden_layers = [8, 16, 8]
output_size = 1
model = EnhancedNN(input_size, hidden_layers, output_size, dropout_probability=0.5)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

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

save_file_name = f"models/model_{len(hidden_layers)}_{EPOCHS}.pth"
torch.save(model.state_dict(), save_file_name)

plt.figure(figsize=(10, 6))
plt.plot(range(epochs), train_losses, label='Training Loss')
plt.plot(range(epochs), test_losses, label='Testing Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss Over Epochs')
plt.legend()
plt.show()


# CHANGE DATA TO TAKE RANDOM DRAWS,
# FIGURE OUT A WAY TO SPLIT THAT WITH TEST

# ALSO TEST THE BEST POST POSSIBLE 'RETURN SAME NUM EVERY TIME'
# ... VALUE TO SEE IF IT LEARNS ANYTHING BETTER THAN THAT