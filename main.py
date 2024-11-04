import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import deeponet

# Load data and convert to PyTorch Tensors
def load_data(train_path, test_path):
    train_data = np.load(train_path, allow_pickle=True)
    test_data = np.load(test_path, allow_pickle=True)

    X_train = (torch.tensor(train_data["X_train0"], dtype=torch.float32),
               torch.tensor(train_data["X_train1"], dtype=torch.float32))
    y_train = torch.tensor(train_data["y_train"], dtype=torch.float32).squeeze()

    X_test = (torch.tensor(test_data["X_test0"], dtype=torch.float32),
              torch.tensor(test_data["X_test1"], dtype=torch.float32))
    y_test = torch.tensor(test_data["y_test"], dtype=torch.float32).squeeze()

    return X_train, y_train, X_test, y_test

def train(model, train_loader, criterion, optimizer, num_epochs):
    losses = []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_branch, batch_trunk, batch_y in train_loader:
            output = model(batch_branch, batch_trunk)
            loss = criterion(output, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    return losses

def evaluate(model, X_test, y_test, criterion):
    model.eval()
    with torch.no_grad():
        test_output = model(X_test[0], X_test[1])
        test_loss = criterion(test_output, y_test)
    print(f"Test Loss: {test_loss.item():.4f}")
    

def plot_loss_curve(losses):
    """
    Plot the curve of loss changing with epoch during training
    :param losses: list of loss values for each epoch
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses) + 1), losses, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.grid()
    plt.show()

# Load data
X_train, y_train, X_test, y_test = load_data("antiderivative_unaligned_train.npz", "antiderivative_unaligned_test.npz")
train_dataset = TensorDataset(X_train[0], X_train[1], y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Model initialization
branch_dim = [X_train[0].shape[1], 40, 40]
trunk_dim = [X_train[1].shape[1], 40, 40]
model = deeponet.DeepONet(branch_dim=branch_dim, trunk_dim=trunk_dim)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model and plot the loss curve
num_epochs = 100
losses = train(model, train_loader, criterion, optimizer, num_epochs)
plot_loss_curve(losses)

# Test the model
evaluate(model, X_test, y_test, criterion)
