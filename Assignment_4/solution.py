import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

class CustomDataset():
    """Custom dataset Loader for California Housing
    
    : Dataset <tuple>: tuple of two numpy arrays where index 0 contains the 
                    features (fetch_california_housing.data) and index 1 
                    contains the target values (fetch_california_housing.target)
     """
    def __init__(self, Dataset:tuple):
        """Separate the input"""
        self.data = Dataset[0]
        self.target = Dataset[1]

    def __len__(self):
        """Returns the lenght of the dataset"""
        return len(self.data)

    def __getitem__(self, idx):
        """returns at index idx"""
        return self.data[idx], self.target[idx]


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader)
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            #correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    print("test cummulative loss: ", test_loss)
    test_loss /= num_batches
    #correct /= size
    print(f"Avg MSE loss : {test_loss:>8f} \n")