import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from tqdm.notebook import tqdm

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


class MyNetwork(nn.Module):
    def __init__(self, lr=0.0001):
        super(MyNetwork, self).__init__()
        self.learning_rate = lr

        self.network = nn.Sequential(
            nn.Linear(32*32*3, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        return self.network(x)


def train_loop(dataloader, model, loss_fn, optimizer):
    print("------- Training Loop -------")
    # size = len(dataloader.dataset)
    size = len(dataloader)
    running_loss = 0
    for n_batch, mini_batch in (pbar := tqdm(enumerate(dataloader, 1),
                                           total=len(dataloader))):
        inputs,labels = mini_batch
        optimizer.zero_grad()

        # Compute prediction and loss
        pred = model(inputs)
        loss = loss_fn(pred, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        pbar.set_description(f"Loss: {running_loss / n_batch:.4f}")
        pbar.refresh()
        # if batch % 10 == 0:
        #     loss, current = loss.item(), batch * len(X)
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    print("Traning Loss:", running_loss/size)
    return running_loss/size
    


def test_loop(dataloader, model, loss_fn):
    print("------- Test Loop --------")
    size = len(dataloader)
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            #correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    #correct /= size
    print(f"Avg MSE loss : {test_loss:>8f} \n")
    return test_loss