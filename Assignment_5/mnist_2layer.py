import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.utils.data import Subset

train_dataset = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = dsets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

input_dim = 784
output_dim = 10


class MyNetwork(torch.nn.Module):
    def __init__(self, input_dim, H1, output_dim):
        super(MyNetwork, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, H1)
        self.linear2 = torch.nn.Linear(H1, output_dim)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        outputs = self.linear2(x)
        return outputs


def run_nn(batch_size=200, lr_rate=0.1, hidden_size=100, max_iter=20, epochs=1):
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    model = MyNetwork(input_dim, hidden_size, output_dim)

    criterion = torch.nn.CrossEntropyLoss()  # computes softmax and then the cross entropy

    optimizer = torch.optim.SGD(model.parameters(), lr=lr_rate)

    _iter = 0

    accuracies = []

    for epoch in range(int(epochs)):
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images.view(-1, 28 * 28))
            labels = Variable(labels)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            # read out gradient and update weights
            optimizer.step()

            _iter += 1
            if _iter % 1 == 0:
                # calculate Accuracy
                correct = 0
                total = 0
                for images, labels in test_loader:
                    images = Variable(images.view(-1, 28 * 28))
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    # for gpu, bring the predicted and labels back to cpu fro python operations to work
                    correct += (predicted == labels).sum()
                accuracy = 100 * correct / total
                accuracies.append(accuracy)
                print("Iteration: {}  Loss: {}  Accuracy: {} ".format(_iter, loss.item(), accuracy))
                if _iter == max_iter:
                    return accuracies
