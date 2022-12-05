# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 11:55:52 2022

@author: bened
"""

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils

############################
#### Data preprocessing ####
############################

housing = fetch_california_housing()

# Normalize the data
scaler = StandardScaler()
data = scaler.fit_transform(housing.data, housing.target)

# Split into train and test data
X_train, X_test, y_train, y_test = train_test_split(data,housing.target,
                                                    train_size= 1000, test_size = 100, 
                                                    random_state = 38)    

X_train = torch.tensor(X_train).type(torch.float32)
X_test = torch.tensor(X_test).type(torch.float32)
y_train = torch.tensor(y_train).type(torch.float32) # shape [1000]
y_train = torch.reshape(y_train,(1000,1)) # reshaped to [1000,1]
y_test = torch.tensor(y_test).type(torch.float32) # shape [100]
y_test = torch.reshape(y_test,(100,1)) # reshaped to [100,1]

#############################################
#### Creating Data sets and Data loaders ####
#############################################

train = data_utils.TensorDataset(X_train, y_train)
train_loader = data_utils.DataLoader(train, batch_size=10)

test = data_utils.TensorDataset(X_test, y_test)
test_loader = data_utils.DataLoader(test, batch_size=10)

#train_features, train_labels = next(iter(train_loader))
#print(f"Feature batch shape: {train_features.size()}")
#print(f"Labels batch shape: {train_labels.size()}")

##############################
#### Defining model class ####
##############################

class Net(nn.Module):
    
    def __init__(self):
        super(Net,self).__init__()
        self.layer1 = nn.Linear(8,32) # Input to hidden
        self.out = nn.Linear(32,1) # Hidden to output
        
    def forward(self, x): # x= feature vector
        x = F.relu(self.layer1(x)) # activation function on input layer
        x = self.out(x)
        return x

########################
#### Model instance ####
########################
    
net = Net()
#print(list(net.parameters()))

#########################
#### Hyperparameters ####
#########################

learning_rate = 10**-3
epochs = 20
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr =learning_rate)


#######################################
#### Defining train and test loops ####
#######################################

def train_loop(dataloader,model,loss_fn,optimizer):
    print('---------- Training losses ----------')
    batch_total = len(dataloader)
    for batch,(X,y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (batch+1) % 10 == 0:
            print(f'Batch [{batch+1}/{batch_total}] | Loss: {float(loss)}')


def test_loop(dataloader,model,loss_fn):
    print('---------- Test losses ----------')
    batch_total = len(dataloader)
    total_loss = 0
    with torch.no_grad():
        for batch,(X,y) in enumerate(dataloader):
            pred = model(X)
            loss = loss_fn(pred,y)
            total_loss += loss
            print(f'Batch [{batch+1}/{batch_total}] | Loss {loss}')
    print(f'## Average loss {total_loss/len(dataloader.dataset)} ##')
        



##################################
#### Run train and test loops ####
##################################

def run_training():
    for e in range(epochs):
        print(f"######## Epoch {e+1} ########")
        print('-----------------------------------------')
        train_loop(train_loader,net,loss_fn,optimizer)
        print('\n')
        test_loop(test_loader,net,loss_fn)
        print("-----------------------------------------\n")