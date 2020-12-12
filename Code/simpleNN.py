#!/usr/bin/env python

""" Implements a basic binary classifier using a shallow neural
    network architecture. Input/Output data is stored in matrix
    format to allow for reuse with other methods

Created: 11/25/2020
"""

__author__ = "Mike Hagenow"

import numpy as np
from PreProcessData import loadFaults
import torch
from torch import nn, optim
import torch.nn.functional as F
import copy


# Define the network class for pytorch

# tutorial usage: https://curiousily.com/posts/build-your-first-neural-network-with-pytorch/
# https://www.analyticsvidhya.com/blog/2019/01/guide-pytorch-neural-networks-case-studies/

class SimpleNN(nn.Module):
    def __init__(self,num_features):
        super(SimpleNN, self).__init__() # call default network constructor
        self.fc1 = torch.nn.Linear(num_features,1000)
        self.fc2 = torch.nn.Linear(1000,1000)
        self.fc3 = torch.nn.Linear(1000,1)
    def forward(self,x):
        x = torch.relu(self.fc1(x))
        # x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

class MutliClassNN(nn.Module):
    def __init__(self,num_features,num_labels):
        super(MutliClassNN, self).__init__() # call default network constructor
        self.fc1 = torch.nn.Linear(num_features,1000)
        # self.fc2 = torch.nn.Linear(1000,1000)
        self.fc3 = torch.nn.Linear(1000,num_labels)
    def forward(self,x):
        x = torch.relu(self.fc1(x))
        # x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

def nn(A,b, epochs=5000):
    # create an instance of the network
    net = SimpleNN(np.shape(A)[1])

    # create the optimizer
    optimizer = optim.Adam(net.parameters(), lr=0.01)

    # Run epochs

    init_weights = copy.deepcopy(net.fc1.weight.data)

    for ii in range(0,epochs):
        network_in = A
        network_target = b

        optimizer.zero_grad()  # zero the gradient buffers
        output = net.forward(torch.Tensor(network_in))
        loss = torch.nn.functional.mse_loss(output, torch.Tensor(network_target))
        loss.backward()
        optimizer.step()  # Does the update
        # if ii%1000==0:
        #     print("Epoch:", ii, "Training Loss: ",loss.item())
        # print(net.fc1.weight.grad)

    # print("Testing the Network!!")
    # # print(net.forward(torch.from_numpy(A).float()).detach().numpy())
    # # print(torch.from_numpy(A).float())
    # print(init_weights)
    # final_weights = net.fc1.weight.data
    # print(final_weights)
    val = net.forward(torch.from_numpy(A).float()).detach().numpy()
    val_bool = np.array([0.0 if i<0.5 else 1.0 for i in val]).reshape((len(val),1))

    # print(torch.Tensor(network_target))
    # print(network_target)

    error_vec = [0 if i[0] == i[1] else 1 for i in np.hstack((val_bool,b))]
    # print("Error:",sum(error_vec) / len(b))

    return copy.deepcopy(net)


def nnMultliClass(As, epochs=5000):
    # create an instance of the network
    net = MutliClassNN(np.shape(As[0])[1],len(As))

    # create the optimizer
    optimizer = optim.Adam(net.parameters(), lr=0.01)

    # Format the data for pytorch
    A = As[0]
    B = np.zeros((np.shape(A)[0],len(As)))
    B[:,0] = 1.0

    for ii in range(1,len(As)):
        A = np.concatenate((A,As[ii]),axis=0)
        B_temp = np.zeros((np.shape(As[ii])[0],len(As)))
        B_temp[:,ii] = 1.0
        B = np.concatenate((B,B_temp),axis=0)

    init_weights = copy.deepcopy(net.fc1.weight.data)

    # run the epochs to train the network
    for ii in range(0, 5000):
        network_in = A
        network_target = B

        optimizer.zero_grad()  # zero the gradient buffers
        output = net.forward(torch.Tensor(network_in))
        loss = torch.nn.functional.mse_loss(output, torch.Tensor(network_target))
        loss.backward()
        optimizer.step()  # Does the update
        # if ii%100==0:
        #     print("Epoch:", ii, "Training Loss: ",loss.item())
        # print(net.fc1.weight.grad)

    return copy.deepcopy(net)

def compareNN():
    # create a hold-out set with approx 20 percent of the data
    X_faults = loadFaults()
    num_class = len(X_faults)


def test_multi():
    X_faults = loadFaults()
    net = nnMultliClass(X_faults)

    temp = net.forward(torch.Tensor(X_faults[3][5, :]))
    print(temp.detach().numpy())
    print(np.argmax(temp.detach().numpy()))

def test():
    X_faults = loadFaults()
    num_class = len(X_faults)
    ii = 6

    # positive label is the 'one'
    X_train_plus1 = X_faults[ii]

    # negative label is the 'all' randomly downsampled to the same length as the positive
    X_train_minus1 = np.vstack([X_faults[i] for i in range(0, num_class) if (i != ii)])

    X_train_temp = np.vstack((X_train_plus1, X_train_minus1))
    y_train_temp = np.vstack((np.ones((np.shape(X_train_plus1)[0], 1)), np.zeros((np.shape(X_train_minus1)[0], 1))))
    w_train_temp = np.vstack((1.0 / (np.shape(X_train_plus1)[0]) * np.ones((np.shape(X_train_plus1)[0], 1)),
                              1.0 / (np.shape(X_train_minus1)[0]) * np.ones((np.shape(X_train_minus1)[0], 1))))

    nn(X_train_temp,y_train_temp,0.0)

if __name__ == "__main__":
    test_multi()




