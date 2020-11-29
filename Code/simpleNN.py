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
        self.fc1 = torch.nn.Linear(num_features,100)
        self.fc2 = torch.nn.Linear(100,1)
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x


def nn(A,b,lam):
    # create an instance of the network
    net = SimpleNN(np.shape(A)[1])
    batch_size = 1400

    # create the optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.001)

    # Run epochs

    init_weights = copy.deepcopy(net.fc1.weight.data)

    for ii in range(0,1000):
        rand_ind = np.random.choice(np.shape(A)[0],batch_size)
        network_in = A[rand_ind,:]
        network_target = b[rand_ind]

        optimizer.zero_grad()  # zero the gradient buffers
        output = net.forward(torch.from_numpy(network_in).float())
        loss = torch.nn.functional.mse_loss(output, torch.from_numpy(network_target).float())
        loss.backward()
        optimizer.step()  # Does the update
        print("Epoch:", ii, "Training Loss: ",loss.item())
        # print(net.fc1.weight.grad)

    print("Testing the Network!!")
    # print(net.forward(torch.from_numpy(A).float()).detach().numpy())
    # print(torch.from_numpy(A).float())
    print(init_weights)
    final_weights = net.fc1.weight.data
    print(final_weights)
    val = net.forward(torch.from_numpy(A).float()).detach().numpy()
    print(val)

    # error_vec = [0 if i[0] == i[1] else 1 for i in np.hstack((, b))]
    # print("Error:",sum(error_vec) / len(b))
    # print(b)


def test():
    X_faults = loadFaults()
    num_class = len(X_faults)
    ii = 3

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
    test()




