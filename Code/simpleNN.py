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


# Define the network class for pytorch

# tutorial usage: https://curiousily.com/posts/build-your-first-neural-network-with-pytorch/

class simpleNN(nn.Module):
    def __init__(self,num_features):
        super(simpleNN, self).__init__() # call default network constructor
        self.fc1 = nn.Linear(num_features,50)
        self.fc2 = nn.Linear(50,10)
        self.fc3 = nn.Linear(3,1)
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x


def nn(A,b,lam):
    # create an instance of the network
    net = simpleNN(np.shape(A)[1])
    loss_fxn = torch.nn.functional.mse_loss()
    # create your optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    for ii in range(0,10):

        rand_ind = np.random.choice(np.shape(A)[0])
        network_in = A[rand_ind,:]
        network_target = b[rand_ind]

        # in your training loop:
        optimizer.zero_grad()  # zero the gradient buffers
        output = net.forward(network_in)
        loss = loss_fxn(output, network_target)
        loss.backward()
        optimizer.step()  # Does the update

        #print("Epoch:", ii, "Training Loss: ", np.mean(loss.item()), "Valid Loss: ", np.mean(valid_loss))




def test():

if __name__ == "__main__":
    test()




