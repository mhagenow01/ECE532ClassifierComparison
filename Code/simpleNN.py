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





def test():
    print("hello")

if __name__ == "__main__":
    test()




