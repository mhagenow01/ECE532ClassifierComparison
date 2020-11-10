#!/usr/bin/env python

""" Gets the data from the UCI datasets and puts in a format
to test with the implemented algorithms
 Created: 11/10/2020
"""

__author__ = "Mike Hagenow"

import numpy as np

def loadFaults():
    # define that space yo yo
    # todo: change to package directory

    data = np.loadtxt('./Data/Faults.csv',delimiter=',')
    print(np.shape(data))

    # First 27 attributes are the independent variables
    X = data[:,0:27]

    # Final 7 attributes are 1/0 for the type of fault
    Y = data[:,27:]

    print(np.shape(X))
    print(np.shape(Y))

    # Confirms that each plate has only one type of fault
    print(np.max(np.sum(Y,axis=1)))

    X_faults = []
    for ii in range(0,7):
        X_faults.append(np.zeros((0,27)))

    # Store as the following X_faults is a list of the data for each type of fault
    # Don't need a Y in this case

    for ii in range(0,np.shape(X)[0]):
        fault_ind_temp = np.argmax(Y[ii,:])
        X_faults[fault_ind_temp] = np.vstack((X_faults[fault_ind_temp],X[ii,:]))

    for ii in range(0,7):
        print(ii,"-",np.shape(X_faults[ii]))

    return X_faults

if __name__ == "__main__":
    loadFaults()




