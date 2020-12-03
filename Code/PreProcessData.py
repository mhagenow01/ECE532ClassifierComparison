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

    data = np.loadtxt('./../Data/Faults.csv',delimiter=',')
    #print(np.shape(data))

    # First 27 attributes are the independent variables
    X = data[:,0:27]

    # Normalize the columns of X
    X = X/np.linalg.norm(X,axis=0)

    # Concatenate 1's on the end as an offset feature
    # X = np.hstack((X,np.ones((np.shape(X)[0],1))))

    print(np.shape(X))

    print(np.linalg.matrix_rank(X))

    # Final 7 attributes are 1/0 for the type of fault
    Y = data[:,27:]

    # print(np.shape(X))
    # print(np.shape(Y))
    #
    # # Confirms that each plate has only one type of fault
    # print(np.max(np.sum(Y,axis=1)))

    X_faults = []
    # seven types of faults for list
    for ii in range(0,7):
        X_faults.append(np.zeros((0,27)))

    # Store as the following X_faults is a list of the data for each type of fault
    # Don't need a Y in this case

    for ii in range(0,np.shape(X)[0]):
        fault_ind_temp = np.argmax(Y[ii,:])
        X_faults[fault_ind_temp] = np.vstack((X_faults[fault_ind_temp],X[ii,:]))

    for ii in range(0,7):
        print(ii,"-",np.shape(X_faults[ii]))
    print("---")
    return X_faults

def loadMocap():
    data = np.loadtxt('./../Data/Postures.csv',delimiter=',',skiprows=2)

    # indicator value for missing - use to determine equal feature lengths
    ind_full = [i for i in range(0,np.shape(data)[0]) if data[i,19]!=-999.999]

    data_full = data[ind_full]

    Y = data_full[:,0]

    X_mocap = []

    # six gestures for the mocap data
    for ii in range(0,int(np.max(Y)+1)):
        X_mocap.append(np.zeros((0,27)))

    for ii in range(0,np.shape(data_full)[0]):
        # calculate the features

        # Using 6 markers
        num_markers = 6
        x_vals_temp = data_full[ii][2,5,8,11,14,17]
        y_vals_temp = data_full[ii][3,6,9,12,15,18]
        z_vals_temp = data_full[ii][4,7,10,13,16,19]

        distances =

        # max distance between pts

        # min distance between pts

        # average distance between pts

        # angles and stuff


    # normalize the features

    return X_mocap

if __name__ == "__main__":
    loadFaults()




