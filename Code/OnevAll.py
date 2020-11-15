#!/usr/bin/env python

""" Implements the one vs all testing philosophy for a set
of training data, regularization choice data, and test data
from a multi-classification problem


 Created: 11/15/2020
"""

__author__ = "Mike Hagenow"

import numpy as np
from Code.PreProcessData import loadFaults
from Code.LSQ import lsq

# Calculate one vs all test results
def onevall(X_train,X_reg,X_test):

    num_class = len(X_train)

    # each type of classification
    for ii in range(0,num_class):

        # Gather the training data for the weights

        # positive label is the 'one'
        X_train_plus1 = X_train[ii]

        # negative label is the 'all' randomly downsampled to the same length as the positive
        X_train_minus1 = np.vstack([X_train[i] for i in range(0,num_class) if (i!=ii)])
        num_rand = np.shape(X_train_plus1)[0]
        indices_rand = np.random.choice(range(0,np.shape(X_train_minus1)[0]),num_rand,replace=False)
        X_train_minus1 = X_train_minus1[indices_rand,:]

        X_train = np.vstack((X_train_plus1, X_train_minus1))
        y_train = np.vstack((np.ones((np.shape(X_train_plus1)[0], 1)), -np.ones((np.shape(X_train_minus1)[0], 1))))

        for lam in [0.0, 0.1, 0.2, 5000000000.0]:
            # print(np.shape(X_train))
            # print(np.shape(y_train))
            w = lsq(X_train, y_train, lam)
            # print("w:",w)

            X_test_plus1 = X_faults[0][int(np.shape(X_faults[0])[0] / 2.0):, :]
            X_test_minus1 = X_faults[1][int(np.shape(X_faults[1])[0] / 2.0):, :]
            X_test = np.vstack((X_test_plus1, X_test_minus1))
            y_test = np.vstack((np.ones((np.shape(X_test_plus1)[0], 1)), -np.ones((np.shape(X_test_minus1)[0], 1))))

            y_hat_test = np.sign(X_test @ w)
            error_vec = [0 if i[0] == i[1] else 1 for i in np.hstack((y_hat_test, y_test))]
            print("Lam:", lam, " Error:", sum(error_vec) / len(y_test))

        print("SH:",np.shape(X_train_plus1))
        print(np.shape(X_train_minus1))


def main():
    X_faults = loadFaults()
    onevall(X_faults,X_faults,X_faults)

if __name__ == "__main__":
    loadFaults()




