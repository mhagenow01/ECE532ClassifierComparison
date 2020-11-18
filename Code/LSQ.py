#!/usr/bin/env python

""" Gets the data from the UCI datasets and puts in a format
to test with the implemented algorithms
 Created: 11/10/2020
"""

__author__ = "Mike Hagenow"

import numpy as np
from PreProcessData import loadFaults

# Calculate the solution to the regularized least squares solution using ridge regression
# e.g., Tikhonov regularization
def lsq(A,b,lam):
    return np.linalg.inv(A.T @ A + lam*np.eye(np.shape(A)[1])) @ A.T @ b


def test():
    X_faults = loadFaults()
    X_train_plus1 = X_faults[0][:int(np.shape(X_faults[0])[0]/2.0),:]
    X_train_minus1 = X_faults[1][:int(np.shape(X_faults[1])[0] / 2.0), :]
    X_train = np.vstack((X_train_plus1,X_train_minus1))
    y_train = np.vstack((np.ones((np.shape(X_train_plus1)[0],1)),-np.ones((np.shape(X_train_minus1)[0],1))))

    for lam in [0.0, 0.1, 0.2, 5000000000.0]:
        # print(np.shape(X_train))
        # print(np.shape(y_train))
        w = lsq(X_train,y_train,lam)
        # print("w:",w)

        X_test_plus1 = X_faults[0][int(np.shape(X_faults[0])[0] / 2.0):, :]
        X_test_minus1 = X_faults[1][int(np.shape(X_faults[1])[0] / 2.0):, :]
        X_test = np.vstack((X_test_plus1,X_test_minus1))
        y_test = np.vstack((np.ones((np.shape(X_test_plus1)[0], 1)), -np.ones((np.shape(X_test_minus1)[0], 1))))

        y_hat_test = np.sign(X_test @ w)
        error_vec = [0 if i[0]==i[1] else 1 for i in np.hstack((y_hat_test,y_test))]
        print("Lam:",lam," Error:",sum(error_vec)/len(y_test))

def testdirect():
    X_faults = loadFaults()
    X_train_plus1 = X_faults[0]
    X_train_minus1 = X_faults[1]
    X_train = np.vstack((X_train_plus1, X_train_minus1))
    y_train = np.vstack((np.ones((np.shape(X_train_plus1)[0], 1)), -np.ones((np.shape(X_train_minus1)[0], 1))))

    # print(np.shape(X_train))
    # print(np.shape(y_train))
    w = lsq(X_train, y_train, 0.0)
    # print("w:",w)

    X_test_plus1 = X_faults[0]
    X_test_minus1 = X_faults[1]
    X_test = np.vstack((X_test_plus1, X_test_minus1))
    y_test = np.vstack((np.ones((np.shape(X_test_plus1)[0], 1)), -np.ones((np.shape(X_test_minus1)[0], 1))))

    y_hat_test = np.sign(X_test @ w)
    error_vec = [0 if i[0] == i[1] else 1 for i in np.hstack((y_hat_test, y_test))]
    print("Lam:", 0.0, " Error:", sum(error_vec) / len(y_test))

if __name__ == "__main__":
    testdirect()




