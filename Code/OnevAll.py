#!/usr/bin/env python

""" Implements the one vs all testing philosophy for a set
of training data, regularization choice data, and test data
from a multi-classification problem


 Created: 11/15/2020
"""

__author__ = "Mike Hagenow"

import numpy as np
from PreProcessData import loadFaults
from LSQ import lsq
from lSVM import lsvm

# Calculate one vs all test results
def onevall(X_train,X_reg,X_test,lams,classfxn):

    num_class = len(X_train)
    best_ws = []

    # each type of classification
    for ii in range(0,num_class):

        # Gather the training data for the weights

        # positive label is the 'one'
        X_train_plus1 = X_train[ii]

        # negative label is the 'all' randomly downsampled to the same length as the positive
        X_train_minus1 = np.vstack([X_train[i] for i in range(0,num_class) if (i!=ii)])
        num_rand = np.shape(X_train_plus1)[0]

        indices_rand = np.random.choice(range(0,np.shape(X_train_minus1)[0]),num_rand,replace=False)
        # X_train_minus1 = X_train_minus1[indices_rand,:]

        X_train_temp = np.vstack((X_train_plus1, X_train_minus1))
        y_train_temp = np.vstack((np.ones((np.shape(X_train_plus1)[0], 1)), -np.ones((np.shape(X_train_minus1)[0], 1))))

        # for each regularization value
        min_err_rate = np.inf

        # Create a set for evaluating the regularization parameters
        X_reg_test_plus1 = X_reg[ii]
        X_reg_test_minus1 = np.vstack([X_reg[i] for i in range(0, num_class) if (i != ii)])
        num_rand = np.shape(X_reg_test_plus1)[0]

        indices_rand = np.random.choice(range(0, np.shape(X_reg_test_minus1)[0]), num_rand, replace=False)
        # X_reg_test_minus1 = X_reg_test_minus1[indices_rand, :]
        X_reg_test = np.vstack((X_reg_test_plus1, X_reg_test_minus1))
        y_reg_test = np.vstack(
            (np.ones((np.shape(X_reg_test_plus1)[0], 1)), -np.ones((np.shape(X_reg_test_minus1)[0], 1))))

        for lam in lams:
            w = classfxn(X_train_temp, y_train_temp, lam)

            y_hat_reg = np.sign(X_reg_test @ w)
            # print(np.shape(w))
            # print(np.shape(y_hat_reg))
            error_vec = [0 if i[0] == i[1] else 1 for i in np.hstack((y_hat_reg, y_reg_test))]
            error_rate = sum(error_vec)/len(y_reg_test)

            if(error_rate<min_err_rate):
                best_w = w
                min_err_rate = error_rate

        # Keep track of the best weights for each
        best_ws.append(best_w)

    # With the test set, determine the overall classification error
    correct = 0
    total = 0
    for ii in range(0,num_class):
        total = total + np.shape(X_test[ii])[0]
        for jj in range(0,np.shape(X_test[ii])[0]):
            # for each data point, get the most clear correct value (aka, highest value)
            correct_class = ii
            best_val = -np.inf
            for kk in range(0,num_class):
                temp = X_test[ii][0,:] @ best_ws[kk]
                # print(temp," - ", best_val)
                if temp > best_val:
                    best_val=temp
                    best_class = kk

            # print("BC:",best_class, " CC:",correct_class)
            if(best_class==correct_class):
                correct = correct + 1

    print("correct:",correct)
    print("total:",total)
    print("classification accuracy:",correct/total)

def main():
    X_faults = loadFaults()
    lams = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1, 2, 5, 10.0, 20.0]
    lams = [0.0]

    # Run everything with least-squares
    onevall(X_faults,X_faults,X_faults,lams,lsq)

    # Run everything with SVM
    onevall(X_faults, X_faults, X_faults, lams, lsvm)

if __name__ == "__main__":
    main()




