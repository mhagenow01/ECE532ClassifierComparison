#!/usr/bin/env python

""" Gets the data from the UCI datasets and puts in a format
to test with the implemented algorithms
 Created: 11/10/2020
"""

__author__ = "Mike Hagenow"

import numpy as np
from PreProcessData import loadFaults

"""
Calculate the solution to the regularized least squares solution using ridge regression
e.g., Tikhonov regularization

This works on a matrix of features A, vector of labels b, and scalar lamda for regularization
"""
def lsq(A,b,lam):
    return np.linalg.inv(A.T @ A + lam*np.eye(np.shape(A)[1])) @ A.T @ b


"""
Calculate the solution to the regularized least squares solution where each sample
also has an associated wieght which are stored in vector, w, which is used
to perform weighted least squares by converting W to a diagonal matrix

source: https://en.wikipedia.org/wiki/Weighted_least_squares
"""
def wlsq(A,w,b,lam):

    # create diagonal weighting matrix
    W = np.diag(w.flatten())

    # calculate regularized lsq using weighting matrix
    return np.linalg.inv(A.T @ W @ A + lam*np.eye(np.shape(A)[1])) @ A.T @ W @ b


"""
Gradient Descent version fo the least squares

Note: Implemented but not actually used for any of the analysis
"""
def wlsqGD(A,w_samples,b,lam,reg='l2',tol=None, tau=None):
    if (tol is None):
        tol = 0.001
    if (tau is None):
        U, s, vt = np.linalg.svd(A)
        tau = 0.75 / (s[0] ** 2)

    # print("s0:",s[0])

    # Set up the gradient descent
    w = np.zeros((np.shape(A)[1],))
    not_converged = True

    max_iters = 500
    num_iterations = 0
    while (not_converged and num_iterations < max_iters):
        w_old = w
        w = w - tau * wgradlsq(w, lam, b, A, w_samples, reg=reg)
        # print(np.linalg.norm(w-w_old))
        if (np.linalg.norm(w - w_old) < tol):
            not_converged = False
        num_iterations = num_iterations + 1
    return w.reshape((len(w)), 1)

"""
Least squares subgradient calculation either works for l2 regularization (default)
or no regularization
"""
def wgradlsq(w,lam,y,X, w_samples, reg='l2'):
    subgradval=np.zeros(np.shape(w))
    for ii in range(0,np.shape(X)[0]):
        if((y[ii] * X[ii,:] @ w)<1.0):
            subgradval = subgradval - w_samples[ii]*y[ii] * X[ii,:].T
    if reg=='l2':
        subgradval = subgradval + 2*lam*w
    else:
        subgradval = subgradval # no regularization
    return subgradval

"""
Least squares for l1 regularizer using proximal gradient descent
"""
def wlsqPGD(A,w_samples,b,lam,reg='l2',tol=None, tau=None):
    if (tol is None):
        tol = 0.001
    if (tau is None):
        U, s, vt = np.linalg.svd(A)
        tau = 0.75 / (s[0] ** 2)

    # print("s0:",s[0])

    # Set up the gradient descent
    w = np.zeros((np.shape(A)[1],))
    not_converged = True

    max_iters = 500
    num_iterations = 0
    while (not_converged and num_iterations < max_iters):
        w_old = w

        # Step in the gradient direction
        w = w - tau * wgradlsq(w, lam, b, A, w_samples, reg='None')

        # Regularization Step
        for ii in range(0,len(w)):
            w[ii] = (np.abs(w[ii])-(lam*tau/2))*float(np.abs(w[ii])-(lam*tau/2)>0) * np.sign(w[ii])

        if (np.linalg.norm(w - w_old) < tol):
            not_converged = False
        num_iterations = num_iterations + 1
    return w.reshape((len(w)), 1)

"""
Next three are various test functions where the multiclass data is turned into binary classification problems
"""

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
    X_train_plus1 = X_faults[3]
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


def testVall():
    X_faults = loadFaults()
    num_class = len(X_faults)
    ii = 3

    # positive label is the 'one'
    X_train_plus1 = X_faults[ii]

    # negative label is the 'all' randomly downsampled to the same length as the positive
    X_train_minus1 = np.vstack([X_faults[i] for i in range(0, num_class) if (i != ii)])

    X_train_temp = np.vstack((X_train_plus1, X_train_minus1))
    y_train_temp = np.vstack((np.ones((np.shape(X_train_plus1)[0], 1)), -np.ones((np.shape(X_train_minus1)[0], 1))))
    w_train_temp = np.vstack((1.0 / (np.shape(X_train_plus1)[0]) * np.ones((np.shape(X_train_plus1)[0], 1)),
                              1.0 / (np.shape(X_train_minus1)[0]) * np.ones((np.shape(X_train_minus1)[0], 1))))

    w = wlsq(X_train_temp, w_train_temp, y_train_temp, 0.0)
    y_hat_test = np.sign(X_train_temp @ w)
    error_vec = [0 if i[0] == i[1] else 1 for i in np.hstack((y_hat_test, y_train_temp))]
    print("Lam:", 0.0, " Error:", sum(error_vec) / len(y_train_temp))

    print((X_train_temp @ w)[0:10])


if __name__ == "__main__":
    testVall()




