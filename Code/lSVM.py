#!/usr/bin/env python

""" Computes the weights for a linear SVM binary classifer
 Created: 11/10/2020
"""

__author__ = "Mike Hagenow"

import numpy as np
from PreProcessData import loadFaults

# Calculate the solution to the regularized least squares solution using
# linear support vector machines (SVM)
def lsvm(A,b,lam,tau=None,tol=None):
    if(tol is None):
        tol = 0.001
    if(tau is None):
        U, s, vt = np.linalg.svd(A)
        tau = 0.75/(s[0]**2)

    #print("s0:",s[0])

    # Set up the gradient descent
    w = np.zeros((np.shape(A)[1],))
    not_converged = True

    max_iters = 100
    num_iterations = 0
    while(not_converged and num_iterations<max_iters):
        w_old = w
        w = w-tau*subgrad(w,lam,b,A)
        #print(np.linalg.norm(w-w_old))
        if(np.linalg.norm(w-w_old)<tol):
            not_converged=False
        num_iterations = num_iterations + 1
    return w.reshape((len(w)),1)

def wlsvm(A,w,b,lam,tau=None,tol=None):
    if(tol is None):
        tol = 0.001
    if(tau is None):
        U, s, vt = np.linalg.svd(A)
        tau = 0.75/(s[0]**2)

    # print("s0:",s[0])

    # Set up the gradient descent
    w = np.zeros((np.shape(A)[1],))
    not_converged = True

    max_iters = 100
    num_iterations = 0
    while(not_converged and num_iterations<max_iters):
        w_old = w
        w = w-tau*subgrad(w,lam,b,A)
        #print(np.linalg.norm(w-w_old))
        if(np.linalg.norm(w-w_old)<tol):
            not_converged=False
        num_iterations = num_iterations + 1
    return w.reshape((len(w)),1)

def subgrad(w,lam,y,X):
    subgradval=np.zeros(np.shape(w))
    for ii in range(0,np.shape(X)[0]):
        if((y[ii] * X[ii,:] @ w)<1.0):
            subgradval = subgradval - y[ii] * X[ii,:].T
    subgradval = subgradval + 2*lam*w
    return subgradval


def test():
    X_faults = loadFaults()
    X_train_plus1 = X_faults[0][:int(np.shape(X_faults[0])[0]/2.0),:]
    X_train_minus1 = X_faults[1][:int(np.shape(X_faults[1])[0] / 2.0), :]
    X_train = np.vstack((X_train_plus1,X_train_minus1))
    y_train = np.vstack((np.ones((np.shape(X_train_plus1)[0],1)),-np.ones((np.shape(X_train_minus1)[0],1))))

    for lam in [0.0, 0.1, 0.2]:
        # print(np.shape(X_train))
        # print(np.shape(y_train))
        w = lsvm(X_train,y_train,lam)
        # print("w:",np.shape(w))

        X_test_plus1 = X_faults[0][int(np.shape(X_faults[0])[0] / 2.0):, :]
        X_test_minus1 = X_faults[1][int(np.shape(X_faults[1])[0] / 2.0):, :]
        X_test = np.vstack((X_test_plus1,X_test_minus1))
        y_test = np.vstack((np.ones((np.shape(X_test_plus1)[0], 1)), -np.ones((np.shape(X_test_minus1)[0], 1))))

        y_hat_test = np.sign(X_test @ w)
        print(np.shape(y_test))
        print(np.shape(y_hat_test))
        error_vec = [0 if i[0]==i[1] else 1 for i in np.hstack((y_hat_test,y_test))]
        print("Lam:",lam," Error:",sum(error_vec)/len(y_test))

if __name__ == "__main__":
    test()




