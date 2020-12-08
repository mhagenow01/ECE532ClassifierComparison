#!/usr/bin/env python

""" Starting script that uses command line arguments to run the evaluations
used in the final report
 Created: 12/5/2020
"""

__author__ = "Mike Hagenow"

import sys
import numpy as np
from PreProcessData import loadFaults
from CrossValidationMulti import crossValidation
from OnevAll import onevall, onevallNN

# Import the classifier models
from LSQ import lsq, wlsq
from lSVM import lsvm, wlsvm
from simpleNN import nn


def noRegFiveFoldClassification():
    X_faults = loadFaults()
    lam = [0.0]

    print("\n\n------------------------")
    print("| Running Cross Validation |")
    print("------------------------")

    crossValidation(X_faults, 5, lam, wlsq)
    crossValidation(X_faults, 5, lam, wlsvm)
    # crossValidation(X_faults, 5, lam, nn)

def trainingAcc():
    X_faults = loadFaults()
    lam = [0.0]

    print("\n\n---------------------------------")
    print("| Running One v All on Training Set |")
    print("-------------------------------------")

    # Run everything with the neural network
    overall_acc = onevallNN(X_faults, X_faults, X_faults)
    print("Neural Network Classification Accuracy: ", overall_acc)

    # Run everything with least-squares
    overall_acc = onevall(X_faults, X_faults, X_faults, lam, wlsq)
    print("wLSQ Classification Accuracy:", overall_acc)

    # Run everything with SVM
    overall_acc = onevall(X_faults, X_faults, X_faults, lam, wlsvm)
    print("SVM Classification Accuracy: ", overall_acc)



def effectRegularizationLSQ_SVM():
    X_faults = loadFaults()
    lams = np.logspace(-6,1,25)

    print("\n\n------------------------------")
    print("| Running Regularization Analysis |")
    print("-----------------------------------")

    results_lsq = []
    results_svm = []

    for ii in range(0,len(lams)):
        print("\n\n------------------------------")
        print("| LAM ",ii," of ",len(lams)," |")
        print("-----------------------------------")
        results_lsq.append(crossValidation(X_faults, 5, [lams[ii]], wlsq))
        results_svm.append(crossValidation(X_faults, 5, [lams[ii]], wlsvm))

    print("\n\n------------------------------")
    print("| Overall Best Classification     |")
    print("-----------------------------------")
    crossValidation(X_faults, 5, lams, wlsq)
    crossValidation(X_faults, 5, lams, wlsvm)

    print("Results for Each Regularizer:")
    print(results_lsq)
    print(results_svm)


def runEvaluations(testname):
    if testname == "genclass":
        noRegFiveFoldClassification()
    elif testname == "regTestl2":
        effectRegularizationLSQ_SVM()
    elif testname == "trainingAcc":
        trainingAcc()
    else:
        print("Test [",testname,"] Not Found")

if __name__ == "__main__":
    testname = sys.argv[1]
    runEvaluations(testname)




